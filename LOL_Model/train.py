import argparse
import os
import os.path as osp
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import sys
from config import config
from utils import common, dataloader, solver, model_opr
from dataloader import NTIRE
from unet_model.unet_lap import UNet
from validate import validate

torch.backends.cudnn.enabled = False


def init_dist(local_rank):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    print('local_rank', local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')
    dist.barrier()


def iteration_enhancement(ins_temporal, gsmooth):
    b, n, c, h, w = ins_temporal.size()
    ins = ins_temporal.view(b * n, c, h, w)
    out = ins.clone()
    for i in range(2):
        out_smooth = gsmooth(out)
        dog = ins - out_smooth
        out = out + dog
    out = torch.clamp(out, 0, 1.0)
    out = out.view(b, n, c, h, w)
    return out


class Model_Train:
    def __init__(self, args= None):
        self.rank = 0
        self.num_gpu = 1
        self.is_distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.num_gpu = int(os.environ['WORLD_SIZE'])
            self.is_distributed = self.num_gpu > 1
        if self.is_distributed:
            rank = args.local_rank
            init_dist(rank)
        common.init_random_seed(config.DATASET.SEED + self.rank)
        model_name = config.model_version
        # set up dirs and log
        exp_dir, cur_dir = osp.split(osp.split(osp.realpath(__file__))[0])
        root_dir = osp.split(exp_dir)[0]

        self.model_log_root_dir = osp.join(root_dir, 'logs', cur_dir)
        common.mkdir(self.model_log_root_dir)

        self.log_dir = osp.join(self.model_log_root_dir, model_name)
        self.model_dir = osp.join(self.log_dir, 'models')
        self.solver_dir = osp.join(self.log_dir, 'solvers')
        if self.rank <= 0:
            common.mkdir(self.log_dir)
            common.mkdir(self.model_dir)
            common.mkdir(self.solver_dir)

            self.save_dir = osp.join(self.log_dir, 'saved_imgs')
            common.mkdir(self.save_dir)
            tb_dir = osp.join(self.log_dir, 'tb_log')
            self.tb_writer = SummaryWriter(tb_dir)
            common.setup_logger('base', self.log_dir, 'train', level=logging.INFO, screen=True, to_file=True)
            self.logger = logging.getLogger('base')

    def load_dataset(self):
        train_dataset = NTIRE(split='train', patch_width=64, path_height=64, rank=self.rank)
        self.train_loader = dataloader.train_loader(train_dataset, config, rank=self.rank, is_dist=self.is_distributed)
        if self.rank <= 0:
            print('---per gpu batch size', config.DATALOADER.IMG_PER_GPU)
            val_dataset = NTIRE(split='test', rank=0)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8,
                                                          pin_memory=True)

    def load_model(self):
        self.model = UNet()
        print("model have {:.3f}M paramerters in total".format(
            sum(x.numel() for x in self.model.parameters()) / 1000000.0))
        if config.CONTINUE_ITER:
            model_path = osp.join(self.model_dir, '%d.pth' % config.CONTINUE_ITER)
            if self.rank <= 0:
                self.logger.info('[Loading] Iter: %d' % config.CONTINUE_ITER)
            model_opr.load_model(self.model, model_path, strict=False, cpu=True)
        elif config.INIT_MODEL:
            model_opr.load_model(self.model, config.INIT_MODEL, strict=False, cpu=True)

        self.device = torch.device(config.MODEL.DEVICE)
        self.model.to(self.device)
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()])

    def solvers(self):
        self.load_dataset()
        self.load_model()
        optimizer = solver.make_optimizer(config, self.model)  # lr without X num_gpu
        lr_scheduler = solver.make_lr_scheduler(config, optimizer)
        iteration = 0
        if config.CONTINUE_ITER:
            solver_path = osp.join(self.solver_dir, '%d.solver' % config.CONTINUE_ITER)
            iteration = model_opr.load_solver(optimizer, lr_scheduler, solver_path)

        best_psnr = 0
        for epoch in range(1000):
            for batch_data in self.train_loader:
                self.model.train()
                iteration = iteration + 1
                lr_img = batch_data[0].to(self.device)
                s1_img = batch_data[1].to(self.device)
                hr_img = batch_data[2].to(self.device)
                loss_dict = self.model(lr_img, s1_img, hr_img)  # ,img_noise,img_clean
                total_loss = sum(loss for loss in loss_dict.values())
                # if float(total_loss.item()) < 0.3: # for strange loss here ...
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                if self.rank <= 0:
                    if iteration % config.LOG_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                        log_str = 'Iter: %d, LR: %.3e, ' % (iteration, optimizer.param_groups[0]['lr'])
                        for key in loss_dict:
                            self.tb_writer.add_scalar(key, loss_dict[key].mean(), global_step=iteration)
                            log_str += key + ': %.4f, ' % float(loss_dict[key])
                        self.logger.info(log_str)

                    if iteration % config.SAVE_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                        self.logger.info('[Saving] Iter: %d' % iteration)

                    if iteration % config.VAL.PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                        self.logger.info('[Validating] Iter: %d' % iteration)
                        self.model.eval()
                        psnr, ssim = validate(self.model, self.val_loader, self.device, iteration,
                                              save_path=self.save_dir, save_img=True)
                        if best_psnr < psnr:
                            best_psnr = psnr
                            model_path = osp.join(self.model_dir, 'best_unet_lolv2.pth')
                            model_opr.save_model(self.model, model_path)

                        self.logger.info('[Val Result] Iter: %d, PSNR: %.4f, SSIM: %.4f best psnr: %.4f' % (
                            iteration, psnr, ssim, best_psnr))

                    if iteration >= config.SOLVER.MAX_ITER:
                        self.logger.info('Finish training process!')
                        break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    model_obj = Model_Train(args)
    model_obj.solvers()


if __name__ == '__main__':
    main()

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.attention_ocr import AttentionOCR
from src.dataset import Lplate_Dataset
from tqdm import tqdm



class Model_Train:
    def __init__(self):
        self.img_width = 160
        self.img_height = 60
        self.num_classes = 16
        self.nh = 512
        self.fload_weight = False
        self.teacher_forcing_ratio = 0.5
        self.save_checkpoint_every = 5
        self.batch_size = 256
        self.lr = 3e-4
        self.n_epoch = 200
        self.n_workers = 1
        self.test_images_path = "./dataset/"
        self.train_images_path = "./dataset/"
        self.device = "cpu"
        self.load_weight = False

        print("-----------------Start training----------------------------")

    def load_dataloader(self):
        ds_train = Lplate_Dataset(self.train_images_path, self.img_width, self.img_height, self.num_classes + 1)
        ds_test = Lplate_Dataset(self.test_images_path, self.img_width, self.img_height, self.num_classes + 1)
        self.tokenizer = ds_train.tokenizer
        self.train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
        self.test_loader = DataLoader(ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)

    def load_model(self):
        self.model = AttentionOCR(self.num_classes)
        self.model.to(device=self.device)
        self.criterion = nn.NLLLoss().cuda()

        # load_weight_path
        if self.load_weight:
            load_weights = torch.load('./inception_v3_google-1a9a5a14.pth')
            names = set()
            for k, w in self.model.incept_model.named_children():
                names.add(k)
            if self.load_weight:
                weights = {}
                for k, w in self.load_weight.items():
                    if k.split('.')[0] in names:
                        weights[k] = w
                self.model.incept_model.load_state_dict(weights)

    def train_batch(self, input_tensor, target_tensor):
        self.model.train()
        decoder_output = self.model(input_tensor, target_tensor)
        loss = 0
        self.optimizer.zero_grad()
        for i in range(decoder_output.size(1)):
            loss += self.criterion(decoder_output[:, i, :].squeeze(), target_tensor[:, i + 1])
        loss.backward()
        self.optimizer.step()
        target_tensor = target_tensor.cpu()
        decoder_output = decoder_output.cpu()
        prediction = torch.zeros_like(target_tensor)
        prediction[:, 0] = self.tokenizer.SOS_token
        for i in range(decoder_output.size(1)):
            prediction[:, i + 1] = decoder_output[:, i, :].squeeze().argmax(1)
        n_right = 0
        n_right_sentence = 0
        for i in range(prediction.size(0)):
            eq = prediction[i, 1:] == target_tensor[i, 1:]
            n_right += eq.sum().item()
            n_right_sentence += eq.all().item()

        _avg_loss = loss.item() / len(decoder_output)
        _acc =  n_right / prediction.size(0) / prediction.size(1)
        _sentence_acc =  n_right_sentence / prediction.size(0)
        return _avg_loss,_acc,  _sentence_acc


    def predict_batch(self, input_tensor):
        self.model.eval()
        decoder_output = self.model(input_tensor)
        return decoder_output


    def eval_batch(self, input_tensor, target_tensor):
        loss = 0
        decoder_output = self.predict_batch(input_tensor)
        for i in range(decoder_output.size(1)):
            loss += self.criterion(decoder_output[:, i, :], target_tensor[:, i + 1])
        target_tensor = target_tensor.cpu()
        decoder_output = decoder_output.cpu()
        prediction = torch.zeros_like(target_tensor)
        prediction[:, 0] = self.tokenizer.SOS_token
        for i in range(decoder_output.size(1)):
            prediction[:, i + 1] = decoder_output[:, i, :].argmax(1)
        n_right = 0
        n_right_sentence = 0
        for i in range(prediction.size(0)):
            eq = prediction[i, 1:] == target_tensor[i, 1:]
            n_right += eq.sum().item()
            n_right_sentence += eq.all().item()

        _avg_loss = loss.item() / len(decoder_output)
        _acc = n_right / prediction.size(0) / prediction.size(1)
        _sentence_acc = n_right_sentence / prediction.size(0)
        return _avg_loss, _acc, _sentence_acc




    def train_epoch(self):
        sum_loss_train = 0
        n_train = 0
        sum_acc = 0
        sum_sentence_acc = 0
        for i, batch in enumerate(tqdm(self.train_loader)):
            x, y = batch
            x = x.to(device=self.device)
            y = y.to(device=self.device)
            loss, acc, sentence_acc = self.train_batch(x, y)
            sum_loss_train += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc
            n_train += 1
        return sum_loss_train / n_train, sum_acc / n_train, sum_sentence_acc / n_train

    def eval_epoch(self):
        sum_loss_eval = 0
        n_eval = 0
        sum_acc = 0
        sum_sentence_acc = 0
        for bi, batch in enumerate(tqdm(self.test_loader)):
            x, y = batch
            x = x.to(device=self.device)
            y = y.to(device=self.device)
            loss, acc, sentence_acc = self.eval_batch(x, y)
            sum_loss_eval += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc
            n_eval += 1
        return sum_loss_eval / n_eval, sum_acc / n_eval, sum_sentence_acc / n_eval


    def run(self):
        self.load_dataloader()
        self.load_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.n_epoch):
            train_loss, train_acc, train_sentence_acc = self.train_epoch()
            eval_loss, eval_acc, eval_sentence_acc = self.eval_epoch()
            print("Epoch %d" % epoch)
            print('train_loss: %.4f, train_acc: %.4f, train_sentence: %.4f' % (train_loss, train_acc, train_sentence_acc))
            print('eval_loss:  %.4f, eval_acc:  %.4f, eval_sentence:  %.4f' % (eval_loss, eval_acc, eval_sentence_acc))
            if epoch % self.save_checkpoint_every == 0 and epoch > 0:
                print('saving checkpoint...')
                torch.save(self.model.state_dict(),'./chkpoint/time_%s_epoch_%s.pth' % (time.strftime('%Y-%m-%d_%H-%M-%S'), epoch))


if __name__ == '__main__':
    model_train_obj = Model_Train()
    model_train_obj.run()

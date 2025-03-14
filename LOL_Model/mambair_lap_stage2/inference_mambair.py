import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import os.path as osp
import sys 
sys.path.append('/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM/')
from utils.model_opr import load_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os
import math 
torch.backends.cudnn.enabled = False
import torchvision
from dataloader import  NYU_v2_datset
from metrics.common import calculate_ssim
from mambair_lap_stage2.mambair_lap import MambaIR
from utils import model_opr




model_name = 'mambair_lap'

split = 'Test'

result_root = './results/{}_lolv2'.format(model_name)
if not os.path.exists(result_root): os.mkdir(result_root)
print(result_root)

result_root_split = './results/{}_lolv2/{}'.format(model_name,split)
if not os.path.exists(result_root_split): os.mkdir(result_root_split)
print(result_root_split)


net = MambaIR(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda() 
model_path = '/dataset/kunzhou/project/low_light_noisy/logs/kan_unet_ei/lolv2_mambair_lap/models/best_mambair_ft.pth'

print("g_net have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters())/1000000.0))
print("backbone have {:.3f}M paramerters in total".format(sum(x.numel() for x in filter(lambda p: p.requires_grad,net.parameters()))/1000000.0))

load_model(net, model_path)


# img_path = "/dataset/kunzhou/project/low_light_noisy/lol_dataset/LOL-v2/Real_captured/Train/Low/"

val_dataset = NYU_v2_datset(split = 'test',rank=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

@torch.no_grad()
def validate():
    net.eval()

    idx = 0
    img_list = []
    for idx, imgs in enumerate(val_loader):
        low_img,img_hn,high_img,snr_prior = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda(), imgs[3].cuda()
        
        if 'snr' not in model_name:
            out = net(low_img,img_hn)[-1]
        else:
            dark = low_img.clone()
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            light = snr_prior
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)
            out = net(low_img,img_hn,mask)[-1]

        resore_img_out_fi = out.clamp(0,1).detach().cpu().numpy()[0]
        resore_img_out_fi = np.transpose(resore_img_out_fi,(1,2,0))

        if True:
           
           cv2.imwrite('{}/{}.png'.format(result_root+'/{}'.format(split),str(idx).zfill(5)),resore_img_out_fi[:,:]*255)
        idx +=1

validate()
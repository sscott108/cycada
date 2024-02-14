#!/usr/bin/env python
# coding: utf-8

# In[2]:

import torch
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import random
import torch
import torch.nn as nn
import config
import copy
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import rasterio as rio
from rasterio.windows import Window
import numpy as np
from os import listdir
from os.path import isfile, join
np.random.seed(42)
import copy
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import cv2
from PIL import Image, ImageEnhance
import itertools
import torchvision.transforms as transforms
import glob
from tqdm.notebook import tqdm
import warnings
from torchvision.utils import make_grid
import functools
import os
import torchvision.models as models
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import pandas as pd
import statistics
from scipy import stats
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import datetime
Tensor  = torch.cuda.FloatTensor


PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "Figure_PDFs"
if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory didn''t exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.') 
    
img_height   = 256
img_width    = 256
channels     = 3

transforms_ = [
    transforms.Resize(int(img_height*1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # reset Conv2d's weight(tensor) with Gaussian Distribution
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) # reset Conv2d's bias(tensor) with Constant(0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # reset BatchNorm2d's weight(tensor) with Gaussian Distribution
            torch.nn.init.constant_(m.bias.data, 0.0) # reset BatchNorm2d's bias(tensor) with Constant(0)
           

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):

        super(GANLoss, self).__init__()
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = gan_mode
        
        self.loss = nn.BCEWithLogitsLoss()
            
            
    def get_target_tensor(self, prediction, target_is_real):
        
        if target_is_real:
            target_tensor = self.real_label
            
        else:
            target_tensor = self.fake_label
        
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
    

    
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def gen_dis_loss(genloss, disloss, iters, save = True, fig_name=''):
#     epoch = range(epochs)
    fig, ax = plt.subplots(1,1, figsize = (6,6))   
    ax.plot(iters, genloss, color='b', linewidth=0.5, label='Generator')
    ax.plot(iters, disloss, color='r', linewidth=0.5, label='Discriminator')
    ax.set_xlabel('Iters')
    ax.set_ylabel('Loss')
    ax.set_title('Generator and Discriminator Loss')
    ax.legend()
    plt.show()
    if save==True:
        fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')
        
def trainloss(mainloss, iters, save = True, fig_name=''):
    fig, ax = plt.subplots(1,1, figsize = (6,6))   
    ax.plot(iters, mainloss, color='b', linewidth=0.5)
    ax.set_xlabel('Iters')
    ax.set_ylabel('Loss')
    plt.show()
    if save==True:
        fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')
        


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
    
    
class CyDataset(Dataset):
    def __init__(self):

        self.D = []
        self.L = []
                
        with open('/datacommons/carlsonlab/srs108/old/ol/Delhi_labeled.pkl', "rb") as fp:
            for station in tqdm(pkl.load(fp)):
                self.D.append(tuple((station['Image'][:,:,:3], station['PM25'])))
                
        with open('/datacommons/carlsonlab/srs108/old/ol/Lucknow.pkl', "rb") as fp:
            for station in tqdm(pkl.load(fp)):
                for datapoint in station:
                    luck_img = datapoint['Image'][:,:,:3]
                    if luck_img.shape == (224, 224,3):  
                        self.L.append(tuple((luck_img, datapoint['PM'])))
                        
        self.L = random.choices(self.L, k= len(self.D))
        
    def __len__(self): return (len(self.D))
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        transform  = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor()])

        d_img = self.D[idx][0]
        d_img = transform(d_img)        
        l_img = self.L[idx][0]
        l_img = transform(l_img)
        
        sample = {
              'D img': d_img,
              'D pm' : torch.tensor(self.D[idx][1]),
              'L img': l_img,
              'L pm' : torch.tensor(self.L[idx][1])
        }
        return sample

    

#LATENT GENERATOR    
class CINResnetGenerator(nn.Module):
    def __init__(self, nlatent, input_nc, output_nc, ngf=64, norm_layer=CondInstanceNorm,
                 use_dropout=False, n_blocks=9, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(CINResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        instance_norm = functools.partial(InstanceNorm2d, affine=True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2*ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2*ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4*ngf, nlatent),
            nn.ReLU(True)
        ]
        
        for i in range(3):
            model += [CINResnetBlock(x_dim=4*ngf, z_dim=nlatent, padding_type=padding_type,
                                     norm_layer=norm_layer, use_dropout=use_dropout, use_bias=True)]

        model += [
            nn.ConvTranspose2d(4*ngf, 2*ngf, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=True),
            norm_layer(2*ngf , nlatent),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = TwoInputSequential(*model)

    def forward(self, input, noise):
        if len(self.gpu_ids)>1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)

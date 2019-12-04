# Author: Tran Minh Quan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#-----------------------------------------------------------------------
import os
import sys
import glob
import random
import shutil
import logging
import argparse
from collections import OrderedDict
from natsort import natsorted
import math
import cv2
import numpy as np
import pandas as pd
import sklearn.metrics
import json
# from easydict import EasyDict as edict
#-----------------------------------------------------------------------
# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.utils.weight_norm as weightNorm
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
import torchvision
# from torchvision.datasets import MNIST
# import torchvision.transforms as transforms
from torchvision.models.densenet import (densenet121, densenet169, densenet161, densenet201)
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

#-----------------------------------------------------------------------
# # Using tensorflow
# import tensorflow as tf

#-----------------------------------------------------------------------
# An efficient dataflow loading for training and testing
# import tensorpack.dataflow as df

from tensorpack import dataflow, imgaug
from tensorpack.dataflow import *
import albumentations as AB


# from test_tube import HyperOptArgumentParser, Experiment
# from pytorch_lightning import * #.models.trainer import Trainer
from test_tube import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

#-----------------------------------------------------------------------
# Global configuration
#
# global args
SHAPE = 512
BATCH = 16
EPOCH = 100
GROUP = 5
DEBUG = False 
DPATH = ''


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}

# Custom model
# from se_dense import *153
# from xception_dropout import *
# from dataset import ImageDataset  # noqa
from chexpert import *


# def dice_loss(inputs, target):
#     # inputs = torch.sigmoid(inputs)
#     smooth = 1.

#     iflat = inputs.view(-1)
#     tflat = target.view(-1)
#     intersection = (iflat * tflat).sum()
    
#     return 1 - ((2. * intersection + smooth) /
#               (iflat.sum() + tflat.sum() + smooth))

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0.25, weight=None):
#         super().__init__()
#         self.gamma = gamma
#         # self.nll = nn.NLLLoss(weight=weight, reduce=False)
#         self.loss = nn.BCELoss()
        
#     def forward(self, inputs, target):
#         # loss = self.nll(inputs, target)
#         loss = self.loss(inputs, target)
        
#         # one_hot = make_one_hot(target.unsqueeze(dim=1), inputs.size()[1])
#         inv_probs = 1 - inputs.exp()
#         focal_weights = (inv_probs * target).sum(dim=1) ** self.gamma
#         loss = loss * focal_weights
        
#         return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, inputs, target):
        if not (target.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as inputs size ({})"
                             .format(target.size(), inputs.size()))
 
        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * target + max_val + \
            ((-max_val).exp() + (-inputs - max_val).exp()).log()
 
        invprobs = F.logsigmoid(-inputs * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

class Classifier(pl.LightningModule):

    def __init__(self, name='DenseNet121', mode='full'):
        super(Classifier, self).__init__()
        self.name = name
        self.mode = mode

        if self.name=='DenseNet121':
            self.model = densenet121(pretrained=False)
            self.model.classifier = nn.Linear(1024, GROUP)
        # elif self.name=='DenseNet169':
        #     self.model = densenet169(pretrained=False)
        #     self.model.classifier = nn.Linear(1024, GROUP)
        elif self.name=='DenseNet201':
            self.model = densenet201(pretrained=False)
            self.model.classifier = nn.Linear(1920, GROUP)
        else:
            self.model = None
        print(self.model)
        # self.criterion = nn.BCELoss()
        self.criterion = FocalLoss()
        self.output_ = []
        self.target_ = []

    def forward(self, x):
        logit = self.model(x) # Focal loss already has sigmoid
        return logit

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        # return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)
        # dataloader_train = DataLoader(
        #     ImageDataset(cfg.train_csv, cfg, mode='train'),
        #     batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        #     drop_last=True, shuffle=True)
        # return dataloader_train
        ds_train = Chexpert(folder=args.data, 
            train_or_valid='train',
            fname='train.csv',
            group=GROUP,
            resize=int(SHAPE),
            debug=DEBUG
            )
        

        ag_train = [
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB), 
            imgaug.RotationAndCropValid(max_deg=20),
            # imgaug.RandomOrderAug(
            #     [
            #         imgaug.BrightnessScale((0.6, 1.4), clip=False),
            #         imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
            #         imgaug.Saturation(0.4, rgb=False),
            #         # rgb-bgr conversion for the constants copied from fb.resnet.torch
            #         imgaug.Lighting(0.1,
            #                         eigval=np.asarray(
            #                          [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
            #                         eigvec=np.array(
            #                          [[-0.5675, 0.7192, 0.4009],
            #                           [-0.5808, -0.0045, -0.8140],
            #                           [-0.5836, -0.6948, 0.4203]],
            #                          dtype='float32')[::-1, ::-1]
            #                         )
            #         ]),
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.60, 1.0), 
                    aspect_ratio_range=(0.95, 1.05),
                    interp=cv2.INTER_LINEAR, target_shape=SHAPE),
            # imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]

        ds_train.reset_state()
        ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        ds_train = MultiProcessRunner(ds_train, num_proc=4, num_prefetch=4)
        ds_train = BatchData(ds_train, BATCH)
        # ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=8)
        ds_train = PrintData(ds_train)
        ds_train = MapData(ds_train, lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))), torch.tensor(dp[1]).float()])
        return ds_train

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)
        # dataloader_dev = DataLoader(
        #     ImageDataset(cfg.dev_csv, cfg, mode='dev'),
        #     batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        #     drop_last=True, shuffle=True)
        # return dataloader_dev
        ds_valid = Chexpert(folder=args.data, #'/u01/data/CheXpert-v1.0-small'
            train_or_valid='valid',
            fname='valid.csv',
            group=GROUP,
            resize=int(SHAPE),
            debug=DEBUG
            )
        ds_valid.reset_state() 
        ag_valid = [
            imgaug.ResizeShortestEdge(SHAPE),
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB), 
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.CenterCrop((SHAPE, SHAPE)),
            # imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32()
        ]

        ds_valid.reset_state()
        ds_valid = AugmentImageComponent(ds_valid, ag_valid, 0) 
        ds_valid = BatchData(ds_valid, BATCH)
        ds_valid = PrintData(ds_valid)
        ds_valid = MapData(ds_valid, lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))), torch.tensor(dp[1]).float()])
        return ds_valid

    # @pl.data_loader
    # def test_dataloader(self):
        # pass 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        return [optimizer]

    def training_step(self, batch, batch_nb):
        inputs, target = batch
        output = self.forward(inputs / 255.0)

        loss = self.criterion(output, target)
        # loss = dice_loss(output, target)
        self.logger.experiment.add_image('train/image', inputs[0] / 255.0, self.global_step, dataformats='CHW')
        return {'loss': loss}


    def validation_step(self, batch, batch_nb):
        inputs, target = batch
        output = self.forward(inputs / 255.0)

        loss = self.criterion(output, target)
        # loss = dice_loss(output, target)
        self.logger.experiment.add_image('valid/image', inputs[0] / 255.0, self.global_step, dataformats='CHW')
        self.output_.append(output)
        self.target_.append(target)
        return {'valid/loss': loss, 
                }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['valid/loss'] for x in outputs]).mean().cpu().numpy()
        auc = []
        output_ = torch.cat(self.output_, 0).detach().cpu().numpy()
        target_ = torch.cat(self.target_, 0).detach().cpu().numpy().astype(np.uint8)
        for idx in range(target_.shape[-1]):
            fpr, tpr, _ = sklearn.metrics.roc_curve(target_[:,idx], output_[:,idx])
            auc.append(sklearn.metrics.auc(fpr, tpr))
        auc = np.array(auc)
        auc_m = auc.mean(axis=-1)
        print('valid/avg_loss', avg_loss)
        print('valid/auc[0]', auc[0])
        print('valid/auc[1]', auc[1])
        print('valid/auc[2]', auc[2])
        print('valid/auc[3]', auc[3])
        print('valid/auc[4]', auc[4])
        print('valid/auc_m', auc_m)

        self.logger.experiment.add_scalar('valid/loss', avg_loss, self.global_step)
        self.logger.experiment.add_scalar('valid/auc[0]', auc[0], self.global_step)
        self.logger.experiment.add_scalar('valid/auc[1]', auc[1], self.global_step)
        self.logger.experiment.add_scalar('valid/auc[2]', auc[2], self.global_step)
        self.logger.experiment.add_scalar('valid/auc[3]', auc[3], self.global_step)
        self.logger.experiment.add_scalar('valid/auc[4]', auc[4], self.global_step)
        self.logger.experiment.add_scalar('valid/auc_m', auc_m, self.global_step)


        self.output_ = []
        self.target_ = []
        return {'valid/avg_loss': avg_loss, 
                'valid/auc[0]': auc[0],
                'valid/auc[1]': auc[1],
                'valid/auc[2]': auc[2],
                'valid/auc[3]': auc[3],
                'valid/auc[4]': auc[4],
                'valid/auc_m': auc_m,
                }

if __name__ == '__main__':
    #-----------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2020, type=int, help='reproducibility')
    parser.add_argument('--gpus', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--num_workers', default=8, type=int, help='num_workers')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--src_csv', help='src csv')
    parser.add_argument('--dst_csv', help='dst csv')
    parser.add_argument('--data', help='Image directory')
    parser.add_argument('--mode', help='small | patch | full', default='small')
    parser.add_argument('--model', help='Model name', default='ResNet50')
    parser.add_argument('--group', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--shape', type=int, default=320)
    parser.add_argument('--sample', action='store_true', help='run inference')
    parser.add_argument('--debug', action='store_true', help='small dataset')
    
    args = parser.parse_args()
    print(args)
    #-----------------------------------------------------------------------
    # Choose the GPU
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    #-----------------------------------------------------------------------
    # Seed the randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    #-----------------------------------------------------------------------
    # Initialize the program
    # TODO
    # writer = SummaryWriter()
    use_cuda = torch.cuda.is_available()
    xpu = torch.device("cuda:{}".format(args.gpus) if torch.cuda.is_available() else "cpu")
    # step = 0

    DEBUG = args.debug
    GROUP = args.group
    SHAPE = args.shape
    BATCH = args.batch
    DPATH = args.data
    #-----------------------------------------------------------------------
    # Train from scratch or load the pretrained network
    #
    # TODO: Create the model
    #-----------------------------------------------------------------------

    print('Create model...')
    model = Classifier(name=args.model)
    print('Built model')

    # TODO: Load the pretrained model
    if args.load:
        # TODO
        chkpt = torch.load(args.load, map_location=xpu)
        model.load_state_dict(chkpt)
        # model = Classifier()
        print('Loaded from {}...'.format(args.load))
    
    

    #-----------------------------------------------------------------------
    # Perform inference
    if args.sample:
        sample(datadir=args.data, model=model)
        sys.exit()
    else:
        


        #-----------------------------------------------------------------------
        # Attach dataflow to model
        # model.fetch_dataflow(ds_train=ds_train,
        #                    ds_valid=None, 
        #                    ds_test=None)
        #-----------------------------------------------------------------------
        # 2 INIT TEST TUBE EXP
        #-----------------------------------------------------------------------

        # init experiment
        # exp = Experiment(
        #     name='Classifier', #hyperparams.experiment_name,
        #     save_dir=os.path.join('runs', args.model, args.mode), #hyperparams.test_tube_save_path,
        #     # autosave=False,
        #     # description='experiment'
        # )

        # exp.save()

        #-----------------------------------------------------------------------
        # 3 DEFINE CALLBACKS
        #-----------------------------------------------------------------------
        # model_save_path = 'checkpoint' #'{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
        # early_stop = EarlyStopping(
        #     monitor='avg_val_loss',
        #     patience=5,
        #     verbose=True,
        #     mode='auto'
        # )


        checkpoint_callback = ModelCheckpoint(
            filepath='checkpoint',
            save_best_only=False,
            verbose=True,
            monitor='valid/avg_loss',
            mode='min',
            prefix=''
        )

        logger = TestTubeLogger(
            save_dir='train_log_pytorch',
            version=0  # An existing version with a saved checkpoint
        )

        #-----------------------------------------------------------------------
        # 4 INIT TRAINER
        #-----------------------------------------------------------------------
        trainer = pl.Trainer(
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            # early_stop_callback=early_stop,
            # default_save_path='runs',
            max_nb_epochs=EPOCH, 
            gpus=-1, #map(int, args.gpus.split(',')), #hparams.gpus,
            # distributed_backend='ddp'
        )

        #-----------------------------------------------------------------------
        # 5 START TRAINING
        #-----------------------------------------------------------------------
        trainer.fit(model)
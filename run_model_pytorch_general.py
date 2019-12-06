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
import torch.backends.cudnn as cudnn
# from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.models as models
# from torchvision.datasets import MNIST
# import torchvision.transforms as transforms
# from torchvision.models.densenet import (densenet121, densenet169, densenet161, densenet201)
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
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

#-----------------------------------------------------------------------
# Global configuration
#
# global args
SHAPE = 512
BATCH = 32
EPOCH = 100
GROUP = 5
DEBUG = False 

from chexpert import *

MODEL_NAMES = sorted(
    name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])
)
print(MODEL_NAMES)

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

class CustomLightningModule(pl.LightningModule):

    def __init__(self, hparams):
        super(CustomLightningModule, self).__init__()
        self.hparams = hparams # architecture is stored here
        print(self.hparams)
        self.model = getattr(torchvision.models, self.hparams.arch)()#(pretrained=self.hparams.pretrained)

        self.model.classifier = nn.Linear(1024, GROUP)
        # if self.hparams.pretrained:
        #     self.model = CustomLightningModule.load_from_checkpoint(
        #         os.path.join(trainer.checkpoint_callback.filepath, "_ckpt_epoch_0.ckpt")
        #         )
        # if self.hparams.load is not None:
        #     self.model = CustomLightningModule.load_from_metrics(weights_path=self.hparams.load)
        # print(self.model.parameters)
        self.criterion = FocalLoss()
        self.output_ = []
        self.target_ = []

    def forward(self, x):
        logit = self.model(x) # Focal loss already has sigmoid
        return logit

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        ds_train = Chexpert(folder='/u01/data/CheXpert-v1.0',
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
        ds_valid = Chexpert(folder='/u01/data/CheXpert-v1.0',
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

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        ds_test2 = Chexpert(folder='/vinbrain/data/mimic_validation_data',
            train_or_valid='valid',
            fname='mimic_valid.csv',
            group=GROUP,
            resize=int(SHAPE),
            debug=DEBUG
            )
        ds_test2.reset_state() 
        ag_valid = [
            imgaug.ResizeShortestEdge(SHAPE),
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB), 
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.CenterCrop((SHAPE, SHAPE)),
            # imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32()
        ]

        ds_test2.reset_state()
        ds_test2 = AugmentImageComponent(ds_test2, ag_valid, 0) 
        ds_test2 = BatchData(ds_test2, BATCH)
        ds_test2 = PrintData(ds_test2)
        ds_test2 = MapData(ds_test2, lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))), torch.tensor(dp[1]).float()])
        return ds_test2

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
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
        return {'valid/loss': loss}

    def test_step(self, batch, batch_nb):
        inputs, target = batch
        output = self.forward(inputs / 255.0)

        loss = self.criterion(output, target)
        # loss = dice_loss(output, target)
        self.logger.experiment.add_image('test2/image', inputs[0] / 255.0, self.global_step, dataformats='CHW')
        self.output_.append(output)
        self.target_.append(target)
        return {'test2/loss': loss}

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

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test2/loss'] for x in outputs]).mean().cpu().numpy()
        auc = []
        output_ = torch.cat(self.output_, 0).detach().cpu().numpy()
        target_ = torch.cat(self.target_, 0).detach().cpu().numpy().astype(np.uint8)
        for idx in range(target_.shape[-1]):
            fpr, tpr, _ = sklearn.metrics.roc_curve(target_[:,idx], output_[:,idx])
            auc.append(sklearn.metrics.auc(fpr, tpr))
        auc = np.array(auc)
        auc_m = auc.mean(axis=-1)
        print('test2/avg_loss', avg_loss)
        print('test2/auc[0]', auc[0])
        print('test2/auc[1]', auc[1])
        print('test2/auc[2]', auc[2])
        print('test2/auc[3]', auc[3])
        print('test2/auc[4]', auc[4])
        print('test2/auc_m', auc_m)

        self.logger.experiment.add_scalar('test2/loss', avg_loss, self.global_step)
        self.logger.experiment.add_scalar('test2/auc[0]', auc[0], self.global_step)
        self.logger.experiment.add_scalar('test2/auc[1]', auc[1], self.global_step)
        self.logger.experiment.add_scalar('test2/auc[2]', auc[2], self.global_step)
        self.logger.experiment.add_scalar('test2/auc[3]', auc[3], self.global_step)
        self.logger.experiment.add_scalar('test2/auc[4]', auc[4], self.global_step)
        self.logger.experiment.add_scalar('test2/auc_m', auc_m, self.global_step)


        self.output_ = []
        self.target_ = []
        return {'test2/avg_loss': avg_loss, 
                'test2/auc[0]': auc[0],
                'test2/auc[1]': auc[1],
                'test2/auc[2]': auc[2],
                'test2/auc[3]': auc[3],
                'test2/auc[4]': auc[4],
                'test2/auc_m': auc_m,
                }

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121', choices=MODEL_NAMES,
                            help='model architecture: ' +
                                 ' | '.join(MODEL_NAMES) +
                                 ' (default: densenet121)')
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2020,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        #                     help='momentum')
        # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
        #                     metavar='W', help='weight decay (default: 1e-4)',
        #                     dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        return parser
        
def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    # parent_parser.add_argument('--data-path', metavar='DIR', type=str,
    #                            help='path to dataset')
    parent_parser.add_argument('--save-path', metavar='DIR', type=str, default="checkpoint", 
                               help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--load', type=str, default=None,
                               help='path to save output')
    parent_parser.add_argument('--evaluate', action='store_true',
                               help='run evaluate')
    parent_parser.add_argument('--predict', action='store_true',
                               help='run predict')
    parser = CustomLightningModule.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = CustomLightningModule(hparams)
    
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    xpu = torch.device("cuda:{}".format(hparams.gpus) if torch.cuda.is_available() else "cpu")

    if hparams.load is not None: # Load the checkpoint here
        chkpt = torch.load(hparams.load, map_location=xpu)
        model.load_state_dict(chkpt['state_dict'])
        print('Loaded from {}...'.format(hparams.load))

    if hparams.evaluate:
        pass
    elif hparams.predict:
        pass
    else:
        logger = TestTubeLogger(
            save_dir='hparams.save_path',
            version=1  # An existing version with a saved checkpoint
        )
        checkpoint_callback = ModelCheckpoint(
            filepath='checkpoint',
            save_best_only=True,
            verbose=True,
            monitor='valid/avg_loss',
            mode='min',
            prefix=''
        )
        trainer = pl.Trainer(
            default_save_path=hparams.save_path,
            gpus=-1, #hparams.gpus,
            max_nb_epochs=hparams.epochs, 
            # checkpoint_callback=checkpoint_callback, 
            early_stop_callback=None,
            # distributed_backend=hparams.distributed_backend,
            # use_amp=hparams.use_16bit
        )
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())


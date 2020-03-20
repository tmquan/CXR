"""
This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import argparse
import os
import random
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

import tensorpack.dataflow
from tensorpack.dataflow import imgaug
from tensorpack.dataflow import AugmentImageComponent
from tensorpack.dataflow import BatchData, MultiProcessRunner, PrintData, MapData, FixedSizeData

import albumentations as AB
import cv2 
import numpy as np

import sklearn.metrics
from vinmec import Vinmec
# pull out resnet names from torchvision models
MODEL_NAMES = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


class ImageNetLightningModel(LightningModule):
    def __init__(self, hparams):
        """
        TODO: add docstring here
        """
        super(ImageNetLightningModel, self).__init__()
        self.hparams = hparams
        if self.hparams.arch=='densenet121':
            self.model = getattr(models, self.hparams.arch)(pretrained=self.hparams.pretrained)
            # self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
            #     stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1024, self.hparams.types), #2 if self.hparams.types==1 else self.hparams.types), # 5 diseases
                nn.Sigmoid(),
            )
            # self.model.classifier = nn.Linear(1024, self.hparams.types)
        else:
            ValueError
        print(self.model)
        # self.criterion = nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')
        # self.criterion = nn.MultiLabelMarginLoss(reduction='mean')
        # self.criterion = nn.L1Loss()
        if self.hparams.types==1:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.average_type = 'binary' if self.hparams.types==1 else 'weighted'
        self.val_output = np.array([])
        self.val_target = np.array([])  
        self.test_output = np.array([])
        self.test_target = np.array([])

    def forward(self, x):
        x = x / 128.0 - 1.0
        return (self.model(x))

    def training_step(self, batch, batch_idx, prefix=''):
        images, target = batch
        output = self.forward(images)
        loss = self.criterion(output, target)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        
        tqdm_dict = {'train_loss': loss}
        result = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return result

    def validation_step(self, batch, batch_idx, prefix='val_'):
        images, target = batch
        output = self.forward(images)
        loss = self.criterion(output, target)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        target = target.detach().cpu().numpy()
        self.val_target = np.concatenate((self.val_target, target), axis=0) if len(self.val_target) > 0 else target
        output = output.detach().cpu().numpy()
        self.val_output = np.concatenate((self.val_output, output), axis=0) if len(self.val_output) > 0 else output
        
        result = OrderedDict({
            'val_loss': loss,
        })
        return result

    def validation_epoch_end(self, outputs, prefix='val_'):
        self.val_output = (self.val_output > self.hparams.threshold).astype(np.float32)
        self.val_target = (self.val_target > self.hparams.threshold).astype(np.float32)
        # print(self.val_output.shape, self.val_target.shape)

        pathologies = ["Airspace_Opacity", "Cardiomegaly", "Fracture", 
                       "Lung_Lesion", "Pleural_Effusion", "Pneumothorax"]
        if self.hparams.types==1:
            pathology=0
        else:
            pathology = pathologies.index(self.hparams.pathology)
        
        self.val_output = self.val_output[:,pathology]
        self.val_target = self.val_target[:,pathology]
        f1_score = sklearn.metrics.fbeta_score(self.val_target, self.val_output, beta=1, average=self.average_type)
        f2_score = sklearn.metrics.fbeta_score(self.val_target, self.val_output, beta=2, average=self.average_type)
        precision_score = sklearn.metrics.precision_score(self.val_target, self.val_output, average=self.average_type)
        recall_score = sklearn.metrics.recall_score(self.val_target, self.val_output, average=self.average_type)

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': val_loss_mean, 
                     'val_f1_score': f1_score,
                     'val_f2_score': f2_score,
                     'val_precision_score': precision_score,
                     'val_recall_score': recall_score,
                     }
        result = {'progress_bar': tqdm_dict, 
                  'log': tqdm_dict, 
                  'val_loss': val_loss_mean,
                  'val_f1_score': f1_score,
                  'val_f2_score': f2_score,
                  'val_precision_score': precision_score,
                  'val_recall_score': recall_score,}
        # print(self.val_output.max(), self.val_target.max())
        # Reset the result
        self.val_output = np.array([])
        self.val_target = np.array([])  
        return result

    def test_step(self, batch, batch_idx, prefix='test_'):
        images, target = batch
        output = self.forward(images)
        loss = self.criterion(output, target)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        target = target.detach().cpu().numpy()
        self.test_target = np.concatenate((self.test_target, target), axis=0) if len(self.test_target) > 0 else target
        output = output.detach().cpu().numpy()
        self.test_output = np.concatenate((self.test_output, output), axis=0) if len(self.test_output) > 0 else output
        
        result = OrderedDict({
            'test_loss': loss,
        })
        return result

    def test_epoch_end(self, outputs, prefix='test_'):
        self.test_output = (self.test_output > self.hparams.threshold).astype(np.float32)
        self.test_target = (self.test_target > self.hparams.threshold).astype(np.float32)
        # print(self.test_output.shape, self.test_target.shape)

        pathologies = ["Airspace_Opacity", "Cardiomegaly", "Fracture", 
                       "Lung_Lesion", "Pleural_Effusion", "Pneumothorax"]
        if self.hparams.types==1:
            pathology=0
        else:
            pathology = pathologies.index(self.hparams.pathology)
        
        self.test_output = self.test_output[:,pathology]
        self.test_target = self.test_target[:,pathology]
        f1_score = sklearn.metrics.fbeta_score(self.test_target, self.test_output, beta=1, average=self.average_type)
        f2_score = sklearn.metrics.fbeta_score(self.test_target, self.test_output, beta=2, average=self.average_type)
        precision_score = sklearn.metrics.precision_score(self.test_target, self.test_output, average=self.average_type)
        recall_score = sklearn.metrics.recall_score(self.test_target, self.test_output, average=self.average_type)

        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tqdm_dict = {'test_loss': test_loss_mean, 
                     'test_f1_score': f1_score,
                     'test_f2_score': f2_score,
                     'test_precision_score': precision_score,
                     'test_recall_score': recall_score,
                     }
        result = {'progress_bar': tqdm_dict, 
                  'log': tqdm_dict, 
                  'test_loss': test_loss_mean,
                  'test_f1_score': f1_score,
                  'test_f2_score': f2_score,
                  'test_precision_score': precision_score,
                  'test_recall_score': recall_score,}
        # print(self.test_output.max(), self.test_target.max())
        # Reset the result
        self.test_output = np.array([])
        self.test_target = np.array([])  
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        ds_train = Vinmec(folder=self.hparams.data_path,
                          is_train='train',
                          fname='train.csv',
                          types=self.hparams.types,
                          pathology=self.hparams.pathology,
                          resize=int(self.hparams.shape))

        ds_train.reset_state()
        ag_train = [
            imgaug.Albumentations(AB.SmallestMaxSize(self.hparams.shape, p=1.0)), 
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.RandomChooseAug([
                imgaug.Albumentations(AB.Blur(blur_limit=4, p=0.25)),  
                imgaug.Albumentations(AB.MotionBlur(blur_limit=4, p=0.25)),  
                imgaug.Albumentations(AB.MedianBlur(blur_limit=4, p=0.25)),  
            ]),
            imgaug.Albumentations(AB.CLAHE(p=1.0)),
            imgaug.RotationAndCropValid(max_deg=25),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0), 
                    aspect_ratio_range=(0.8, 1.2),
                    interp=cv2.INTER_AREA, 
                    target_shape=self.hparams.shape),
            imgaug.ToFloat32(),
        ]
        ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        # Label smoothing
        ag_label = [ 
            imgaug.BrightnessScale((0.8, 1.2), clip=False),
        ]
        # ds_train = AugmentImageComponent(ds_train, ag_label, 1)
        ds_train = BatchData(ds_train, self.hparams.batch, remainder=True)
        ds_train = PrintData(ds_train)
        if self.hparams.debug:
            ds_train = FixedSizeData(ds_train, 2)
        ds_train = MultiProcessRunner(ds_train, num_proc=8, num_prefetch=16)
        ds_train = MapData(ds_train,
                           lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2)) ), 
                                       torch.tensor(dp[1]).float() ])
        return ds_train

    def val_dataloader(self):
        ds_valid = Vinmec(folder=self.hparams.data_path,
                          is_train='valid',
                          fname='valid.csv',
                          types=self.hparams.types,
                          pathology=self.hparams.pathology,
                          resize=int(self.hparams.shape))

        ds_valid.reset_state()
        ag_valid = [
            imgaug.Albumentations(AB.SmallestMaxSize(self.hparams.shape, p=1.0)),  
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.ToFloat32(),
        ]
        ds_valid = AugmentImageComponent(ds_valid, ag_valid, 0)
        ds_valid = BatchData(ds_valid, self.hparams.batch, remainder=True)
        ds_valid = PrintData(ds_valid)
        # ds_valid = MultiProcessRunner(ds_valid, num_proc=4, num_prefetch=16)
        ds_valid = MapData(ds_valid,
                           lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2)) ), 
                                       torch.tensor(dp[1]).float() ])
        return ds_valid

    def test_dataloader(self):
        ds_test = Vinmec(folder=self.hparams.data_path,
                          is_train='valid',
                          fname='test.csv',
                          types=self.hparams.types,
                          pathology=self.hparams.pathology,
                          resize=int(self.hparams.shape))

        ds_test.reset_state()
        ag_test = [
            imgaug.Albumentations(AB.SmallestMaxSize(self.hparams.shape, p=1.0)),  
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.ToFloat32(),
        ]
        ds_test = AugmentImageComponent(ds_test, ag_test, 0)
        ds_test = BatchData(ds_test, self.hparams.batch, remainder=True)
        ds_test = PrintData(ds_test)
        # ds_test = MultiProcessRunner(ds_test, num_proc=4, num_prefetch=16)
        ds_test = MapData(ds_test,
                           lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2)) ), 
                                       torch.tensor(dp[1]).float() ])
        return ds_test

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--arch', metavar='ARCH', default='densenet121', choices=MODEL_NAMES,
                            help='model architecture: ' +
                                 ' | '.join(MODEL_NAMES) +
                                 ' (default: densenet121)')
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2222,
                            help='seed for initializing training. ')
        parser.add_argument('--batch', default=64, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--debug', action='store_true',
                            help='use fast mode')
        return parser


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_path', metavar='DIR', default=".", type=str,
                               help='path to dataset')
    parent_parser.add_argument('--save_path', metavar='DIR', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--info_path', metavar='DIR', default="train_log_pytorch", 
                               help='path to logging output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('--fast_dev_run',
                               default=False, action='store_true',
                               help='fast_dev_run: runs 1 batch of train, test, val (ie: a unit test)')

    parent_parser.add_argument('--types', type=int, default=1)
    parent_parser.add_argument('--threshold', type=float, default=0.5)
    parent_parser.add_argument('--pathology', default='Fracture')
    parent_parser.add_argument('--shape', type=int, default=320)

    # Inference purpose
    parent_parser.add_argument('--load', action='store_true', 
                               help='path to logging output')
    parent_parser.add_argument('--pred', action='store_true', help='run predict')
    parent_parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    
    
    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = ImageNetLightningModel(hparams)
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



    trainer = pl.Trainer(
        default_save_path=hparams.save_path,
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        early_stop_callback=None,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit,
        fast_dev_run=hparams.fast_dev_run,
    )
    if hparams.eval:
        trainer.run_evaluation()
    else:
        trainer.fit(model)
        trainer.test()

if __name__ == '__main__':
    main(get_args())
# coding=utf-8
# Author: Tran Minh Quan
import cv2
import random
import numpy as np
import pandas as pd
from datetime import datetime

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.predict import FeedfreePredictor, PredictConfig
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils.stats import BinaryStatistics
import albumentations as AB
import argparse
import sys
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
# tf = tf.compat.v1
tf.disable_v2_behavior()

from vinmec import Vinmec
from models.inceptionbn import InceptionBN
from models.shufflenet import ShuffleNet
from models.densenet import DenseNet121
from models.resnet import ResNet101
from models.vgg16 import VGG16


def visualize_tensors(name, imgs, scale_func=lambda x: (x + 1.) * 128., max_outputs=1):
    """Generate tensor for TensorBoard (casting, clipping)
    Args:
        name: name for visualization operation
        *imgs: multiple tensors as list
        scale_func: scale input tensors to fit range [0, 255]
    Example:
        visualize_tensors('viz1', [img1])
        visualize_tensors('viz2', [img1, img2, img3], max_outputs=max(30, BATCH))
    """
    xy = scale_func(tf.concat(imgs, axis=2))
    xy = tf.cast(tf.clip_by_value(xy, 0, 255), tf.uint8, name='viz')
    tf.summary.image(name, xy, max_outputs=30)


class CustomBinaryClassificationStats(Inferencer):
    """
    Compute precision / recall in binary classification, given the
    prediction vector and the label vector.
    """

    def __init__(self, pred_tensor_name, label_tensor_name, prefix='validation'):
        """
        Args:
            pred_tensor_name(str): name of the 0/1 prediction tensor.
            label_tensor_name(str): name of the 0/1 label tensor.
        """
        self.pred_tensor_name = pred_tensor_name
        self.label_tensor_name = label_tensor_name
        self.prefix = prefix

    def _before_inference(self):
        self.stat = BinaryStatistics()

    def _get_fetches(self):
        return [self.pred_tensor_name, self.label_tensor_name]

    def _on_fetches(self, outputs):
        pred, label = outputs
        # Remove Pneumonia/infection
        pred = pred[:, 0:5]
        label = label[:, 0:5]
        self.stat.feed((pred + 0.5).astype(np.int32), label)

    def _after_inference(self):
        return {self.prefix + '_precision': self.stat.precision,
                self.prefix + '_recall': self.stat.recall,
                self.prefix + '_f1_score': 2 * (self.stat.precision * self.stat.recall) / (1 * self.stat.precision + self.stat.recall),
                self.prefix + '_f2_score': 5 * (self.stat.precision * self.stat.recall) / (4 * self.stat.precision + self.stat.recall),
                }


def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    Args:
        logits: of shape (b, ...).
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(
            logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)


class Model(ModelDesc):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

    def inputs(self):
        return [tf.TensorSpec([None, self.config.shape, self.config.shape, 1], tf.float32, 'image'),
                tf.TensorSpec([None, self.config.types], tf.float32, 'label')
                ]

    def build_graph(self, image, label):
        image = image / 128.0 - 1.0

        if self.config.name == 'VGG16':
            output = VGG16(image, classes=self.config.types)
        elif self.config.name == 'ShuffleNet':
            output = ShuffleNet(image, classes=self.config.types)
        elif self.config.name == 'ResNet101':
            output = ResNet101(image, mode=self.config.mode, classes=self.config.types)
        elif self.config.name == 'DenseNet121':
            output = DenseNet121(image, classes=self.config.types)
        elif self.config.name == 'InceptionBN':
            output = InceptionBN(image, classes=self.config.types)
        else:
            pass

        logit = tf.sigmoid(output, name='logit')
        loss_xent = class_balanced_sigmoid_cross_entropy(output, label, name='loss_xent')

        # Visualization
        visualize_tensors('image', [image], scale_func=lambda x: x * 128.0 + 128.0, 
                          max_outputs=max(64, self.config.batch))
        # Regularize the weight of model 
        wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        cost = tf.add_n([loss_xent, wd_cost], name='cost')
        add_moving_summary(loss_xent)
        add_moving_summary(wd_cost)
        add_moving_summary(cost)
        return cost

    def optimizer(self):
        lrate = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        add_moving_summary(lrate)
        optim = tf.train.AdamOptimizer(lrate, beta1=0.5, epsilon=1e-3)
        return optim


def eval(model, sessinit, dataflow):
    """
    Eval a classification model on the dataset. It assumes the model inputs are
    named "input" and "label", and contains "logit" in the graph.
    """
    evaluator_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['image', 'label'],
        output_names=['logit']
    )

    stat = BinaryStatistics()

    # This does not have a visible improvement over naive predictor,
    # but will have an improvement if image_dtype is set to float32.
    evaluator = OfflinePredictor(evaluator_config)
    for dp in dataflow:
        image = dp[0]
        label = dp[1]
        estim = evaluator(image, label)[0]
        stat.feed((estim + 0.5).astype(np.int32), label)

    print('_precision: \t{}'.format(stat.precision))
    print('_recall: \t{}'.format(stat.recall))
    print('_f1_score: \t{}'.format(2 * (stat.precision *
                                        stat.recall) / (1 * stat.precision + stat.recall)))
    print('_f2_score: \t{}'.format(5 * (stat.precision *
                                        stat.recall) / (4 * stat.precision + stat.recall)))
    pass


def pred(model, sessinit, dataflow):
    """
    Eval a classification model on the dataset. It assumes the model inputs are
    named "input", and contains "logit" in the graph.
    """
    predictor_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['image'],
        output_names=['logit']
    )

    predictor = OfflinePredictor(predictor_config)
    estims = []
    for dp in dataflow:
        image = dp[0]
        estim = predictor(image)[0]
        estims.append(estim)

    return np.squeeze(np.array(estims))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--name', help='Model name', default='DenseNet121')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--pred', action='store_true', help='run prediction')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data', default='/u01/data/Vimmec_Data_small/', help='Data directory')
    parser.add_argument('--save', default='train_log/', help='Saving directory')
    parser.add_argument('--mode', default='none', help='Additional mode of resnet')
    
    parser.add_argument('--types', type=int, default=5)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--shape', type=int, default=256)

    config = parser.parse_args()

    if config.seed:
        os.environ['PYTHONHASHSEED']=str(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_random_seed(config.seed)

    if config.gpus:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    model = Model(config=config)

    if config.eval:
        ds_test2 = Vinmec(folder=config.data,
                          is_train='valid',
                          fname='valid.csv',
                          types=config.types,
                          resize=int(config.shape))

        ag_test2 = [
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]
        ds_test2.reset_state()
        ds_test2 = AugmentImageComponent(ds_test2, ag_test2, 0)
        ds_test2 = BatchData(ds_test2, 1)
        ds_test2 = PrintData(ds_test2)

        eval(model, SmartInit(config.load), ds_test2)
        sys.exit(0)

    elif config.pred:
        ds_test3 = Vinmec(folder=config.data,
                          is_train='test',
                          fname='test.csv',
                          types=config.types,
                          resize=int(config.shape))

        ag_test3 = [
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]
        ds_test3.reset_state()
        ds_test3 = AugmentImageComponent(ds_test3, ag_test3, 0)
        ds_test3 = BatchData(ds_test3, 1)
        ds_test3 = PrintData(ds_test3)

        estims = pred(model, SmartInit(config.load), ds_test3)

        # Read and write new csv
        fname = 'test.csv'
        csv_file = os.path.join(config.data, fname)
        df = pd.read_csv(csv_file)
        print(df)
        df = df['Images']
        tname = 'test_{}.csv'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(tname)
        # 0 ['Atelectasis']
        # 1 ['Cardiomegaly']
        # 2 ['Consolidation']
        # 3 ['Edema']
        # 4 ['Pleural Effusion']
        df['Pleural_Effusion'] = pd.Series(estims[:, 4], index=df.index)
        df['Edema'] = pd.Series(estims[:, 3], index=df.index)
        df['Consolidation'] = pd.Series(estims[:, 2], index=df.index)
        df['Atelectasis'] = pd.Series(estims[:, 0], index=df.index)
        df['Cardiomegaly'] = pd.Series(estims[:, 1], index=df.index)
        df.to_csv(tname, index=False)
        sys.exit(0)

    else:
        logger.set_logger_dir(os.path.join(
            config.save, config.name, config.mode, str(config.shape), str(config.types), ), 'd')

        # Setup the dataset for training
        ds_train = Vinmec(folder=config.data,
                          is_train='train',
                          fname='train.csv',
                          types=config.types,
                          resize=int(config.shape))

        ds_valid = Vinmec(folder=config.data,
                          is_train='train',
                          fname='valid.csv',
                          types=config.types,
                          resize=int(config.shape))

        ds_other = Vinmec(folder='/u01/data/CXR/CheXpert-v1.0-small/',
                          is_train='train',
                          fname='valid_chexpert_vinmec_format.csv',
                          types=config.types,
                          resize=int(config.shape))

        ds_train = ConcatData([ds_train, ds_valid, ds_other])
        ag_train = [
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.RotationAndCropValid(max_deg=25),
            imgaug.Albumentations(AB.CLAHE(p=0.5)),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0),
                                                aspect_ratio_range=(0.8, 1.2),
                                                interp=cv2.INTER_LINEAR, target_shape=config.shape),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from
                 # fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]
        ds_train.reset_state()
        ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        ds_train = BatchData(ds_train, config.batch)
        ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=2)
        ds_train = PrintData(ds_train)

        # Setup the dataset for validating
        ds_test2 = Vinmec(folder=config.data,
                          is_train='valid',
                          fname='valid.csv',
                          types=config.types,
                          resize=int(config.shape))

        ag_test2 = [
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]
        ds_test2.reset_state()
        ds_test2 = AugmentImageComponent(ds_test2, ag_test2, 0)
        ds_test2 = BatchData(ds_test2, config.batch)
        # ds_test2 = MultiProcessRunnerZMQ(ds_test2, num_proc=1)
        ds_test2 = PrintData(ds_test2)

        # Setup the config
        config = TrainConfig(
            model=model,
            dataflow=ds_train,
            callbacks=[
                ModelSaver(),
                MinSaver('cost'),
                ScheduledHyperParamSetter('learning_rate',
                                          [(0, 1e-2), (20, 1e-3), (50, 1e-4), (100, 1e-5), (150, 1e-6)]),
                InferenceRunner(ds_test2, [CustomBinaryClassificationStats('logit', 'label'),
                                           ScalarStats('loss_xent'),
                                           ])
            ],
            max_epoch=300,
            session_init=SmartInit(config.load),
        )

        trainer = SyncMultiGPUTrainerParameterServer(max(get_num_gpu(), 1))

        launch_train_with_config(config, trainer)

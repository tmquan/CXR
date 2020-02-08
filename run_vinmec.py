# coding=utf-8
# Author: Tran Minh Quan
import cv2
import numpy as np
import tensornets as tn
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils.stats import BinaryStatistics
import albumentations as AB
from vinmec import Vinmec
import argparse
import os

from absl import flags
import tensorflow as tf
tf.disable_v2_behavior()
from tensorlayer.cost import binary_cross_entropy, dice_coe


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
                self.prefix + '_f1_score': 2 * (self.stat.precision * self.stat.recall) / (1*self.stat.precision + self.stat.recall),
                self.prefix + '_f2_score': 5 * (self.stat.precision * self.stat.recall) / (4*self.stat.precision + self.stat.recall),
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
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)

class Model(ModelDesc):
    """[summary]
    [description]
    Extends:
        ModelDesc
    """

    def __init__(self, config):
        """[summary]
        [description]
        Keyword Arguments:
            name {str} -- [description] (default: {'DenseNet121'})
        """
        super(Model, self).__init__()
        self.config = config

    def inputs(self):
        """[summary]
        [description]
        """
        return [tf.TensorSpec([None, self.config.shape, self.config.shape, 1], tf.float32, 'image'),
                tf.TensorSpec([None, self.config.types], tf.float32, 'label')
                ]

    def build_graph(self, image, label):
        image = image / 128.0 - 1.0
        assert tf.test.is_gpu_available()
        # with tf.name_scope('cnn'):
        if self.config.name == 'DenseNet121':
            models = tn.DenseNet121(image, is_training=self.training, classes=self.config.types)
        elif self.config.name == 'DenseNet169':
            models = tn.DenseNet169(image, is_training=self.training, classes=self.config.types)
        elif self.config.name == 'DenseNet201':
            models = tn.DenseNet201(image, is_training=self.training, classes=self.config.types)
        elif self.config.name == 'ResNet101':
            models = tn.ResNet101(image, is_training=self.training, classes=self.config.types)
        elif self.config.name == 'InceptionResNet2':
            models = tn.InceptionResNet2(image, is_training=self.training, classes=self.config.types)
        elif self.config.name == 'VGG19':
            models = tn.VGG19(image, is_training=self.training, classes=self.config.types)
        else:
            pass

        models.print_outputs()
        output = tf.tanh(models.logits)

        logit = tf.identity(output / 2.0 + 0.5, name='logit') #tf.sigmoid(output, name='logit')
        loss_xent = class_balanced_sigmoid_cross_entropy(output, label, name='loss_xent')

        # Visualization
        wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        visualize_tensors('image', [image], scale_func=lambda x: x * 128.0 + 128.0, max_outputs=max(64, self.config.batch))
        cost = tf.add_n([loss_xent, wd_cost], name='cost')
        add_moving_summary(loss_xent)
        add_moving_summary(wd_cost)
        add_moving_summary(cost)
        return cost

    def optimizer(self):
        lrate = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        # optim = tf.train.MomentumOptimizer(lrate, 0.9, use_nesterov=True)
        optim = tf.train.AdamOptimizer(lrate, beta1=0.5, epsilon=1e-3)
        return optim


def eval():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--name', help='Model name', default='DenseNet121')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--pred', action='store_true', help='run prediction')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data', default='/u01/data/Vimmec_Data_small/', help='Data directory')
    parser.add_argument('--save', default='train_log/', help='Saving directory')
    parser.add_argument('--types', type=int, default=6)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--shape', type=int, default=256)

    config = parser.parse_args()
    if config.gpus:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    model = Model(config=config)

    if config.eval:
        pass
    else:
        logger.set_logger_dir(os.path.join(
            config.save, config.name, str(config.shape), str(config.types), ), 'd')

        # Setup the dataset for training
        ds_train = Vinmec(folder=config.data, train_or_valid='train', fname='train.csv', types=config.types, resize=int(config.shape))
        ag_train = [
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.RotationAndCropValid(max_deg=25),
            # imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.6, 1.0),
                                                aspect_ratio_range=(0.8, 1.2),
                                                interp=cv2.INTER_LINEAR, target_shape=config.shape),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]
        # ds_train.reset_state()
        # ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        # ds_train = BatchData(ds_train, config.batch)
        # ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=2)
        # ds_train = PrintData(ds_train)
        # ds_train = FixedSizeData(ds_train, 200) # For debugging purpose

        # Setup the dataset for validating
        ds_valid = Vinmec(folder=config.data, train_or_valid='train', fname='valid.csv', types=config.types, resize=int(config.shape))

        ag_valid = [
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            imgaug.RotationAndCropValid(max_deg=25),
            # imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.6, 1.0),
                                                aspect_ratio_range=(0.8, 1.2),
                                                interp=cv2.INTER_LINEAR, target_shape=config.shape),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]
        # ds_valid.reset_state()
        # ds_valid = AugmentImageComponent(ds_valid, ag_valid, 0)
        # ds_valid = BatchData(ds_valid, config.batch)
        # # ds_valid = MultiProcessRunnerZMQ(ds_valid, num_proc=1)
        # ds_valid = PrintData(ds_valid)
        ds_train = ConcatData([ds_train, ds_valid])
        ds_train.reset_state()
        ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        ds_train = BatchData(ds_train, config.batch)
        ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=2)
        ds_train = PrintData(ds_train)

         # Setup the dataset for validating
        ds_test2 = Vinmec(folder=config.data, train_or_valid='valid', fname='valid.csv', types=config.types, resize=int(config.shape))

        ag_test2 = [
            # imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            # imgaug.Albumentations(AB.CLAHE(p=1)),
            # imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
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
                PeriodicTrigger(ModelSaver(), every_k_epochs=5),
                PeriodicTrigger(MinSaver('cost'), every_k_epochs=10),
                ScheduledHyperParamSetter('learning_rate',
                                          [(0, 1e-2), (20, 1e-3), (50, 1e-4), (100, 1e-5)]), #, interp='linear'
                InferenceRunner(ds_test2, [CustomBinaryClassificationStats('logit', 'label'),
                                           ScalarStats('loss_xent'),
                                           ])
            ],
            max_epoch=300,
            session_init=SmartInit(config.load),
        )

        trainer = SyncMultiGPUTrainerParameterServer(max(get_num_gpu(), 1))

        launch_train_with_config(config, trainer)

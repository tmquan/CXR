import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, SmartInit, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils import logger

import numpy as np


# DenseNet net
# @layer_register(log_shape=True)
def conv(name, l, channel, stride):
    return Conv2D(name, l, channel, 3, stride=stride,
                  nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))

# @layer_register(log_shape=True)    
def add_layer(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    growthRate = 12
    with tf.variable_scope(name) as scope:
        c = BatchNorm('bn1', l)
        c = tf.nn.relu(c)
        c = conv('conv1', c, growthRate, 1)
        l = tf.concat([c, l], 3)
    return l

# @layer_register(log_shape=True)
def add_transition(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        l = BatchNorm('bn1', l)
        l = tf.nn.relu(l)
        l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
        l = AvgPooling('pool', l, 2)
    return l


def DenseNet(image, classes=5):
    depth = 40
    N = int((depth - 4)  / 3)
    growthRate = 12
    l = conv('conv0', image, 16, 1)
    with tf.variable_scope('block1') as scope:

        for i in range(N):
            l = add_layer('dense_layer.{}'.format(i), l)
        l = add_transition('transition1', l)

    with tf.variable_scope('block2') as scope:

        for i in range(N):
            l = add_layer('dense_layer.{}'.format(i), l)
        l = add_transition('transition2', l)

    with tf.variable_scope('block3') as scope:

        for i in range(N):
            l = add_layer('dense_layer.{}'.format(i), l)
    l = BatchNorm('bnlast', l)
    l = tf.nn.relu(l)
    l = GlobalAvgPooling('gap', l)
    output = FullyConnected('linear', l, out_dim=classes, nl=tf.identity)

    return output

import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu

import tensornets as tn

from chexpert import *
"""
To train:
    ./cifar10-resnet.py --gpu 0,1
"""

BATCH = 128
SHAPE = 224

class Model(ModelDesc):

    def __init__(self, name='ResNet50'):
        super(Model, self).__init__()
        self.name = name
        if self.name=='ResNet50':
            net = tn.ResNet50
        elif self.name=='ResNet101':
            net = tn.ResNet101
        elif self.name=='ResNet152':
            net = tn.ResNet152
        else:
            pass

    def inputs(self):
        return [tf.TensorSpec([None, SHAPE, SHAPE, 1], tf.float32, 'image'),
                tf.TensorSpec([None, 14], tf.float32, 'label')]

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()
        
        logit = tn.ResNet50(image, stem=True)
        logit = tf.reduce_mean(logit, [1, 2], name='avgpool')
        logit = FullyConnected('fc', logit, 14, activation='sigmoid')
        
        def dice_loss(y_true, y_pred):
            num = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
            den = tf.reduce_sum(y_true + y_pred, axis=-1)

            return 1 - (num) / (den + 1e-6)
        cost = tf.reduce_mean(dice_loss(label, logit), name='cost')
        # cost = 1.0 - tf.contrib.metrics.streaming_auc(tf.squeeze(label), tf.squeeze(logit))
        # tf.nn.sigmoid(logits, name='output')

        # cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        # cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        # wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_vector')
        # monitor training error
        # add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          500000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost)
        add_moving_summary(wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')
        # return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data():
    ds_train = Chexpert(folder='/u01/data/CheXpert-v1.0-small', 
        train_or_valid='train',
        resize=SHAPE,
        )
    ds_train.reset_state()
    ds_train = BatchData(ds_train, BATCH)
    ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=8)
    
    ds_valid = Chexpert(folder='/u01/data/CheXpert-v1.0-small', 
        train_or_valid='valid',
        resize=SHAPE,
        )
    ds_valid.reset_state()
    ds_valid = BatchData(ds_valid, BATCH)
    ds_valid = MultiProcessRunnerZMQ(ds_valid, num_proc=8)
    

    aug_train_image = [
        imgaug.CenterPaste((SHAPE*1.12, SHAPE*1.12)),
        imgaug.RandomCrop((SHAPE, SHAPE)),
        # imgaug.MapImage(lambda x: x - pp_mean),
    ]
    # aug_train_label = [
    #     imgaug.MapImage(lambda x: x[x==-1.0] = np.random.normal(len(x==-1.0))), # uncertainty
    #     imgaug.MapImage(lambda x: x[x==-2.0] = np.random.uniform(len(x==-2.0))) # unmentioned
    # ]


    # aug_valid_image = [
    #     imgaug.MapImage(lambda x: x - pp_mean)
    # ]

    # ds_train = AugmentImageComponent(ds_train, aug_train_image, 0)
    # ds_valid = AugmentImageComponent(ds_valid, aug_valid_image, 0)

    return ds_train, ds_valid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='Image directory')
    parser.add_argument('--model', help='Model name', default='ResNet152')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--shape', type=int, default=320)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch
    SHAPE = args.shape

    logger.auto_set_dir()

    ds_train, ds_valid = get_data()
    model=Model(name=args.model)
    # ds_train = PrintData(ds_train)
    config = TrainConfig(
        model=model,
        dataflow=ds_train,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=100),
            InferenceRunner(ds_valid,
                            [ScalarStats('cost'), 
                             # ClassificationError('wrong_vector')
                             ]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (100, 0.01), (200, 0.001), (300, 0.0002)])
        ],
        max_epoch=400,
        session_init=SmartInit(args.load),
    )

    trainer = SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))

    # trainer.train_with_defaults()
    launch_train_with_config(config, trainer) 
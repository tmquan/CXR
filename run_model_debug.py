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
SHAPE = 320

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
        
        def dice_loss(y_pred, y_true):
            num = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
            den = tf.reduce_sum(y_true + y_pred, axis=-1)

            return 1 - (num) / (den + 1e-6)
        cost = tf.reduce_mean(dice_loss(logit, label), name='cost')
        auc, auc_update_op = tf.metrics.auc( predictions=logit, labels=label, curve = 'ROC' )
        auc_variables = [ v for v in tf.local_variables() if v.name.startswith( "AUC" ) ]

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
        # add_moving_summary(auc_variables)
        [ add_moving_summary(v) for v in tf.local_variables() if v.name.startswith( "AUC" ) ]


        # Visualization
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        visualize_tensors('image', [image], max_outputs=max(64, BATCH), scale_func=lambda x: (x) * 255)
        return tf.add_n([cost, wd_cost], name='cost')
        # return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(folder='/u01/data/CheXpert-v1.0-small'):
    ds_train = Chexpert(folder=folder, 
        train_or_valid='train',
        resize=2*SHAPE,
        )
    
    
    ds_valid = Chexpert(folder=folder, 
        train_or_valid='valid',
        resize=2*SHAPE,
        )
   
    # aug_train_image = [
    #     imgaug.CenterPaste((SHAPE*1.12, SHAPE*1.12)),
    #     imgaug.RandomCrop((SHAPE, SHAPE)),
    #     # imgaug.MapImage(lambda x: x - pp_mean),
    # ]
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

def get_augmentation():
    aug_train = [
        # It's OK to remove the following augs if your CPU is not fast enough.
        # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
        # Removing lighting leads to a tiny drop in accuracy.
        imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB), 
        # imgaug.ToUint8(),
        imgaug.RotationAndCropValid(max_deg=5,),
        # imgaug.ToUint8(),

        # imgaug.Resize((SHAPE*1.12, SHAPE*1.12)),
        imgaug.RandomOrderAug(
            [
                imgaug.BrightnessScale((0.6, 1.4), clip=False),
                imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
                imgaug.Saturation(0.4, rgb=False),
                # rgb-bgr conversion for the constants copied from fb.resnet.torch
                imgaug.Lighting(0.1,
                                eigval=np.asarray(
                                 [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                eigvec=np.array(
                                 [[-0.5675, 0.7192, 0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948, 0.4203]],
                                 dtype='float32')[::-1, ::-1]
                                )
                ]),
        imgaug.ToUint8(),
        imgaug.GoogleNetRandomCropAndResize(interp=cv2.INTER_LINEAR, target_shape=SHAPE),
        # imgaug.ToUint8(),
        imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY)
        # imgaug.ToUint8(),
        # imgaug.RandomCrop((SHAPE, SHAPE)),
    ]

    aug_valid = [
        imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
        # imgaug.ToUint8(),
        # imgaug.Resize((SHAPE*1.12, SHAPE*1.12), interp=cv2.INTER_LINEAR),
        # imgaug.ToUint8(),
        imgaug.RandomCrop((SHAPE, SHAPE)),
        # imgaug.ToUint8(),
        imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY)
    ]

    return aug_train, aug_valid
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data', help='Image directory')
    parser.add_argument('--model', help='Model name', default='ResNet152')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--shape', type=int, default=320)
    parser.add_argument('--sample', action='store_true', help='run sampling')
    
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch
    SHAPE = args.shape

    # logger.auto_set_dir()
    logger.set_logger_dir(os.path.join('train_log', args.model), 'd')

    ds_train, ds_valid = get_data(folder=args.data)
    ag_train, ag_valid = get_augmentation()

    ds_train = PrintData(ds_train)
    ds_valid = PrintData(ds_valid)

    ds_train.reset_state()
    ds_valid.reset_state()

    ds_train = AugmentImageComponent(ds_train, ag_train, 0)
    ds_valid = AugmentImageComponent(ds_valid, ag_valid, 0)
    
    ds_train = BatchData(ds_train, BATCH)
    ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=8)
    
    ds_valid = BatchData(ds_valid, BATCH)
    ds_valid = MultiProcessRunnerZMQ(ds_valid, num_proc=8)
    

    
    model = Model(name=args.model)
    # ds_train = PrintData(ds_train)
    config = TrainConfig(
        model=model,
        dataflow=ds_train,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=100),
            InferenceRunner(ds_valid,
                            [   ScalarStats('cost'), 
                                # ScalarStats('AUC*'), 
                                # [ ScalarStats(v)  for v in tf.local_variables() if v.name.startswith( "AUC" )]
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
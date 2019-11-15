import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.utils.gpu import get_num_gpu

import tensornets as tn

from chexpert import *
"""
To train:
    ./cifar10-resnet.py --gpu 0,1
"""

BATCH = 128
SHAPE = 320
GROUP = 14

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
    def __init__(self, name='ResNet50', mode='small'):
        super(Model, self).__init__()
        self.name = name
        self.mode = mode
        if self.name=='ResNet50':
            net = tn.ResNet50
        elif self.name=='ResNet101':
            net = tn.ResNet101
        elif self.name=='ResNet152':
            net = tn.ResNet152
        else:
            pass

    def inputs(self):
        if self.mode== 'space_to_depth':
            return [tf.TensorSpec([None, 3072, 3072, 1], tf.float32, 'image'),
                    tf.TensorSpec([None, GROUP], tf.float32, 'label')]
        else:
            return [tf.TensorSpec([None, SHAPE, SHAPE, 1], tf.float32, 'image'),
                    tf.TensorSpec([None, GROUP], tf.float32, 'label')]
        

    def build_graph(self, image, label):
        if self.mode== 'space_to_depth':
            image = tf.space_to_depth(image, SHAPE)

        image = image / 128.0 - 1
        assert tf.test.is_gpu_available()
        
        logit = tn.ResNet50(image, stem=True)
        logit = tf.reduce_mean(logit, [1, 2], name='avgpool')
        logit = FullyConnected('fc', logit, GROUP, activation='sigmoid')
        logit = tf.identity(logit, name='logit')
        def dice_loss(predictions, labels):
            num = 2 * tf.reduce_sum(labels * predictions, axis=-1)
            den = tf.reduce_sum(labels + predictions, axis=-1)

            return 1 - (num) / (den + 1e-6)
        loss = tf.reduce_mean(dice_loss(predictions=logit, labels=label), name='loss')
        auc, update_op = tf.metrics.auc(predictions=logit, labels=label) 
        auc_variables = [ v for v in tf.local_variables() if v.name.startswith( "auc" ) ]
        auc_reset_op = tf.initialize_variables( auc_variables )
     
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        # loss = tf.reduce_mean(loss, name='cross_entropy_loss')

        # wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_vector')
        # monitor training error
        # add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          500000, 0.2, True)
        wd_loss = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_loss') 
        add_moving_summary(auc)
        add_moving_summary(update_op)
        add_moving_summary(loss)
        add_moving_summary(wd_loss)

        # Visualization
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        visualize_tensors('image', [image], max_outputs=max(64, BATCH))
        cost = tf.add_n([loss, wd_loss], name='cost')
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(folder='/u01/data/CheXpert-v1.0-small', mode='small', group=14, debug=False):
    if mode=='small':
        _shape = SHAPE
    else:
        _shape = 1024*3
    ds_train = Chexpert(folder=folder, 
        train_or_valid='train',
        group=group,
        resize=int(1.05*_shape),
        debug=debug
        )
    
    
    ds_valid = Chexpert(folder=folder, 
        train_or_valid='valid',
        group=group,
        resize=int(1.05*_shape),
        debug=debug
        )
 
    return ds_train, ds_valid

def get_augmentation():
    aug_train = [
        # It's OK to remove the following augs if your CPU is not fast enough.
        # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
        # Removing lighting leads to a tiny drop in accuracy.
        # imgaug.Resize((SHAPE*1.12, SHAPE*1.12)),
        # imgaug.ToUint8(),
        imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB), 
        imgaug.Rotation(max_deg=15,),
        # imgaug.ToUint8(),

        
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
        # imgaug.ToUint8(),
        imgaug.GoogleNetRandomCropAndResize(interp=cv2.INTER_LINEAR, target_shape=SHAPE),
        # imgaug.ToUint8(),
        imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY)
        # imgaug.ToUint8(),
        # imgaug.RandomCrop((SHAPE, SHAPE)),
    ]

    aug_valid = [
        # imgaug.Resize((SHAPE*1.12, SHAPE*1.12), interp=cv2.INTER_LINEAR),
        # imgaug.ToUint8(),
        imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
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
    parser.add_argument('--mode', help='small | random_patch | space_to_depth', default='small')
    parser.add_argument('--debug', help='Small size', action='store_true')
    parser.add_argument('--model', help='Model name', default='ResNet50')
    parser.add_argument('--group', type=int, default=14)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--shape', type=int, default=320)
    parser.add_argument('--sample', action='store_true', help='run sampling')
    
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch
    SHAPE = args.shape
    GROUP = args.group
    DEBUG = args.debug

    # logger.auto_set_dir()
    logger.set_logger_dir(os.path.join('train_log', args.model, 'data_'+args.mode, 'shape_'+str(SHAPE), 'group_'+str(GROUP), ), 'd')

    ds_train, ds_valid = get_data(folder=args.data, group=GROUP, debug=DEBUG)
    ag_train, ag_valid = get_augmentation()
   
    ds_train.reset_state()
    ds_valid.reset_state()

    ds_train = AugmentImageComponent(ds_train, ag_train, 0)
    ds_valid = AugmentImageComponent(ds_valid, ag_valid, 0)
    
    ds_train = BatchData(ds_train, BATCH)
    ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=8)
    
    ds_valid = BatchData(ds_valid, BATCH)
    ds_valid = MultiProcessRunnerZMQ(ds_valid, num_proc=1)
    
    ds_train = PrintData(ds_train)
    ds_valid = PrintData(ds_valid)
    
    model = Model(name=args.model)
    # ds_train = PrintData(ds_train)
    config = TrainConfig(
        model=model,
        dataflow=ds_train,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=10),
            InferenceRunner(ds_valid,
                            [   BinaryClassificationStats('logit', 'label'),
                                ScalarStats('auc/value'), 
                                ScalarStats('auc/update_op'), 
                                ScalarStats('cost'), 
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
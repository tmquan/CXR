import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.utils.gpu import get_num_gpu

import tensornets as tn
import sklearn
# from tensorlayer.cost import * #binary_cross_entropy, absolute_difference_error, dice_coe, cross_entropy
from chexpert import *
"""
To train:
    
"""

BATCH = 128
SHAPE = 320
GROUP = 14
TRAIN = True
class AUCStatistics(object):
    """
    Statistics for binary decision,
    including precision, recall, false positive, false negative
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.nr_pos = 0  # positive label
        self.nr_neg = 0  # negative label
        self.nr_pred_pos = 0
        self.nr_pred_neg = 0
        self.corr_pos = 0   # correct predict positive
        self.corr_neg = 0   # correct predict negative
        self.auc = 0
        self.shape = [-1, -1]

    def feed(self, pred, label):
        """
        Args:
            pred (np.ndarray): binary array.
            label (np.ndarray): binary array of the same size.
        """
        assert pred.shape == label.shape, "{} != {}".format(pred.shape, label.shape)
        pred = (pred > 0.5)*1 # Threshold
        label = (label > 0.5)*1 # Threshold
        self.shape = pred.shape
        self.nr_pos += (label == 1).sum()
        self.nr_neg += (label == 0).sum()
        self.nr_pred_pos += (pred == 1).sum()
        self.nr_pred_neg += (pred == 0).sum()
        self.corr_pos += ((pred == 1) & (pred == label)).sum()
        self.corr_neg += ((pred == 0) & (pred == label)).sum()
        try:
            self.auc = sklearn.metrics.roc_auc_score(y_true=label, y_score=pred)
        except:
            self.auc = -1.0
            pass

    @property
    def precision(self):
        if self.nr_pred_pos == 0:
            return 0
        return self.corr_pos * 1. / self.nr_pred_pos

    @property
    def recall(self):
        if self.nr_pos == 0:
            return 0
        return self.corr_pos * 1. / self.nr_pos

    @property
    def false_positive(self):
        if self.nr_pred_pos == 0:
            return 0
        return 1 - self.precision

    @property
    def false_negative(self):
        if self.nr_pos == 0:
            return 0
        return 1 - self.recall

    @property
    def roc(self):
        return self.auc
    
class AUCStats(Inferencer):
    """
    Compute precision / recall in binary classification, given the
    prediction vector and the label vector.
    """

    def __init__(self, pred_tensor_name, label_tensor_name, prefix='val'):
        """
        Args:
            pred_tensor_name(str): name of the 0/1 prediction tensor.
            label_tensor_name(str): name of the 0/1 label tensor.
        """
        self.pred_tensor_name = pred_tensor_name
        self.label_tensor_name = label_tensor_name
        self.prefix = prefix

    def _before_inference(self):
        self.stat = AUCStatistics()

    def _get_fetches(self):
        return [self.pred_tensor_name, self.label_tensor_name]

    def _on_fetches(self, outputs):
        pred, label = outputs
        self.stat.feed(pred, label)

    def _after_inference(self):
        return {
                # self.prefix + '_shape[0]': self.stat.shape[0],
                # self.prefix + '_shape[1]': self.stat.shape[1],
                self.prefix + '_nr_pos': self.stat.nr_pos,
                self.prefix + '_nr_neg': self.stat.nr_neg,
                self.prefix + '_nr_pred_pos': self.stat.nr_pred_pos,
                self.prefix + '_nr_pred_neg': self.stat.nr_pred_neg,
                self.prefix + '_corr_pos': self.stat.corr_pos,
                self.prefix + '_corr_neg': self.stat.corr_neg,
                self.prefix + '_precision': self.stat.precision,
                self.prefix + '_recall': self.stat.recall, 
                self.prefix + '_roc': self.stat.roc
                }
 
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

def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)

class Model(ModelDesc):
    def __init__(self, name='ResNet50', mode='small'):
        super(Model, self).__init__()
        self.name = name
        self.mode = mode
        if self.name=='ResNet50':
            self.net = tn.ResNet50
        elif self.name=='ResNet101':
            self.net = tn.ResNet101
        elif self.name=='ResNet152':
            self.net = tn.ResNet152

        if self.name=='ResNet50v2':
            self.net = tn.ResNet50v2
        elif self.name=='ResNet101v2':
            self.net = tn.ResNet101v2
        elif self.name=='ResNet152v2':
            self.net = tn.ResNet152v2

        if self.name=='ResNeXt50c32':
            self.net = tn.ResNeXt50c32
        elif self.name=='ResNeXt101c32':
            self.net = tn.ResNeXt101c32
        elif self.name=='ResNeXt101c64':
            self.net = tn.ResNeXt101c64

        elif self.name=='DenseNet121':
            self.net = tn.DenseNet121
        elif self.name=='DenseNet169':
            self.net = tn.DenseNet169
        else:
            pass

    def inputs(self):
        if self.mode== 'space_to_depth':
            return [tf.TensorSpec([None, 3072, 3072, 1], tf.float32, 'image'),
                    tf.TensorSpec([None, GROUP], tf.float32, 'label'), 
                    tf.TensorSpec([None, 320, 320, 1], tf.float32, 'fname'),]
        else:
            return [tf.TensorSpec([None, SHAPE, SHAPE, 1], tf.float32, 'image'),
                    tf.TensorSpec([None, GROUP], tf.float32, 'label'), 
                    tf.TensorSpec([None, SHAPE, SHAPE, 1], tf.float32, 'fname'),
                    ]
        

    def build_graph(self, image, label, fname):
        if self.mode== 'space_to_depth':
            image = tf.space_to_depth(image, SHAPE)
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        fname = tf.cast(fname, tf.float32)
        image = image / 128.0 - 1.0
        fname = fname / 128.0 - 1.0
        assert tf.test.is_gpu_available()
        
        # with argscope([Conv2D, BatchNorm, GlobalAvgPooling], data_format='channels_last'), \
        #         argscope(Conv2D, kernel_initializer=tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')):
        with tf.variable_scope('Model'):
            output = self.net(image, stem=True, is_training=True)
            # if self.name=='ResNeXt50c32':
            #     output = tn.ResNeXt50c32(image, stem=True, classes=GROUP, is_training=True)
            # output = tf.reduce_mean(output, [1, 2], name='avgpool')
            # output.print_outputs()
            output = GlobalAvgPooling('gap', output)
            output = FullyConnected('fc1', output, 1024, activation=BNLReLU,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
            output = FullyConnected('fc', output, 512, activation=BNLReLU,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
        linear = FullyConnected('linear', output, GROUP, activation=BNLReLU,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

        loss_xentropy = tf.losses.sigmoid_cross_entropy(label, linear, reduction=tf.losses.Reduction.NONE)
        # loss_xentropy = tf.identity(loss_xentropy, name='loss_xentropy')
        loss_xentropy = tf.reduce_mean(loss_xentropy, name='loss_xentropy')

        logit = tf.sigmoid(linear, name='logit')
        def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
            """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
            of two batch of data, usually be used for binary image segmentation
            i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

            Parameters
            -----------
            output : Tensor
                A distribution with shape: [batch_size, ....], (any dimensions).
            target : Tensor
                The target distribution, format the same with `output`.
            loss_type : str
                ``jaccard`` or ``sorensen``, default is ``jaccard``.
            axis : tuple of int
                All dimensions are reduced, default ``[1,2,3]``.
            smooth : float
                This small value will be added to the numerator and denominator.
                    - If both output and target are empty, it makes sure dice is 1.
                    - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

            Examples
            ---------
            >>> import tensorlayer as tl
            >>> outputs = tl.act.pixel_wise_softmax(outputs)
            >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

            References
            -----------
            - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

            """
            inse = tf.reduce_sum(output * target, axis=axis)
            if loss_type == 'jaccard':
                l = tf.reduce_sum(output * output, axis=axis)
                r = tf.reduce_sum(target * target, axis=axis)
            elif loss_type == 'sorensen':
                l = tf.reduce_sum(output, axis=axis)
                r = tf.reduce_sum(target, axis=axis)
            else:
                raise Exception("Unknow loss_type")
            
            dice = (2. * inse + smooth) / (l + r + smooth) # 1 minus
            ##
            dice = 1 - tf.reduce_mean(dice, name='dice_coe')
            return dice

        loss_dice = tf.identity(dice_coe(logit, label, axis=(1), loss_type='jaccard'), name='loss_dice') 
        
        # # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          500000, 0.2, True)
        wd_loss = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_loss') 
        add_moving_summary(loss_xentropy)
        add_moving_summary(loss_dice)
        add_moving_summary(wd_loss)

        # Visualization
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        visualize_tensors('image', [image, fname], max_outputs=max(64, BATCH))
        cost = tf.add_n([loss_xentropy, loss_dice, wd_loss], name='cost')
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
        resize=int(_shape),
        debug=debug
        )
    
    
    ds_valid = Chexpert(folder=folder, 
        train_or_valid='valid',
        group=group,
        resize=int(_shape),
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
        imgaug.RandomPaste((int(1.15*SHAPE), int(1.15*SHAPE))),
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
        # imgaug.GoogleNetRandomCropAndResize(interp=cv2.INTER_LINEAR, target_shape=SHAPE),
        # imgaug.ToUint8(),
        imgaug.RandomCrop((SHAPE, SHAPE)),
        imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY)
        # imgaug.ToUint8(),
    ]

    aug_valid = [
        # imgaug.Resize((SHAPE*1.12, SHAPE*1.12), interp=cv2.INTER_LINEAR),
        # imgaug.ToUint8(),
        imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
        imgaug.RandomPaste((int(1.15*SHAPE), int(1.15*SHAPE))),
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
    parser.add_argument('--group', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--shape', type=int, default=320)
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch
    SHAPE = args.shape
    GROUP = args.group
    DEBUG = args.debug
    TRAIN = args.train

    # logger.auto_set_dir()
    logger.set_logger_dir(os.path.join('train_log', args.model, 'data'+args.mode, 'shape'+str(SHAPE), 'group'+str(GROUP), ), 'd')

    ds_train, ds_valid = get_data(folder=args.data, group=GROUP, debug=DEBUG)
    ag_train, ag_valid = get_augmentation()
   
    ds_train.reset_state()
    ds_valid.reset_state()

    ds_train = AugmentImageComponent(ds_train, ag_train, 0)
    ds_valid = AugmentImageComponent(ds_valid, ag_valid, 0) 

    ds_train = BatchData(ds_train, BATCH)
    ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=8)
    
    ds_valid = BatchData(ds_valid, BATCH)
    # ds_valid = MultiProcessRunnerZMQ(ds_valid, num_proc=1)
    ds_train = PrintData(ds_train)
    ds_valid = PrintData(ds_valid)
    
    model = Model(name=args.model)
    # ds_train = PrintData(ds_train)
    config = TrainConfig(
        model=model,
        dataflow=ds_train,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=10),
            # InferenceRunner(ds_train,
            #                 [AUCStats('logit', 'label', prefix='train'),]),
            InferenceRunner(ds_valid,
                            [ScalarStats('loss_dice'), ScalarStats('loss_xentropy'),
                            AUCStats('logit', 'label', prefix='valid'),]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(0, 1e-2), (50, 1e-3), (100, 1e-4), (150, 1e-5)])
        ],
        max_epoch=200,
        session_init=SmartInit(args.load),
    )

    trainer = SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))

    # trainer.train_with_defaults()
    launch_train_with_config(config, trainer) 
# coding=utf-8
import argparse
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensornets as tn

from contextlib import contextmanager
from tensorpack import *
from tensorpack.contrib.keras import KerasPhaseCallback
from tensorpack.contrib.keras import KerasModel
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.utils.argtools import memoized
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.tower import get_current_tower_context
import albumentations as AB
# import tensornets as tn
import sklearn
# from tensorlayer.cost import * #binary_cross_entropy, absolute_difference_error, dice_coe, cross_entropy
from chexpert import *
"""
To train:
    
"""

BATCH = 128
SHAPE = 640
GROUP = 14
TRAIN = True

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

    def inputs(self):
        return [tf.TensorSpec([None, SHAPE, SHAPE, 1], tf.float32, 'image'),
                tf.TensorSpec([None, GROUP], tf.float32, 'label'), 
                ]
        

    def build_graph(self, image, label):
        image = image / 255.0 
        assert tf.test.is_gpu_available()
        with tf.name_scope('cnn'):
            if self.name=='ResNeXt50c32':
                models = tn.ResNeXt50c32(image, stem=True, is_training=True, classes=GROUP)
            elif self.name=='ResNeXt101c32':
                models = tn.ResNeXt101c32(image, stem=True, is_training=True, classes=GROUP)
            elif self.name=='ResNeXt101c64':
                models = tn.ResNeXt101c64(image, stem=True, is_training=True, classes=GROUP)
            elif self.name=='DenseNet121':
                models = tn.DenseNet121(image, stem=True, is_training=True, classes=GROUP)
            elif self.name=='DenseNet169':
                models = tn.DenseNet169(image, stem=True, is_training=True, classes=GROUP)
            elif self.name=='DenseNet201':
                models = tn.DenseNet201(image, stem=True, is_training=True, classes=GROUP)
            elif self.name=='NASNetAlarge':
                models = tn.NASNetAlarge(image, stem=True, is_training=True, classes=GROUP)
            elif self.name=='PNASNetlarge':
                models = tn.PNASNetlarge(image, stem=True, is_training=True, classes=GROUP)
            elif self.name=='InceptionResNet2':
                models = tn.InceptionResNet2(image, stem=True, is_training=True, classes=GROUP)
            else:
                pass
            # output = tf.identity(models.logits)
        output = tf.identity(models)
        output = Dropout('dropout_stem', output, rate=0.5)
        output = GlobalAvgPooling('gap', output)
        # output = Dropout('dropout_pool', output, rate=0.5)
        output = FullyConnected('fc1024', output, 1024, activation=tf.nn.relu)
        # output = Dropout('dropout_feat', output, rate=0.5) 
        output = FullyConnected('linear', output, GROUP)
        loss_xentropy = tf.losses.sigmoid_cross_entropy(label, output)
        loss_xentropy = tf.reduce_mean(loss_xentropy, name='loss_xentropy')

        logit = tf.sigmoid(output, name='logit')

        
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
                raise Exception("Unknown loss_type")
            
            dice = (2. * inse + smooth) / (l + r + smooth) # 1 minus
            ##
            dice = 1 - tf.reduce_mean(dice, name='dice_coe')
            return dice

        loss_dice = tf.identity(dice_coe(logit, label, axis=(1), loss_type='jaccard'), name='loss_dice') 
        
        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          500000, 0.2, True)
        wd_loss = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_loss')
        # wd_loss = tf.add_n(M.losses, name='regularize_loss') 
        add_moving_summary(loss_xentropy)
        add_moving_summary(loss_dice)
        add_moving_summary(wd_loss)

        # Visualization
        add_param_summary(('.*/W', ['histogram']))   # monitor W
        visualize_tensors('image', [image], scale_func=lambda x: x * 255, max_outputs=max(64, BATCH))
        cost = tf.add_n([3*loss_xentropy, loss_dice, wd_loss], name='cost')
        add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

def eval_classification(model, sessinit, folder=None, src_csv=None, dst_csv=None, has_groundtruth=False, resize=SHAPE, channel=1):
    """
    Eval a classification model on the dataset. It assumes the model inputs are
    named "input" and "label", and contains "wrong-top1" and "wrong-top5" in the graph.
    """
    df = pd.read_csv(src_csv)
    print(df.info())
    print(df)
    # Result will be store in da
    da = pd.DataFrame(columns=['Study', 'Cardiomegaly','Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    

    indices = list(range(len(df))) # Get the total image for deployment
    imread_mode = cv2.IMREAD_GRAYSCALE if channel == 1 else cv2.IMREAD_COLOR
    if resize is not None:
        resize = shape2d(resize)


    # if has_groundtruth:
    #     # pass # TODO
    #     pred_func = OfflinePredictor(PredictConfig(
    #         model=model,
    #         session_init=sessinit,
    #         input_names=['image'],
    #         output_names=['logit']
    #     ))
    # else:      
    pred_func = OfflinePredictor(PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['image'],
        output_names=['logit']
    ))
    
    

    for idx in indices:
        path = df.iloc[idx]['Path']
        if folder is not None:
            f = os.path.join(os.path.dirname(folder), path) # Get parent directory
        else:
            f = os.path.join(os.path.dirname(src_csv), path) # Get parent directory
        image = cv2.imread(f, imread_mode)
        print(f)
        assert image is not None, f
        if resize is not None:
            image = cv2.resize(image, tuple(resize[::-1]))
        if channel == 1:
            image = image[:, :, np.newaxis]
        image = image[np.newaxis,:,:,:]
        logit = np.array(pred_func(image) )
        logit = np.squeeze(logit).astype(np.float32)
        print(logit)
        da = da.append({'Study'       : os.path.dirname(path),
                   'Cardiomegaly'       : logit[0],
                   'Edema'              : logit[1],
                   'Consolidation'      : logit[2],
                   'Atelectasis'        : logit[3],
                   'Pleural Effusion'   : logit[4],
                   },
                  ignore_index=True, sort=False)  
    print(da)
    da = da.drop_duplicates(['Study'], keep='first') # Remove latteral 
    print(da)      
    da.to_csv(dst_csv, index=False)

    if has_groundtruth:
        pass

class VisualizeRunner(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            input_names=['image'],
            output_names=['logit'])

    def _before_train(self):
        self.ds_valid = Chexpert(folder=args.data, #'/u01/data/CheXpert-v1.0-small'
            train_or_valid='valid',
            fname='valid.csv',
            group=GROUP,
            resize=int(SHAPE),
            debug=DEBUG
            )
        self.ds_valid.reset_state() 
        self.ag_valid = [
            imgaug.ResizeShortestEdge(SHAPE),
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB), 
            imgaug.Albumentations(AB.CLAHE(p=1)),
            imgaug.CenterCrop((SHAPE, SHAPE)),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32()
        ]

        self.ds_valid.reset_state()
        self.ds_valid = AugmentImageComponent(self.ds_valid, self.ag_valid, 0) 
        self.ds_valid = BatchData(self.ds_valid, BATCH)
        # self.ds_valid = MultiProcessRunnerZMQ(self.ds_valid, num_proc=1)

    def _trigger(self):
        self.images = []
        self.labels = []
        self.output = []
        for dp in self.ds_valid.get_data():
            self.trainer.monitors.put_image('valid_image', dp[0])
            self.labels.append(dp[1])
            self.output.append(np.array(self.pred(dp[0])[0]))
        self.labels = np.array(self.labels)
        self.output = np.array(self.output)
        self.labels = np.reshape(self.labels, (-1, self.labels.shape[-1])) #Vectorize
        self.output = self.output.reshape(self.labels.shape)
        auc = []
        # print(self.labels.shape, self.output.shape)
        for idx in range(self.labels.shape[-1]):
            fpr, tpr, _ = sklearn.metrics.roc_curve(self.labels[:,idx], self.output[:,idx])
            auc.append(sklearn.metrics.auc(fpr, tpr))
        auc = np.array(auc)
        # print(auc.shape, auc[0], auc[1], auc[2], auc[3], auc[4], np.mean(auc))
        # self.trainer.monitors.put_summary((auc.shape, auc[0], auc[1], auc[2], auc[3], auc[4], np.mean(auc)))
        self.trainer.monitors.put_scalar('auc[0]', auc[0])
        self.trainer.monitors.put_scalar('auc[1]', auc[1])
        self.trainer.monitors.put_scalar('auc[2]', auc[2])
        self.trainer.monitors.put_scalar('auc[3]', auc[3])
        self.trainer.monitors.put_scalar('auc[4]', auc[4])
        self.trainer.monitors.put_scalar('auc_m ', np.mean(auc))
        return None
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--src_csv', help='src csv')
    parser.add_argument('--dst_csv', help='dst csv')
    parser.add_argument('--data', help='Image directory')
    parser.add_argument('--mode', help='small | patch | full', default='small')
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


    model = Model(name=args.model, mode=args.mode)

    BATCH = args.batch
    SHAPE = args.shape
    GROUP = args.group
    DEBUG = args.debug
    TRAIN = args.train

    if args.eval:
        assert args.src_csv
        batch = 128    # something that can run on one gpu
        eval_classification(model=model, sessinit=SmartInit(args.load), folder=None, 
                            src_csv=args.src_csv, dst_csv=args.dst_csv, has_groundtruth=False)
    else:
        logger.set_logger_dir(os.path.join('train_log', args.model, 'data'+args.mode, 'shape'+str(SHAPE), 'group'+str(GROUP), ), 'd')


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
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]

        ds_train.reset_state()
        ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        ds_train = BatchData(ds_train, BATCH)
        ds_train = MultiProcessRunnerZMQ(ds_train, num_proc=8)
        ds_train = PrintData(ds_train)

        config = TrainConfig(
            model=model,
            dataflow=ds_train,
            callbacks=[
                # KerasPhaseCallback(True),   # for Keras training
                PeriodicTrigger(ModelSaver(), every_k_epochs=1),
                PeriodicTrigger(MinSaver('cost'), every_k_epochs=2),
                VisualizeRunner(),
                ScheduledHyperParamSetter('learning_rate',
                                          [(0, 1e-2), (50, 1e-3), (100, 1e-4), (150, 1e-5)], interp='linear'),
            ],
            max_epoch=200,
            session_init=SmartInit(args.load),
        )

        trainer = SyncMultiGPUTrainerParameterServer(max(get_num_gpu(), 1))

        launch_train_with_config(config, trainer) 
        # if get_num_gpu() <= 1:
        #     # single GPU:
        #     launch_train_with_config(config, SimpleTrainer())
        # else:
        #     # multi GPU:
        #     launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(2))
        
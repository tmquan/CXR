import argparse
import glob2
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from natsort import natsorted

from tensorpack import *
from tensorpack.utils.argtools import shape2d
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.varreplace import freeze_variables
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack.tfutils.symbolic_functions as symbf

import tensorflow as tf

class Chexpert(RNGDataFlow):
    # https://github.com/tensorpack/tensorpack/blob/master/tensorpack/dataflow/image.py
    """ Produce images read from a list of files as (h, w, c) arrays. """
    def __init__(self, folder, group=14, train_or_valid='train', channel=1, resize=None, debug=False, shuffle=False):
        """
        
        """
        self.version = "1.0.0"
        self.description = "CheXpert is a large dataset of chest X-rays and competition for automated chest \nx-ray interpretation, which features uncertainty labels and radiologist-labeled \nreference standard evaluation sets. It consists of 224,316 chest radiographs \nof 65,240 patients, where the chest radiographic examinations and the associated \nradiology reports were retrospectively collected from Stanford Hospital. Each \nreport was labeled for the presence of 14 observations as positive, negative, \nor uncertain. We decided on the 14 observations based on the prevalence in the \nreports and clinical relevance.\n",
        self.citation = "@article{DBLP:journals/corr/abs-1901-07031,\n  author    = {Jeremy Irvin and Pranav Rajpurkar and Michael Ko and Yifan Yu and Silviana Ciurea{-}Ilcus and Chris Chute and Henrik Marklund and Behzad Haghgoo and Robyn L. Ball and Katie Shpanskaya and Jayne Seekins and David A. Mong and Safwan S. Halabi and Jesse K. Sandberg and Ricky Jones and David B. Larson and Curtis P. Langlotz and Bhavik N. Patel and Matthew P. Lungren and Andrew Y. Ng},\n  title     = {CheXpert: {A} Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison},\n  journal   = {CoRR},\n  volume    = {abs/1901.07031},\n  year      = {2019},\n  url       = {http://arxiv.org/abs/1901.07031},\n  archivePrefix = {arXiv},\n  eprint    = {1901.07031},\n  timestamp = {Fri, 01 Feb 2019 13:39:59 +0100},\n  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-07031},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}\n"
        self.folder = folder
        self.group = group
        self.is_train = True if train_or_valid=='train' else False
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.debug = debug
        self.shuffle = shuffle
        self.small = True if "small" in self.folder else False
        self.csvfile = os.path.join(self.folder, "train.csv") if self.is_train else os.path.join(self.folder, "valid.csv")
        print(self.folder)
        # Read the csv
        self.df = pd.read_csv(self.csvfile)
        print(self.df.info())

        # Process the label file
        # Convert the data to numerical type
        # Sex
        self.df['Sex'].mask(self.df['Sex'] == 'Female', 0.0, inplace=True)
        self.df['Sex'].mask(self.df['Sex'] == 'Male', 1.0, inplace=True)
        self.df['Sex'][(self.df['Sex'] != 1.0) & (self.df['Sex'] != 0.0)] = 0.5
        self.df['Sex'] = self.df['Sex'].astype(np.float64)

        # Age
        self.df['Age'] = self.df['Age'].astype(np.float64) / 100

        # Frontal/Lateral
        self.df['Frontal/Lateral'].mask(self.df['Frontal/Lateral'] == 'Frontal', 0.0, inplace=True)
        self.df['Frontal/Lateral'].mask(self.df['Frontal/Lateral'] == 'Lateral', 1.0, inplace=True)
        self.df['Frontal/Lateral'][(self.df['Frontal/Lateral'] != 1.0) & (self.df['Frontal/Lateral'] != 0.0)] = 0.5
        self.df['Frontal/Lateral'] = self.df['Frontal/Lateral'].astype(np.float64)

        # AP/PA
        self.df['AP/PA'].mask(self.df['AP/PA'] == 'AP', 0.0, inplace=True)
        self.df['AP/PA'].mask(self.df['AP/PA'] == 'PA', 1.0, inplace=True)
        self.df['AP/PA'][(self.df['AP/PA'] != 1.0) & (self.df['AP/PA'] != 0.0)] = 0.5
        self.df['AP/PA'] = self.df['AP/PA'].astype(np.float64)

        # In memory dataflow
        self.is_first = True
        self.dataflow = []
        # Process the dat
        # self.df = self.df.fillna(2.0, inplace=True) # blank, NaN
        
        # self.df['No Finding'][(self.df['No Finding'] == -1.0)] = 0.5
        # self.df['Enlarged Cardiomediastinum'][(self.df['Enlarged Cardiomediastinum'] == -1.0)] = 0.5
        # self.df['Cardiomegaly'][(self.df['Cardiomegaly'] == -1.0)] = 0.5
        # self.df['Lung Opacity'][(self.df['Lung Opacity'] == -1.0)] = 0.5
        # self.df['Lung Lesion'][(self.df['Lung Lesion'] == -1.0)] = 0.5
        # self.df['Edema'][(self.df['Edema'] == -1.0)] = 0.5
        # self.df['Consolidation'][(self.df['Consolidation'] == -1.0)] = 0.5
        # self.df['Pneumonia'][(self.df['Pneumonia'] == -1.0)] = 0.5
        # self.df['Atelectasis'][(self.df['Atelectasis'] == -1.0)] = 0.5
        # self.df['Pneumothorax'][(self.df['Pneumothorax'] == -1.0)] = 0.5
        # self.df['Pleural Effusion'][(self.df['Pleural Effusion'] == -1.0)] = 0.5
        # self.df['Pleural Other'][(self.df['Pleural Other'] == -1.0)] = 0.5
        # self.df['Fracture'][(self.df['Fracture'] == -1.0)] = 0.5
        # self.df['Support Devices'][(self.df['Support Devices'] == -1.0)] = 0.5
         
        # # -1.0 to 0.0
        # self.df['No Finding'][(self.df['No Finding'] == -1.0)] = 0.0
        # self.df['Enlarged Cardiomediastinum'][(self.df['Enlarged Cardiomediastinum'] == -1.0)] = 0.0
        # self.df['Cardiomegaly'][(self.df['Cardiomegaly'] == -1.0)] = 0.0
        # self.df['Consolidation'][(self.df['Consolidation'] == -1.0)] = 0.0
        # self.df['Pneumonia'][(self.df['Pneumonia'] == -1.0)] = 0.0
        # self.df['Pneumothorax'][(self.df['Pneumothorax'] == -1.0)] = 0.0
        # self.df['Pleural Other'][(self.df['Pleural Other'] == -1.0)] = 0.0

        # # -1.0 to 1.0
        # self.df['No Finding'][(self.df['No Finding'] == -1.0)] = 1.0
        # self.df['Enlarged Cardiomediastinum'][(self.df['Enlarged Cardiomediastinum'] == -1.0)] = 1.0
        # self.df['Cardiomegaly'][(self.df['Cardiomegaly'] == -1.0)] = 1.0
        # self.df['Consolidation'][(self.df['Consolidation'] == -1.0)] = 1.0
        # self.df['Pneumonia'][(self.df['Pneumonia'] == -1.0)] = 1.0
        # self.df['Pneumothorax'][(self.df['Pneumothorax'] == -1.0)] = 1.0
        # self.df['Pleural Other'][(self.df['Pleural Other'] == -1.0)] = 1.0
        
    def reset_state(self):
        self.rng = get_rng(self)   

    def __len__(self):
        if self.debug:
            return 250
        else:
            return len(self.df)

    def __iter__(self):
        indices = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(indices)
        if self.is_first:
            for idx in indices:
            
                f = os.path.join(os.path.dirname(self.folder), self.df.iloc[idx]['Path']) # Get parent directory
                image = cv2.imread(f, self.imread_mode)
                assert image is not None, f
                # print(f)
                if self.channel == 3:
                    image = image[:, :, ::-1]
                if self.resize is not None:
                    image = cv2.resize(image, tuple(self.resize[::-1]))
                if self.channel == 1:
                    image = image[:, :, np.newaxis]

                # Construct the prior
                prior = []
                prior.append(self.df.iloc[idx]['Sex'])
                prior.append(self.df.iloc[idx]['Age'])
                prior.append(self.df.iloc[idx]['Frontal/Lateral'])
                prior.append(self.df.iloc[idx]['AP/PA'])
                prior = np.array(prior, dtype = np.float32)

                # Construct the label
                # No Finding 
                # Enlarged Cardiomediastinum  
                # Cardiomegaly    
                # Lung Opacity    
                # Lung Lesion Edema   
                # Consolidation   
                # Pneumonia   
                # Atelectasis 
                # Pneumothorax    
                # Pleural Effusion    
                # Pleural Other   
                # Fracture    
                # Support Devices

                # 2nd place
                # self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                #          {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
                # if index == 5 or index == 8:
                #         labels.append(self.dict[1].get(value))
                #         if self.dict[1].get(
                #                 value) == '1' and \
                #                 self.cfg.enhance_index.count(index) > 0:
                #             flg_enhance = True
                # elif index == 2 or index == 6 or index == 10:
                #     labels.append(self.dict[0].get(value))
                #     if self.dict[0].get(
                #             value) == '1' and \
                #             self.cfg.enhance_index.count(index) > 0:
                #         flg_enhance = True

                # HV
                # label_names_0 = ['Cardiomegaly', 'Consolidation',
                #              'No Finding', 'Enlarged Cardiomediastinum', 'Pneumonia', 'Pneumothorax', 'Pleural Other']
                # df[label_names_0] = 1 * (df[label_names_0] > 0)  # convert uncertain -1 to negative 0
                # label = list(np.abs(row[5:].values))  # convert uncertain -1 to positive 1

                label = []
                if self.group==14:
                    # label = np.zeros(14)
                    # label.append(self.df.iloc[idx]['No Finding'])
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['No Finding'] == -1.0 else self.df.iloc[idx]['No Finding'])
                    label.append( 0.0 if self.df.iloc[idx]['No Finding'] == -1.0 else self.df.iloc[idx]['No Finding'])
                    # label.append(self.df.iloc[idx]['Enlarged Cardiomediastinum'])
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Enlarged Cardiomediastinum'] == -1.0 else self.df.iloc[idx]['Enlarged Cardiomediastinum'])
                    label.append( 0.0 if self.df.iloc[idx]['Enlarged Cardiomediastinum'] == -1.0 else self.df.iloc[idx]['Enlarged Cardiomediastinum'])
                    # label.append(self.df.iloc[idx]['Cardiomegaly']) #2: -1 to 0
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Cardiomegaly'] == -1.0 else self.df.iloc[idx]['Cardiomegaly'])
                    label.append( 0.0 if self.df.iloc[idx]['Cardiomegaly'] == -1.0 else self.df.iloc[idx]['Cardiomegaly'])
                    # label.append(self.df.iloc[idx]['Lung Opacity'])
                    # label.append( (np.random.normal(loc=0.8) > 0.5)*1.0 if self.df.iloc[idx]['Lung Opacity'] == -1.0 else self.df.iloc[idx]['Lung Opacity'])
                    label.append( 1.0 if self.df.iloc[idx]['Lung Opacity'] == -1.0 else self.df.iloc[idx]['Lung Opacity'])
                    # label.append(self.df.iloc[idx]['Lung Lesion'])
                    # label.append( (np.random.normal(loc=0.8) > 0.5)*1.0 if self.df.iloc[idx]['Lung Lesion'] == -1.0 else self.df.iloc[idx]['Lung Lesion'])
                    label.append( 1.0 if self.df.iloc[idx]['Lung Lesion'] == -1.0 else self.df.iloc[idx]['Lung Lesion'])
                    # label.append(self.df.iloc[idx]['Edema']) # 5: -1 to 1
                    # label.append( (np.random.normal(loc=0.8) > 0.5)*1.0 if self.df.iloc[idx]['Edema'] == -1.0 else self.df.iloc[idx]['Edema'])
                    label.append( 1.0 if self.df.iloc[idx]['Edema'] == -1.0 else self.df.iloc[idx]['Edema'])
                    # label.append(self.df.iloc[idx]['Consolidation']) #6 : -1 to 0
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Consolidation'] == -1.0 else self.df.iloc[idx]['Consolidation'])
                    label.append( 0.0 if self.df.iloc[idx]['Consolidation'] == -1.0 else self.df.iloc[idx]['Consolidation'])
                    # label.append(self.df.iloc[idx]['Pneumonia'])
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Pneumonia'] == -1.0 else self.df.iloc[idx]['Pneumonia'])
                    label.append( 0.0 if self.df.iloc[idx]['Pneumonia'] == -1.0 else self.df.iloc[idx]['Pneumonia'])
                    # label.append(self.df.iloc[idx]['Atelectasis']) # 8: -1 to 1
                    # label.append( (np.random.normal(loc=0.8) > 0.5)*1.0 if self.df.iloc[idx]['Atelectasis'] == -1.0 else self.df.iloc[idx]['Atelectasis'])
                    label.append( 1.0 if self.df.iloc[idx]['Atelectasis'] == -1.0 else self.df.iloc[idx]['Atelectasis'])
                    # label.append(self.df.iloc[idx]['Pneumothorax'])
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Pneumothorax'] == -1.0 else self.df.iloc[idx]['Pneumothorax'])
                    label.append( 0.0 if self.df.iloc[idx]['Pneumothorax'] == -1.0 else self.df.iloc[idx]['Pneumothorax'])
                    # label.append(self.df.iloc[idx]['Pleural Effusion']) # 10: -1 to 0
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Pleural Effusion'] == -1.0 else self.df.iloc[idx]['Pleural Effusion'])
                    label.append( 0.0 if self.df.iloc[idx]['Pleural Effusion'] == -1.0 else self.df.iloc[idx]['Pleural Effusion'])
                    # label.append(self.df.iloc[idx]['Pleural Other'])
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Pleural Other'] == -1.0 else self.df.iloc[idx]['Pleural Other'])
                    label.append( 0.0 if self.df.iloc[idx]['Pleural Other'] == -1.0 else self.df.iloc[idx]['Pleural Other'])
                    # label.append(self.df.iloc[idx]['Fracture'])
                    # label.append( (np.random.normal(loc=0.8) > 0.5)*1.0 if self.df.iloc[idx]['Fracture'] == -1.0 else self.df.iloc[idx]['Fracture'])
                    label.append( 1.0 if self.df.iloc[idx]['Fracture'] == -1.0 else self.df.iloc[idx]['Fracture'])
                    # label.append(self.df.iloc[idx]['Support Devices'])
                    # label.append( (np.random.normal(loc=0.8) > 0.5)*1.0 if self.df.iloc[idx]['Support Devices'] == -1.0 else self.df.iloc[idx]['Support Devices'])
                    label.append( 1.0 if self.df.iloc[idx]['Support Devices'] == -1.0 else self.df.iloc[idx]['Support Devices'])
                
                elif self.group==5:
                    # label.append(self.df.iloc[idx]['Cardiomegaly']) #2: -1 to 0
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Cardiomegaly'] == -1.0 else self.df.iloc[idx]['Cardiomegaly'])
                    label.append( 0.0 if self.df.iloc[idx]['Cardiomegaly'] == -1.0 else self.df.iloc[idx]['Cardiomegaly'])
                    # label.append(self.df.iloc[idx]['Edema']) # 5: -1 to 1
                    # label.append( (np.random.normal(loc=0.8) > 0.5)*1.0 if self.df.iloc[idx]['Edema'] == -1.0 else self.df.iloc[idx]['Edema'])
                    label.append( 1.0 if self.df.iloc[idx]['Edema'] == -1.0 else self.df.iloc[idx]['Edema'])
                    # label.append(self.df.iloc[idx]['Consolidation']) #6 : -1 to 0
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Consolidation'] == -1.0 else self.df.iloc[idx]['Consolidation'])
                    label.append( 0.0 if self.df.iloc[idx]['Consolidation'] == -1.0 else self.df.iloc[idx]['Consolidation'])
                    # label.append(self.df.iloc[idx]['Atelectasis']) # 8: -1 to 1
                    # label.append( (np.random.normal(loc=0.8) > 0.5)*1.0 if self.df.iloc[idx]['Atelectasis'] == -1.0 else self.df.iloc[idx]['Atelectasis'])
                    label.append( 1.0 if self.df.iloc[idx]['Atelectasis'] == -1.0 else self.df.iloc[idx]['Atelectasis'])
                    # label.append(self.df.iloc[idx]['Pleural Effusion']) # 10: -1 to 0
                    # label.append( (np.random.normal(loc=0.2) > 0.5)*1.0 if self.df.iloc[idx]['Pleural Effusion'] == -1.0 else self.df.iloc[idx]['Pleural Effusion'])
                    label.append( 0.0 if self.df.iloc[idx]['Pleural Effusion'] == -1.0 else self.df.iloc[idx]['Pleural Effusion'])
                
                label = np.nan_to_num(label, copy=True, nan=0)
                label = np.array(label, dtype = np.uint8)
                # label = np.array(label, dtype = np.float32)
                # if np.isnan(label).any():
                #     print(label)
                group = label.copy()
                # group[label==2.0] = 1.0*(np.random.normal(loc=0.2, size=len(label==2.0)) > 0.5)
                # label = np.nan_to_num(label)
                # group = np.nan_to_num(group, copy=True, nan=0)
                # group = np.array(group, dtype = np.uint8)

                # Get filename and compare to the list
                fname = np.zeros((320, 320, 3), dtype=np.uint8)
                cv2.putText(
                    fname,
                    str(f.split('/')[-3:])+str(group),
                    (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, #0.15 * image_scale,
                    (255, 255, 255), 
                    1, cv2.LINE_AA)
                fname = cv2.cvtColor(fname, cv2.COLOR_RGB2GRAY)
                if self.channel == 1:
                    fname = fname[:, :, np.newaxis]

                # 
                # image = image.astype(np.float32)
                self.dataflow.append([image, group, fname])
                yield [image, group, fname]
            self.is_first=False

        else:
            for idx in indices:
                yield self.dataflow[idx].copy()




if __name__ == '__main__':
    ds = Chexpert(folder='/u01/data/CheXpert-v1.0-small', 
        train_or_valid='train',
        resize=1024*3
        )
    ds.reset_state()
    ds = PrintData(ds)
    ds = MultiProcessRunnerZMQ(ds, num_proc=8)
    ds = BatchData(ds, 32)
    TestDataSpeed(ds).start()
    
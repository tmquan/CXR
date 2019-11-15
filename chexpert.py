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
    def __init__(self, folder, train_or_valid='train', channel=1, resize=None, shuffle=False):
        """
        
        """
        self.version = "1.0.0"
        self.description = "CheXpert is a large dataset of chest X-rays and competition for automated chest \nx-ray interpretation, which features uncertainty labels and radiologist-labeled \nreference standard evaluation sets. It consists of 224,316 chest radiographs \nof 65,240 patients, where the chest radiographic examinations and the associated \nradiology reports were retrospectively collected from Stanford Hospital. Each \nreport was labeled for the presence of 14 observations as positive, negative, \nor uncertain. We decided on the 14 observations based on the prevalence in the \nreports and clinical relevance.\n",
        self.citation = "@article{DBLP:journals/corr/abs-1901-07031,\n  author    = {Jeremy Irvin and Pranav Rajpurkar and Michael Ko and Yifan Yu and Silviana Ciurea{-}Ilcus and Chris Chute and Henrik Marklund and Behzad Haghgoo and Robyn L. Ball and Katie Shpanskaya and Jayne Seekins and David A. Mong and Safwan S. Halabi and Jesse K. Sandberg and Ricky Jones and David B. Larson and Curtis P. Langlotz and Bhavik N. Patel and Matthew P. Lungren and Andrew Y. Ng},\n  title     = {CheXpert: {A} Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison},\n  journal   = {CoRR},\n  volume    = {abs/1901.07031},\n  year      = {2019},\n  url       = {http://arxiv.org/abs/1901.07031},\n  archivePrefix = {arXiv},\n  eprint    = {1901.07031},\n  timestamp = {Fri, 01 Feb 2019 13:39:59 +0100},\n  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-07031},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}\n"
        self.folder = folder
        self.is_train = True if train_or_valid=='train' else False
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
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

        # Process the dat
        self.df.fillna(0.0) # blank, NaN
        
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
         
        # -1.0 to 0.0
        self.df['No Finding'][(self.df['No Finding'] == -1.0)] = 0.0
        self.df['Enlarged Cardiomediastinum'][(self.df['Enlarged Cardiomediastinum'] == -1.0)] = 0.0
        self.df['Cardiomegaly'][(self.df['Cardiomegaly'] == -1.0)] = 0.0
        self.df['Consolidation'][(self.df['Consolidation'] == -1.0)] = 0.0
        self.df['Pneumonia'][(self.df['Pneumonia'] == -1.0)] = 0.0
        self.df['Pneumothorax'][(self.df['Pneumothorax'] == -1.0)] = 0.0
        self.df['Pleural Other'][(self.df['Pleural Other'] == -1.0)] = 0.0

        # -1.0 to 1.0
        self.df['No Finding'][(self.df['No Finding'] == -1.0)] = 1.0
        self.df['Enlarged Cardiomediastinum'][(self.df['Enlarged Cardiomediastinum'] == -1.0)] = 1.0
        self.df['Cardiomegaly'][(self.df['Cardiomegaly'] == -1.0)] = 1.0
        self.df['Consolidation'][(self.df['Consolidation'] == -1.0)] = 1.0
        self.df['Pneumonia'][(self.df['Pneumonia'] == -1.0)] = 1.0
        self.df['Pneumothorax'][(self.df['Pneumothorax'] == -1.0)] = 1.0
        self.df['Pleural Other'][(self.df['Pleural Other'] == -1.0)] = 1.0
        
    def reset_state(self):
        self.rng = get_rng(self)   

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        indices = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(indices)
        for idx in indices:
            f = os.path.join(os.path.dirname(self.folder), self.df.iloc[idx]['Path']) # Get parent directory
            # print(f)
            image = cv2.imread(f, self.imread_mode)
            assert image is not None, f
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
            label = []
            label.append(self.df.iloc[idx]['No Finding'])
            label.append(self.df.iloc[idx]['Enlarged Cardiomediastinum'])
            label.append(self.df.iloc[idx]['Cardiomegaly'])
            label.append(self.df.iloc[idx]['Lung Opacity'])
            label.append(self.df.iloc[idx]['Lung Lesion'])
            label.append(self.df.iloc[idx]['Edema'])
            label.append(self.df.iloc[idx]['Consolidation'])
            label.append(self.df.iloc[idx]['Pneumonia'])
            label.append(self.df.iloc[idx]['Atelectasis'])
            label.append(self.df.iloc[idx]['Pneumothorax'])
            label.append(self.df.iloc[idx]['Pleural Effusion'])
            label.append(self.df.iloc[idx]['Pleural Other'])
            label.append(self.df.iloc[idx]['Fracture'])
            label.append(self.df.iloc[idx]['Support Devices'])
            label = np.array(label, dtype = np.float32)
            label = np.nan_to_num(label)
            
            # label[label==-1.0] = np.random.normal(len(label==-1.0))     # uncertainty
            # label[label==-2.0] = np.random.uniform(len(label==-2.0))    # unmentioned
            yield [image,label]



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
    
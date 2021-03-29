"""
    author:
        name: Do Viet Chinh
        email: dovietchinh1998@gmail.com
        facebook: https://www.facebook.com/profile.php?id=100005935236259
    
    data:
        27.3.2021
"""

import tensorflow as tf
from augmenter import RandomAugment
import cv2
import pandas as pd
import sys
import os
from augmenter import RandomAugment 
path_import = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_import +'/../' )
from configs import config

# load augmenter params from config file

N = config.N
M = config.M
SHUFFLE_DATA = config.SHUFFLE_DATA

augmenter = RandomAugment(N,M)

class DataSequence(tf.keras.utils.Sequence):
    """DataSequence : ( TENSORFLOW )
        class define will wraper folowing steps:
            - reading images
            - normalizing images
            - fetching a minibatch of data and collating them into batched samples
            - shuffle data if configs.SHUFFLE set to True
    Args:
        __init__  
        __len__          return total steps per epoch
        __getitem__      return a batched data (x,y)
        on_epoch_end     this function will excute when reaching end of epochs
    """
    def __init__(self,data_folder,data_frame, batch_size=32, phase = 'val'):
        try:
            assert phase in ['train','val','test'], "Invalid keyword, phase must be in 'train','val' or 'test'"
        except Exception as msg:
            print(msg)

        self.data_folder = data_folder 
        self.batch_size = batch_size 
        self.phase = phase 
        if phase =='train':
            df_0 = data_frame[data_frame.mask_mouth==0]
            df_1 = data_frame[data_frame.mask_mouth==1]
            over_sampling = len(df_0)//len(df_1)-1
            self.data_frame = pd.concat([df_0]+[df_1]*over_sampling)
        else:
            self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)//self.batch_size

    def __getitem__(self,index):
        current_batch = self.data_frame[index*self.batch_size : (index+1)*self.batch_size]

        data = []
        labels = []

        for ob in current_batch.iloc:
            path = ob.path
            mask_mouth = ob.mask_mouth
            img = cv2.imread(os.path.join(self.data_folder,path), cv2.IMREAD_COLOR)
            if img is None:
                print('Unable to read this img: ',path)
                continue
            if self.phase =='train':
                img = augmenter(img)
            data.append(img)
            labels.append(mask_mouth)
        
        data = tf.convert_to_tensor(data, tf.float32)
        labels = tf.convert_to_tensor(labels, tf.float32)
        data = data/255.
        return data,labels

    def on_epoch_end(self,):
        if SHUFFLE_DATA:
            self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        return

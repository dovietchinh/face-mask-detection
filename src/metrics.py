"""
    author:
        name: Do Viet Chinh
        email: dovietchinh1998@gmail.com
        facebook: https://www.facebook.com/profile.php?id=100005935236259
    
    data:
        27.3.2021
"""

import tensorflow as tf
import tensorflow.keras.callbacks
import sklearn
import sklearn.metrics
import numpy as np
from sklearn.metrics import classification_report

class macro_acc_recall_f1(tf.keras.callbacks.Callback):
    def get_validation_data(self,validation_data):
        self.validation_data = validation_data
    
    
    def on_epoch_end(self, epoch,logs=None):
        predict = []
        ground_truth = []
        predict_gender = []
        ground_truth_gender = []
        
        for i in range(len(self.validation_data)):
            out = self.model.predict(self.validation_data[i][0])[1]
            out = tf.concat(out,axis=-1)
            predict.append(out)
            
            out_gender = self.model.predict(self.validation_data[i][0])[0]
            out_gender = tf.concat(out_gender,axis=-1)
            predict_gender.append(out_gender)
            
            
            out1 = self.validation_data[i][1][1]
            out1 = tf.concat(out1,axis=-1)
            ground_truth.append(out1)
            
            out1_gender = self.validation_data[i][1][0]
            out1_gender = tf.concat(out1_gender,axis=-1)
            ground_truth_gender.append(out1_gender)
               
        predict = tf.concat(predict,axis=0)
        ground_truth = tf.concat(ground_truth,axis=0)  
        predict_gender = tf.concat(predict_gender,axis=0)
        ground_truth_gender = tf.concat(ground_truth_gender,axis=0)  
        
        
        predict = tf.argmax(predict,axis=1).numpy().reshape(-1)
        predict_gender = tf.argmax(predict_gender,axis=1).numpy().reshape(-1)
        
        ground_truth = tf.argmax(ground_truth,axis=1).numpy().reshape(-1)
        ground_truth_gender = tf.argmax(ground_truth_gender,axis=1).numpy().reshape(-1)
            
        print('-----------AGE----------')
        
        target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']
        print(classification_report(ground_truth, predict, target_names=target_names, digits=4))        
        
        print('-----------GENDER----------')
        
        target_names = ['female', 'male']
        print(classification_report(ground_truth_gender, predict_gender, target_names=target_names, digits=4))
        
        return
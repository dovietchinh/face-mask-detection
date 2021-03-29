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
    def get_validation_data(self,validation_data,model):
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch,logs=None):
        predict = []
        ground_truth = []

        
        for i in range(len(self.validation_data)):
            out = self.model(self.validation_data[i][0])
            predict.append(out)
            
            
          
            out1 = self.validation_data[i][1]
            ground_truth.append(out1)

               
        predict = tf.concat(predict,axis=0)
        ground_truth = tf.concat(ground_truth,axis=0)  
        predict = predict.numpy().reshape(-1)
        predict = (predict >0.5).astype('int8')
        ground_truth = ground_truth.numpy().reshape(-1)

            
        print('-----------FACE_MASK----------')
        
        target_names = ['class 0', 'class 1']
        print(classification_report(ground_truth, predict, target_names=target_names, digits=4))        

        return
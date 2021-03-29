"""
    author:
        name: Do Viet Chinh
        email: dovietchinh1998@gmail.com
        facebook: https://www.facebook.com/profile.php?id=100005935236259
    
    data:
        27.3.2021
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,GlobalAveragePooling2D
import sys 
import os
path_import = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_import +'/../')
from configs import config

# load model architecture params from config



def build_model(inputs_shape,conv2d_params,fc_params):
    inputs = tf.keras.Input(inputs_shape)
    for index, (channels, kernel_size) in enumerate(conv2d_params):
        if index == 0:
            x = Conv2D(channels,kernel_size, activation='relu', padding = 'same', name = 'Conv2D_'+str(index+1))(inputs)
            x = MaxPooling2D(2,padding='same')(x)
        if index !=0 and index != (len(conv2d_params)-1):
            x = Conv2D(channels,kernel_size, activation='relu', padding = 'same', name = 'Conv2D_'+str(index+1))(x)
            x = MaxPooling2D(2,padding='same')(x)
        if index == len(conv2d_params)-1:
            x = Conv2D(channels,kernel_size, activation='relu', padding = 'same', name = 'Conv2D_'+str(index+1))(x)
    
    x = GlobalAveragePooling2D()(x)
    
    for index, (fc_param, activation_function) in enumerate(fc_params):
        x = Dense(fc_param,activation = activation_function, name = 'Dense_'+str(index+1))(x)
        
    model = tf.keras.Model(inputs,x)

    return model



"""
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = build_model((INPUT_SHAPE,INPUT_SHAPE,3), CONV2D_PARAMS, FC_PARAMS)
    
    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        agc_gradients = agc.adaptive_clip_grad(trainable_params, gradients, 
                                               clip_factor=self.clip_factor, eps=self.eps)
        self.optimizer.apply_gradients(zip(gradients, trainable_params))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        predictions = self.mini_vgg(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}
"""


if __name__ =='__main__':
    INPUT_SHAPE = config.INPUT_SHAPE
    CONV2D_PARAMS = config.CONV2D_PARAMS
    FC_PARAMS = config.FC_PARAMS
    model = build_model((INPUT_SHAPE,INPUT_SHAPE,3), CONV2D_PARAMS, FC_PARAMS)
    model.summary()
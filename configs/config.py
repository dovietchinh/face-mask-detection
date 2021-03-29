"""
author:
    name: Do Viet Chinh
    email: dovietchinh1998@gmail.com
    facebook: https://www.facebook.com/profile.php?id=100005935236259
date:
    26.3.2021
"""
# Model Architecture

USE_GPU = True

INPUT_SHAPE = 112
CONV2D_PARAMS = [ (64,3), (128,3), (256,3), (512,3), (1024,3) ]          #every elemens is a paris (num_of_channel, kernel_size) in Conv2D layer
"""
MODEL_ARCHITECTURE = {
    'Conv2D_1':{
        'num_channel' : 64,
        'kernel_size' : 3,
        'activations' : None,
        'pooling': False,
    },
    'Conv2D_1_dup':{
        'num_channel' : 64,
        'kernel_size' : 3,
        'activations' : 'relu',
        'pooling': True,
    },
    'Conv2D_2':{
        'num_channel' : 128,
        'kernel_size' : 3,
        'activations' : 'relu',
        'pooling': True,
    },
    'Conv2D_3':{
        'num_channel' : 256,
        'kernel_size' : 3,
        'activations' : 'relu',
        'pooling': True,
    },
    'Conv2D_4':{
        'num_channel' : 512,
        'kernel_size' : 3,
        'activations' : 'relu',
        'pooling': True,
    },
    'Conv2D_5':{
        'num_channel' : 1024,
        'kernel_size' : 3,
        'activations' : 'relu',
        'pooling': True,
    },
}
"""
FC_PARAMS = [ (1024,'relu'),
             (1,'sigmoid'), 
]



# Training

EPOCHS = 50
INIT_EPOCH = 0                                 
BATCH_SIZE = 32
OPTIMIZER = 'Adam'                              # [str]
LEARNING_RATE = None                           # [float] initial learning rate
LOSS_FUNCTION = 'binary_crossentropy'          # [str]
METRICS = 'acc'                                # [str]
USE_MULTIPROCESS = False                        # [Boolean] 
SHUFFLE_DATA = False                           # [Boolean] shuffle data after training an epoch

CHECKPOINT_PATH = './checkpoint/'
LOG_PATH = './log/'

# Augment Param

N = 4
M = 10


# Preprocess data 

PADDING = True
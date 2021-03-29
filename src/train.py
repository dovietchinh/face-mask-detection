"""
    author:
        name: Do Viet Chinh
        email: dovietchinh1998@gmail.com
        facebook: https://www.facebook.com/profile.php?id=100005935236259
    
    data:
        27.3.2021
"""

import tensorflow as tf 
import pandas as pd
from metrics import macro_acc_recall_f1
from datasequence import DataSequence
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from model import build_model
from utils import split_data,lr_function
import sys 
import os 
path_import = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_import+'/../')
from configs import config
import pathlib
# load params from config

INPUT_SHAPE = config.INPUT_SHAPE
EPOCHS = config.EPOCHS
INIT_EPOCH = config.INIT_EPOCH    
BATCH_SIZE = config.BATCH_SIZE                             
OPTIMIZER = config.OPTIMIZER                              
LEARNING_RATE = config.LEARNING_RATE                    
LOSS_FUNCTION = config.LOSS_FUNCTION
METRICS = config.METRICS                                
USE_MULTIPROCESS = config.USE_MULTIPROCESS              
SHUFFLE_DATA = config.SHUFFLE_DATA                      
INPUT_SHAPE = config.INPUT_SHAPE
CONV2D_PARAMS = config.CONV2D_PARAMS
FC_PARAMS = config.FC_PARAMS


data_folder = '/u01/DATA/AGE_GENDER/RMFD_resize/'
file_csv = '/u01/DATA/AGE_GENDER/RMFD_resize/mask_mouth_data.csv'

df = pd.read_csv(file_csv)
df_train,df_val,df_test = split_data(df)

data_sequence_train = DataSequence(data_folder, df_train, batch_size=BATCH_SIZE, phase='train')
data_sequence_val = DataSequence(data_folder, df_val, batch_size=BATCH_SIZE, phase='val')
data_sequence_test = DataSequence(data_folder, df_test, batch_size=BATCH_SIZE, phase='test')

model = build_model((INPUT_SHAPE,INPUT_SHAPE,3), CONV2D_PARAMS, FC_PARAMS)

checkpoint_path = config.CHECKPOINT_PATH
log_path = config.LOG_PATH

pathlib.Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
model.compile( OPTIMIZER,
                loss=LOSS_FUNCTION,
                metrics = 'acc',
                )

mc = ModelCheckpoint(filepath=os.path.join(
    checkpoint_path, "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)
tb = TensorBoard(log_dir=log_path, write_graph=True)

lr_scheduler = LearningRateScheduler(lr_function)
model.fit(data_sequence_train,
                    epochs=EPOCHS,
                    initial_epoch=INIT_EPOCH,    
                    validation_data=data_sequence_val,
                    use_multiprocessing=config.USE_MULTIPROCESS,
                    callbacks=[mc,tb,lr_scheduler],
                        verbose=1 )


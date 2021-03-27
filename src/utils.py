"""
    author:
        name: Do Viet Chinh
        email: dovietchinh1998@gmail.com
        facebook: https://www.facebook.com/profile.php?id=100005935236259
    
    data:
        27.3.2021
"""

import pandas as pd
import tensorflow as tf

def split_data(df):
    df_0 = df[df.mask_mouth==0]
    df_1 = df[df.mask_mouth==1]

    len0 = len(df_0)
    len1 = len(df_1)

    df_0_train = df_0[ : int(0.7*len0) ]
    df_0_val = df_0[ int(0.7*len0): int(0.9*len0) ]
    df_0_test = df_0[ int(0.9*len0) : ]

    df_1_train = df_1[ : int(0.7*len1) ]
    df_1_val = df_1[ int(0.7*len1): int(0.9*len1) ]
    df_1_test = df_1[ int(0.9*len1) : ]

    df_train = pd.concat([df_0_train,df_1_train])
    df_val = pd.concat([df_0_val,df_1_val])
    df_test = pd.concat([df_0_test,df_1_test])
    
    return df_train,df_val,df_test

def lr_function(epoch,current_lr):
    if epoch <=3:
        return 5e-4
    if 3<= epoch <10:
        return 1e-3
    if epoch >=10:
        return current_lr *tf.math.exp(-0.1)

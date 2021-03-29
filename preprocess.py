import pandas as pd
import cv2
import numpy as np
import os
from configs import config
mask_folder = './data/RMFD/AFDB_face_dataset'
no_mask_folder = './data/RMFD/AFDB_masked_face_dataset'
mask_mouth = []
a = os.listdir(mask_folder)
b = os.listdir(no_mask_folder)
path = []
save_folder = './data/RMFD_resize/'
def padding_img(img):
	img_ = img.copy()
	height,width,_= img_.shape
	if height > width:
		padding_values = [[0,0],[(height-width)//2,(height-width)//2],[0,0]]
	else:
		padding_values = [[0,0],[(width-height)//2,(width-height)//2],[0,0]]
	img__ = np.pad(img_,padding_values,'edge')
	return img_
if __name__ =='__main__':
	count = 0
	for i in a:
		list_img = os.listdir(mask_folder + '/' + i)
		for j in list_img:
				path_ = mask_folder + '/'+ i+'/' + j
				img = cv2.imread(path_)
                if config.PADDING == TrueL
			    	img = padding_img(img)
                img = cv2.resize(img,(config.INPUT_SHAPE,config.INIT_SHAPE))
                directory = os.path.dirname(save_folder+path_)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(save_folder + path_,img)
                count +=1
                print(count,end='\r')
	for i in b:
		list_img = os.listdir(no_mask_folder + '/'+i)
		for j in list_img:
				path_ = mask_folder + '/'+ i+'/' + j
				img = cv2.imread(path_)
                if config.PADDING == True:
				    img = padding_img(img)
                img = cv2.resize(img,(config.INPUT_SHAPE,config.INIT_SHAPE))
                directory = os.path.dirname(save_folder+path_)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(save_folder + path_,img)
                count += 1
                print(count,end='\r')
				
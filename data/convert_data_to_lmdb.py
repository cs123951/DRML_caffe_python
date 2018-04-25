# -*- coding: utf-8 -*-
import os  
import glob  
import random  
import numpy as np  
  
import cv2  
  
import caffe  
from caffe.proto import caffe_pb2  
import lmdb  
  
#Size of images  
IMAGE_WIDTH = 170  
IMAGE_HEIGHT = 170  
  
# train_lmdb、validation_lmdb 路径  
train_lmdb = 'train_ck_lmdb'  
#validation_lmdb = 'val_data_lmdb'  
  
# 如果存在了这个文件夹, 先删除  
#os.system('rm -rf  ' + train_lmdb)  
#os.system('rm -rf  ' + validation_lmdb)  
  
# 读取图像  
f = open("E:/Projects/jupyterProject/Face/data/au_train.txt")
train_data = [img.strip('\n') for img in f.readlines()]
test_data = [img.strip('\n') for img in f.readlines()]
f.close()
  
# Shuffle train_data  
# 打乱数据的顺序  
#random.shuffle(train_data)  
  
# 图像的变换, 直方图均衡化, 以及裁剪到 IMAGE_WIDTH x IMAGE_HEIGHT 的大小  
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):  
    #Histogram Equalization  
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])  
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])  
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])  
  
    #Image Resizing, 三次插值  
#    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)  
    return img  
  
def make_datum(img, label):  
    #image is numpy.ndarray format. BGR instead of RGB  
    return caffe_pb2.Datum(  
        channels=3,  
        width=IMAGE_WIDTH,  
        height=IMAGE_HEIGHT,  
        label=label,  
        data=np.rollaxis(img, 2).tobytes())
    # or .tostring() if numpy < 1.9  

print('\nCreating train_lmdb')  
in_db = lmdb.open(train_lmdb,map_size=int(1e10))   
with in_db.begin(write=True) as in_txn: # 创建操作数据库句柄  
    for in_idx, img_path in enumerate(train_data):  
#        if in_idx %  6 == 0: # 只处理 5/6 的数据作为训练集  
#            continue         # 留下 1/6 的数据用作验证集  
        # 读取图像. 做直方图均衡化、裁剪操作  
        cv_img = cv2.imread(img_path.split(" ")[0], cv2.IMREAD_COLOR)  
        img = transform_img(cv_img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)  
        #this label is a fake label
        label = 1 
  
        datum = make_datum(img, label)  
        # '{:0>5d}'.format(in_idx):  
        #      lmdb的每一个数据都是由键值对构成的,  
        #      因此生成一个用递增顺序排列的定长唯一的key  
        #print('{:0>5d}'.format(in_idx) + ':' + img_path)
        in_txn.put('{:0>5d}'.format(in_idx).encode('ascii'), datum.SerializeToString()) #调用句柄，写入内存  
        
# 结束后记住释放资源，否则下次用的时候打不开。。。  
in_db.close()   
  
# 创建验证集 lmdb 格式文件  
#print('\nCreating validation_lmdb')  
#in_db = lmdb.open(validation_lmdb,map_size=int(1e10))  
#with in_db.begin(write=True) as in_txn:  
#    for in_idx, img_path in enumerate(train_data):  
#        if in_idx % 6 != 0:  
#            continue  
#        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
#        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)  
#        label = 0  
#        datum = make_datum(img, label)  
#        in_txn.put('{:0>5d}'.format(in_idx).encode('ascii'), datum.SerializeToString())  
##        print('{:0>5d}'.format(in_idx) + ':' + img_path) 
#in_db.close()  
print('\nFinished processing all images')

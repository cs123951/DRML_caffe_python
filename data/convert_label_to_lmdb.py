# -*- coding: utf-8 -*-
import sys   
import numpy as np  
import lmdb  
import caffe  
  
# 根据多标签的位置选择从数据库、文件等中读取每幅图片的多标签，将其构造成一维的np.array类型，并追加入all_labels列表  
all_labels = []  
f = open("E:/Projects/jupyterProject/Face/data/label_490_0_1_to_be_converted.txt")
train_labels = [labels.strip('\n') for labels in f.readlines()]
#test_labels = [labels.strip('\n') for labels in f.readlines()]
for tl in train_labels:
    label_str = tl.strip(' ').split(' ')
    label_int = []
    for ls in label_str:
        label_int.append(int(ls))
    all_labels.append(label_int)
f.close()

# 创建标签LMDB   
lmdb_path = "train_label_lmdb"  
env = lmdb.open(lmdb_path,map_size=int(1e10))  
with env.begin(write=True) as txn:  
    for in_idx, labels in enumerate(all_labels):
        if in_idx %  6 == 0: # 只处理 5/6 的数据作为训练集  
            continue         # 留下 1/6 的数据用作验证集  
        datum = caffe.proto.caffe_pb2.Datum()  
        datum.channels = len(labels)  
        datum.height = 1  
        datum.width =  1  
        datum.data = np.array(labels).tostring()          # or .tobytes() if numpy < 1.9   
        datum.label = 0  
        txn.put('{:0>5d}'.format(in_idx).encode('ascii'), datum.SerializeToString())  
env.close()

# 创建标签LMDB   
lmdb_path = "val_label_lmdb"  
env = lmdb.open(lmdb_path,map_size=int(1e10))  
with env.begin(write=True) as txn:  
    for in_idx, labels in enumerate(all_labels):
        if in_idx %  6 != 0: # 只处理 5/6 的数据作为训练集  
            continue         # 留下 1/6 的数据用作验证集  
        datum = caffe.proto.caffe_pb2.Datum()  
        datum.channels = np.array(labels).shape[0]  
        datum.height = 1  
        datum.width =  1  
        datum.data = np.array(labels).tostring()          # or .tobytes() if numpy < 1.9   
        datum.label = 0  
        txn.put('{:0>5d}'.format(in_idx).encode('ascii'), datum.SerializeToString())  
env.close()
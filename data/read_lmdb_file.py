# -*- coding: utf-8 -*-
import caffe  
from caffe.proto import caffe_pb2  
  
import lmdb  
import cv2  
import numpy as np  
  
lmdb_env = lmdb.open('train_lmdb', readonly=True) # 打开数据文件  
lmdb_txn = lmdb_env.begin() # 生成处理句柄  
lmdb_cursor = lmdb_txn.cursor() # 生成迭代器指针  
datum = caffe_pb2.Datum() # caffe 定义的数据类型  
  
for key, value in lmdb_cursor: # 循环获取数据  
    datum.ParseFromString(value) # 从 value 中读取 datum 数据  
  
    label = datum.label  
    data = caffe.io.datum_to_array(datum)  
    print(data.shape)
    print(datum.channels)
    image = data.transpose(1, 2, 0)  
    cv2.imshow('image_name', image)
#    cv2.imwrite('lena.png',img)
    cv2.waitKey(0)  
  
cv2.destroyAllWindows()  
lmdb_env.close()  

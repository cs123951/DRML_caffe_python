# -*- coding: utf-8 -*-
import caffe  
import numpy as np  
  
  
class MultilabelSigmoidCrossEntropyLossLayer(caffe.Layer):
    """ 
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer 
    to demonstrate the class interface for developing layers in Python. 
    """  
  
    def setup(self, bottom, top):  
        # check input pair  
        if len(bottom) != 2:  
            raise Exception("Need two inputs to compute distance.")
  
    def reshape(self, bottom, top):  
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)  
#        loss output is scalar  
        top[0].reshape(1)
  
    def forward(self, bottom, top):  
        ip2 = np.array(bottom[0].data)  
        label = np.array(bottom[1].data)
        sum_loss = 0
        self.y_pred = np.zeros([ip2.shape[0], ip2.shape[1]])
        for i in range(ip2.shape[0]):
            for j in range(0, ip2.shape[1]):
                self.y_pred[i,j] = 1/(1+np.exp(-ip2[i,j]))
                y_true = int(label[i,j]>0)*np.log(self.y_pred[i,j])
                y_false = int(label[i,j]<0)*np.log(1-self.y_pred[i,j])
                sum_loss += y_true + y_false
        top[0].data[...] = -sum_loss/ip2.shape[0]
  
    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            ip2 = np.array(bottom[0].data)
            label = np.array(bottom[1].data)
            count = ip2.shape[1] # count is the number of parameters
            num = ip2.shape[0] # num is the number of pictures
            for i in range(count):
                for j in range(num):
                    if label[j,i] > 0:
                        bottom[0].diff[j,i] = (ip2[j,i]-1)
                    if label[j,i] < 0:
                        bottom[0].diff[j,i] = -ip2[j,i]

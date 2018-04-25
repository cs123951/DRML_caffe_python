# -*- coding: utf-8 -*-
import caffe
class BoxLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.top_num = 64
        self.width = params['width']
        self.height = params['height']


        
    def reshape(self, bottom, top):
        for i in range(self.top_num):
            top[i].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1],self.height,self.width)
        
    def forward(self, bottom, top):
#        bottom shape: [batch_size,num_of_kernels,width,height]
        self.big_width = bottom[0].data.shape[1]
        self.big_height = bottom[0].data.shape[2]
        n_width = self.big_width//self.width
        n_height = self.big_height//self.height
        for itt in range(bottom[0].data.shape[0]):
            top_id = 0
            for ww in range(n_width):
                for hh in range(n_height):
                    top[top_id].data[itt,...] = bottom[0].data[itt,:,ww*self.width:ww*self.width+self.width,hh*self.height:hh*self.height+self.height]
                    top_id += 1
        

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            self.big_width = bottom[0].data.shape[1]
            self.big_height = bottom[0].data.shape[2]
            n_width = self.big_width//self.width
            n_height = self.big_height//self.height
            for itt in range(bottom[0].data.shape[0]):
                top_id = 0
                for ww in range(n_width):
                    for hh in range(n_height):
#                        top[top_id].data[itt,...] = bottom[0].data[itt,:,ww*self.width:ww*self.width+self.width,hh*self.height:hh*self.height+self.height]
                        bottom[0].diff[itt,:,ww*self.width:ww*self.width+self.width,hh*self.height:hh*self.height+self.height] = top[top_id].diff[itt,...]
                        top_id += 1
            


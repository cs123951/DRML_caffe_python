# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import caffe
class SpliceLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.big_width = params['width']
        self.big_height = params['height']

        
    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].shape[0],bottom[0].shape[1],self.big_width,self.big_height)
        
    def forward(self, bottom, top):
        self.width = bottom[0].data.shape[2]
        self.height = bottom[0].data.shape[3]
        n_width = self.big_width//self.width
        n_height = self.big_height//self.height
        for itt in range(bottom[0].data.shape[0]):
            top_id = 0
            for ww in range(n_width):
                for hh in range(n_height):
                    top[0].data[itt,:,ww*self.width:ww*self.width+self.width,hh*self.height:hh*self.height+self.height] = bottom[top_id].data[itt,:,:,:]
                    top_id += 1
        

    def backward(self, top, propagate_down, bottom):
        self.width = bottom[0].data.shape[2]
        self.height = bottom[0].data.shape[3]
        n_width = self.big_width//self.width
        n_height = self.big_height//self.height
        for itt in range(bottom[0].data.shape[0]):
            top_id = 0
            for ww in range(n_width):
                for hh in range(n_height):
                    bottom[top_id].diff[itt,:,:,:] = top[0].diff[itt,:,ww*self.width:ww*self.width+self.width,hh*self.height:hh*self.height+self.height]
                    top_id += 1
            



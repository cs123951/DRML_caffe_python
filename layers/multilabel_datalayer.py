# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image
import cv2

class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self):
        mean = np.load('data/mean_ck.npy')
        self.mean = np.array(mean, dtype=np.float32)
#        print("self.mean")
#        print(self.mean.shape)
        self.scale = 1.0/255

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe
        prototxt.
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
#       self.mean.shape  [3,170,170]
#       im.shape [170,170,3]
        
        im = im.transpose((2, 0, 1))
        im -= self.mean

        im *= self.scale

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)

class MultilabelDataLayer(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layers parameters.
        params = eval(self.param_str)
        

        # Check the paramameters for validity.
        check_params(params)
        self.num_labels = params['num_labels']

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layers
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])

        top[1].reshape(self.batch_size, self.num_labels)

        print_info("MultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
#            if itt == 63:
#                print(self.batch_loader._cur)
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_image()
            # Add directly to the caffe data layers
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layers to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.dataset_root = params['dataset_root']
        self.im_shape = params['im_shape']
        self.num_labels = params['num_labels']
        # get list of image indexes.
        if params['split'] == 'train':
            self.isshuffle = True
            self.isflip = True
        else:
            self.isshuffle = False
            self.isflip = False
        self.indexlist = [line.strip('\n') for line in open(self.dataset_root)]
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()
        
        if self.isshuffle:
            shuffle(self.indexlist)

        print("BatchLoader initialized with {} images".format(
            len(self.indexlist)))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        rowdata = self.indexlist[self._cur]  # Get the image index
        rowdata = rowdata.split(' ')
        image_file_name = rowdata[0]
        im = np.asarray(cv2.imread(image_file_name, cv2.IMREAD_COLOR))
        
#        im = np.asarray(Image.open(image_file_name))

#        im = scipy.misc.imresize(im, self.im_shape)  # resize
#        print("im.shape")
#        print(im.shape)

        # do a simple horizontal flip as data augmentation
#        if self.isflip:
#            flip = np.random.choice(2)*2-1
#            im = im[:, ::flip, :]

        # Load and prepare ground truth
        multilabel = np.zeros(self.num_labels).astype(np.float32)
        for j in range(self.num_labels):
            # in the multilabel problem we don't care how MANY instances
            # there are of each class. Only if they are present.
            # The "-1" is b/c we are not interested in the background
            # class.
            multilabel[j] = np.float(rowdata[j+1])

        self._cur += 1
        return self.transformer.preprocess(im), multilabel

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'dataset_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print("{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape']))

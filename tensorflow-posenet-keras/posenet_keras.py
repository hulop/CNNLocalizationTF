##############################################################################
#The MIT License (MIT)
#
#Copyright (c) 2018 IBM Corporation, Carnegie Mellon University and others
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
##############################################################################

from scipy.misc import imread, imresize

from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate
from keras.layers import ZeroPadding2D, Dropout, Flatten
from keras.layers import Reshape, Activation, BatchNormalization
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
import keras.backend as K
import tensorflow as tf
import numpy as np
import h5py
import math
import os
import sys

def create_posenet_inception_v1(weights_path=None, trainable=True):
    # not implemented
    posenet = None
    
    return posenet

def create_posenet_inception_v3(weights_path=None, trainable=True):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # add top layer
    model_output = base_model.output
    
    cls3_fc_pose_xyz = Dense(3,name='cls3_fc_pose_xyz')(model_output)
    
    cls3_fc_pose_wpqr = Dense(4,name='cls3_fc_pose_wpqr')(model_output)
    
    posenet = Model(inputs=base_model.input, outputs=[cls3_fc_pose_xyz, cls3_fc_pose_wpqr])
    
    if weights_path:
        weights_path_ext = os.path.splitext(weights_path)[-1]
        if weights_path_ext==".h5":
            posenet.load_weights(weights_path, by_name=True)
        else:
            print("invalid weight file : " + weights_path)
            sys.exit()
    
    if not trainable:
        for layer in posenet.layers:
            layer.trainable = False
    
    return posenet

def create_posenet_mobilenet_v1(weights_path=None, trainable=True):
    # create the base pre-trained model
    alpha = 1.0
    dropout=1e-3
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)
    base_model = MobileNet(input_shape=input_shape, alpha=alpha, dropout=dropout, weights='imagenet', include_top=False)
    
    # add top layer
    model_output = base_model.output
    
    if K.image_data_format() == 'channels_first':
        shape = (int(1024 * alpha), 1, 1)
    else:
        shape = (1, 1, int(1024 * alpha))
    
    model_output = GlobalAveragePooling2D()(model_output)
    model_output = Reshape(shape, name='reshape_1')(model_output)
    model_output = Dropout(dropout, name='dropout')(model_output)
    
    conv_pose_xyz = Conv2D(1024, (1, 1),
                           padding='same', name='conv_pose_xyz')(model_output)
    
    conv_pose_xyz_flat = Flatten()(conv_pose_xyz)
    
    cls_fc_pose_xyz = Dense(3,name='cls_fc_pose_xyz')(conv_pose_xyz_flat)
    
    conv_pose_wpqr = Conv2D(1024, (1, 1),
                            padding='same', name='conv_pose_wpqr')(model_output)
    
    conv_pose_wpqr_flat = Flatten()(conv_pose_wpqr)
    
    cls_fc_pose_wpqr = Dense(4,name='cls_fc_pose_wpqr')(conv_pose_wpqr_flat)
    
    # this is the model we will train
    posenet = Model(inputs=base_model.input, outputs=[cls_fc_pose_xyz, cls_fc_pose_wpqr])
    
    if weights_path:
        weights_path_ext = os.path.splitext(weights_path)[-1]
        if weights_path_ext==".h5":
            posenet.load_weights(weights_path, by_name=True)
        else:
            print("invalid weight file : " + weights_path)
            sys.exit()
    
    if not trainable:
        for layer in posenet.layers:
            layer.trainable = False
    
    return posenet

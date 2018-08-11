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
from keras import backend as K
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np
import h5py
import math
import os
import sys

def create_posenet_inception_v1(num_beacon, image_weights_path=None, beacon_weights_path=None, trainable=True):
    # not implemented
    image_beacon_posenet = None
    
    return image_beacon_posenet

def create_posenet_inception_v3(num_beacon, image_weights_path=None, beacon_weights_path=None, trainable=True):
    # image network
    # create the base pre-trained model
    image_base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # add top layer
    image_model_output = image_base_model.output
    
    # beacon network
    beacon_input = Input(shape=(num_beacon, 1, 1))
    
    beacon_icp1_out1 = Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp1_out1')(beacon_input)
    
    beacon_icp4_out1 = Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp4_out1')(beacon_icp1_out1)
    
    beacon_icp7_out1 = Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp7_out1')(beacon_icp4_out1)
    
    beacon_cls3_fc1_flat = Flatten()(beacon_icp7_out1)
    
    beacon_cls3_fc1_pose = Dense(2048,activation='relu',name='beacon_cls3_fc1_pose')(beacon_cls3_fc1_flat)
    
    # image, beacon classify 3
    image_beacon_cls3_fc1_pose = concatenate([image_model_output, beacon_cls3_fc1_pose],axis=1,name='image_beacon_cls3_fc1_pose')
    
    image_beacon_cls3_fc_pose_xyz = Dense(3,name='image_beacon_cls3_fc_pose_xyz')(image_beacon_cls3_fc1_pose)
        
    image_beacon_cls3_fc_pose_wpqr = Dense(4,name='image_beacon_cls3_fc_pose_wpqr')(image_beacon_cls3_fc1_pose)
    
    image_beacon_posenet = Model(inputs=[image_base_model.input, beacon_input], outputs=[image_beacon_cls3_fc_pose_xyz, image_beacon_cls3_fc_pose_wpqr])
    
    if image_weights_path:
	print("start load image network weights")
        image_weights_path_ext = os.path.splitext(image_weights_path)[-1]
        if image_weights_path_ext==".npy":
	    weights_data = np.load(weights_path).item()
	    for layer in image_beacon_posenet.layers:
	        if layer.name in weights_data.keys():
	            layer_weights = weights_data[layer.name]
	            layer.set_weights((layer_weights['weights'], layer_weights['biases']))
	    print("finish load beacon network weights")
        elif image_weights_path_ext==".h5":
            image_beacon_posenet.load_weights(image_weights_path, by_name=True)
	    print("finish load image network weights")
        else:
            print("invalid weight file : " + image_weights_path)
            sys.exit()
            
    if beacon_weights_path:
	print("start load beacon network weights")
        beacon_weights_path_ext = os.path.splitext(beacon_weights_path)[-1]
        if beacon_weights_path_ext==".h5":
            image_beacon_posenet.load_weights(beacon_weights_path, by_name=True)
	    print("finish load beacon network weights")
        else:
            print("invalid weight file : " + beacon_weights_path)
            sys.exit()
    
    if not trainable:
        for layer in image_beacon_posenet.layers:
            layer.trainable = False
    
    return image_beacon_posenet

def create_posenet_mobilenet_v1(num_beacon, image_weights_path=None, beacon_weights_path=None, trainable=True):
    # image network
    # create the base pre-trained model
    alpha = 1.0
    dropout=1e-3
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)
    image_base_model = MobileNet(input_shape=input_shape, alpha=alpha, dropout=dropout, weights='imagenet', include_top=False)
    
    # add top layer
    image_model_output = image_base_model.output
    
    if K.image_data_format() == 'channels_first':
        shape = (int(1024 * alpha), 1, 1)
    else:
        shape = (1, 1, int(1024 * alpha))
    
    image_model_output = GlobalAveragePooling2D()(image_model_output)
    image_model_output = Reshape(shape, name='reshape_1')(image_model_output)
    image_model_output = Dropout(dropout, name='dropout')(image_model_output)
    
    image_conv_pose_xyz = Conv2D(1024, (1, 1),
                           padding='same', name='conv_pose_xyz')(image_model_output)
    
    image_conv_pose_xyz_flat = Flatten()(image_conv_pose_xyz)
    
    image_conv_pose_wpqr = Conv2D(1024, (1, 1),
                                  padding='same', name='conv_pose_wpqr')(image_model_output)
    
    image_conv_pose_wpqr_flat = Flatten()(image_conv_pose_wpqr)
    
    # beacon network
    beacon_input = Input(shape=(num_beacon, 1, 1))
    
    beacon_icp1_out1 = Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp1_out1')(beacon_input)
    
    beacon_icp4_out1 = Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp4_out1')(beacon_icp1_out1)
    
    beacon_icp7_out1 = Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp7_out1')(beacon_icp4_out1)
    
    beacon_cls3_fc1_flat = Flatten()(beacon_icp7_out1)
    
    beacon_cls3_fc1_pose = Dense(2048,activation='relu',name='beacon_cls3_fc1_pose')(beacon_cls3_fc1_flat)
    
    # image, beacon classify 3
    image_beacon_fc_pose_xyz = concatenate([image_conv_pose_xyz_flat, beacon_cls3_fc1_pose],axis=1,name='image_beacon_fc_pose_xyz')
    
    image_beacon_cls_fc_pose_xyz = Dense(3,name='image_beacon_cls_fc_pose_xyz')(image_beacon_fc_pose_xyz)

    image_beacon_fc_pose_wpqr = concatenate([image_conv_pose_wpqr_flat, beacon_cls3_fc1_pose],axis=1,name='image_beacon_fc_pose_wpqr')
    
    image_beacon_cls_fc_pose_wpqr = Dense(4,name='image_beacon_cls_fc_pose_wpqr')(image_beacon_fc_pose_wpqr)
    
    image_beacon_posenet = Model(inputs=[image_base_model.input, beacon_input], outputs=[image_beacon_cls_fc_pose_xyz, image_beacon_cls_fc_pose_wpqr])
    
    if image_weights_path:
	print("start load image network weights")
        image_weights_path_ext = os.path.splitext(image_weights_path)[-1]
        if image_weights_path_ext==".npy":
	    weights_data = np.load(weights_path).item()
	    for layer in image_beacon_posenet.layers:
	        if layer.name in weights_data.keys():
	            layer_weights = weights_data[layer.name]
	            layer.set_weights((layer_weights['weights'], layer_weights['biases']))
	    print("finish load beacon network weights")
        elif image_weights_path_ext==".h5":
            image_beacon_posenet.load_weights(image_weights_path, by_name=True)
	    print("finish load image network weights")
        else:
            print("invalid weight file : " + image_weights_path)
            sys.exit()
            
    if beacon_weights_path:
	print("start load beacon network weights")
        beacon_weights_path_ext = os.path.splitext(beacon_weights_path)[-1]
        if beacon_weights_path_ext==".h5":
            image_beacon_posenet.load_weights(beacon_weights_path, by_name=True)
	    print("finish load beacon network weights")
        else:
            print("invalid weight file : " + beacon_weights_path)
            sys.exit()
    
    if not trainable:
        for layer in image_beacon_posenet.layers:
            layer.trainable = False
    
    return image_beacon_posenet

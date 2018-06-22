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
from keras.layers import TimeDistributed, LSTM
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

def create_posenet_inception_v1(num_beacon, image_beacon_weights_path=None, trainable=True):
    # not implemented
    image_beacon_posenet = None
    
    return image_beacon_posenet

def create_posenet_inception_v3(num_beacon, image_beacon_weights_path=None, trainable=True):
    # create the base pre-trained model
    image_input = Input(shape=(None, 299, 299, 3))
    
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False
    
    # add top layer
    model_output = TimeDistributed(base_model)(image_input)
    
    # beacon subnet 1
    beacon_input = Input(shape=(None, num_beacon, 1, 1))
    
    beacon_icp1_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp1_out1'))(beacon_input)
    beacon_icp1_out1.trainable = False
    
    # beacon subnet 2
    beacon_icp4_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp4_out1'))(beacon_icp1_out1)
    beacon_icp4_out1.trainable = False
    
    # beacon subnet 3
    beacon_icp7_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp7_out1'))(beacon_icp4_out1)
    beacon_icp7_out1.trainable = False
        
    beacon_cls3_fc1_flat = TimeDistributed(Flatten())(beacon_icp7_out1)
    beacon_cls3_fc1_flat.trainable = False
    
    beacon_cls3_fc1_pose = TimeDistributed(Dense(2048,activation='relu',name='beacon_cls3_fc1_pose'))(beacon_cls3_fc1_flat)
    beacon_cls3_fc1_pose.trainable = False

    # image, beacon classify 3
    image_beacon_cls3_fc1_pose = concatenate([model_output, beacon_cls3_fc1_pose],name='image_beacon_cls3_fc1_pose')
    
    image_beacon_lstm = LSTM(256,return_sequences=True,name='image_beacon_lstm')(image_beacon_cls3_fc1_pose)
    
    image_beacon_lstm_dense_xyz = TimeDistributed(Dense(128,activation='relu'),name='image_beacon_lstm_dense_xyz')(image_beacon_lstm)
    
    image_beacon_lstm_pose_xyz = TimeDistributed(Dense(3),name='image_beacon_lstm_pose_xyz')(image_beacon_lstm_dense_xyz)
    
    image_beacon_lstm_dense_wpqr = TimeDistributed(Dense(128,activation='relu'),name='image_beacon_lstm_dense_wpqr')(image_beacon_lstm)
    
    image_beacon_lstm_pose_wpqr = TimeDistributed(Dense(4),name='image_beacon_lstm_pose_wpqr')(image_beacon_lstm_dense_wpqr)
    
    image_beacon_posenet = Model(inputs=[image_input, beacon_input], outputs=[image_beacon_lstm_pose_xyz, image_beacon_lstm_pose_wpqr])
    
    if image_beacon_weights_path:
	print("start load image beacon network weights")
        image_beacon_weights_path_ext = os.path.splitext(image_beacon_weights_path)[-1]
        if image_beacon_weights_path_ext==".npy":
	    weights_data = np.load(image_beacon_weights_path).item()
	    for layer in image_beacon_posenet.layers:
	        if layer.name in weights_data.keys():
	            layer_weights = weights_data[layer.name]
	            layer.set_weights((layer_weights['weights'], layer_weights['biases']))
	    print("finish load imaege beacon network weights")
        elif image_beacon_weights_path_ext==".h5":
            image_beacon_posenet.load_weights(image_beacon_weights_path, by_name=True)
	    print("finish load image beacon network weights")
        else:
            print("invalid weight file : " + image_weights_path)
            sys.exit()
    
    if not trainable:
        for layer in image_beacon_posenet.layers:
            layer.trainable = False
    
    return image_beacon_posenet

def create_posenet_mobilenet_v1(num_beacon, image_beacon_weights_path=None, trainable=True):
    # create the base pre-trained model
    if K.image_data_format() == 'channels_first':
        image_input = Input(shape=(None, 3, 224, 224), name='input_1')
    else:
        image_input = Input(shape=(None, 224, 224, 3), name='input_1')
    
    alpha = 1.0
    dropout=1e-3
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)
    base_model = MobileNet(input_shape=input_shape, alpha=alpha, dropout=dropout, weights='imagenet', include_top=False)
    base_model.trainable = False
    
    # add top layer
    model_output = TimeDistributed(base_model)(image_input)
    
    if K.image_data_format() == 'channels_first':
        shape = (int(1024 * alpha), 1, 1)
    else:
        shape = (1, 1, int(1024 * alpha))
    
    model_output = TimeDistributed(GlobalAveragePooling2D())(model_output)
    model_output = TimeDistributed(Reshape(shape, name='reshape_1'))(model_output)
    model_output = TimeDistributed(Dropout(dropout, name='dropout'))(model_output)

    image_conv_pose_xyz = TimeDistributed(Conv2D(1024, (1, 1),
                                                 padding='same', name='conv_pose_xyz'))(model_output)
    
    image_conv_pose_xyz_flat = TimeDistributed(Flatten())(image_conv_pose_xyz)
    
    image_conv_pose_wpqr = TimeDistributed(Conv2D(1024, (1, 1),
                                                  padding='same', name='conv_pose_wpqr'))(model_output)
    
    image_conv_pose_wpqr_flat = TimeDistributed(Flatten())(image_conv_pose_wpqr)
    
    # beacon subnet 1
    beacon_input = Input(shape=(None, num_beacon, 1, 1), name='input_2')
    
    beacon_icp1_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp1_out1'))(beacon_input)
    beacon_icp1_out1.trainable = False
    
    # beacon subnet 2
    beacon_icp4_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp4_out1'))(beacon_icp1_out1)
    beacon_icp4_out1.trainable = False
    
    # beacon subnet 3
    beacon_icp7_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp7_out1'))(beacon_icp4_out1)
    beacon_icp7_out1.trainable = False
        
    beacon_cls3_fc1_flat = TimeDistributed(Flatten())(beacon_icp7_out1)
    beacon_cls3_fc1_flat.trainable = False
    
    beacon_cls3_fc1_pose = TimeDistributed(Dense(2048,activation='relu',name='beacon_cls3_fc1_pose'))(beacon_cls3_fc1_flat)
    beacon_cls3_fc1_pose.trainable = False

    # image, beacon classify 3
    image_beacon_cls3_fc1_pose_xyz = concatenate([image_conv_pose_xyz_flat, beacon_cls3_fc1_pose],name='image_beacon_cls3_fc1_pose_xyz')
    
    image_beacon_lstm_xyz = LSTM(256,return_sequences=True,name='image_beacon_lstm_xyz')(image_beacon_cls3_fc1_pose_xyz)
    
    image_beacon_lstm_dense_xyz = TimeDistributed(Dense(128,activation='relu'),name='image_beacon_lstm_dense_xyz')(image_beacon_lstm_xyz)
    
    image_beacon_lstm_pose_xyz = TimeDistributed(Dense(3),name='image_beacon_lstm_pose_xyz')(image_beacon_lstm_dense_xyz)
    
    
    image_beacon_cls3_fc1_pose_wpqr = concatenate([image_conv_pose_wpqr_flat, beacon_cls3_fc1_pose],name='image_beacon_cls3_fc1_pose_wpqr')
    
    image_beacon_lstm_wpqr = LSTM(256,return_sequences=True,name='image_beacon_lstm_wpqr')(image_beacon_cls3_fc1_pose_wpqr)
    
    image_beacon_lstm_dense_wpqr = TimeDistributed(Dense(128,activation='relu'),name='image_beacon_lstm_dense_wpqr')(image_beacon_lstm_wpqr)
    
    image_beacon_lstm_pose_wpqr = TimeDistributed(Dense(4),name='image_beacon_lstm_pose_wpqr')(image_beacon_lstm_dense_wpqr)
    
    image_beacon_posenet = Model(inputs=[image_input, beacon_input], outputs=[image_beacon_lstm_pose_xyz, image_beacon_lstm_pose_wpqr])
    
    if image_beacon_weights_path:
	print("start load image beacon network weights")
        image_beacon_weights_path_ext = os.path.splitext(image_beacon_weights_path)[-1]
        if image_beacon_weights_path_ext==".npy":
	    weights_data = np.load(image_beacon_weights_path).item()
	    for layer in image_beacon_posenet.layers:
	        if layer.name in weights_data.keys():
	            layer_weights = weights_data[layer.name]
	            layer.set_weights((layer_weights['weights'], layer_weights['biases']))
	    print("finish load imaege beacon network weights")
        elif image_beacon_weights_path_ext==".h5":
            image_beacon_posenet.load_weights(image_beacon_weights_path, by_name=True)
	    print("finish load image beacon network weights")
        else:
            print("invalid weight file : " + image_weights_path)
            sys.exit()
    
    if not trainable:
        for layer in image_beacon_posenet.layers:
            layer.trainable = False
    
    return image_beacon_posenet

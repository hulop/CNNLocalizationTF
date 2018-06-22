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
from keras.models import Model
import tensorflow as tf
import numpy as np
import h5py
import math
import os
import sys

def create_posenet(num_beacon, weights_path=None, trainable=True):
    beacon_input = Input(shape=(None, num_beacon, 1, 1))
    
    beacon_icp1_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp1_out1'))(beacon_input)
    beacon_icp1_out1.trainable = False
    
    '''
    beacon_cls1_fc1_flat = TimeDistributed(Flatten())(beacon_icp1_out1)
    
    beacon_cls1_fc1_pose = TimeDistributed(Dense(1024,activation='relu',name='beacon_cls1_fc1_pose'))(beacon_cls1_fc1_flat)
    
    beacon_cls1_fc_pose_xyz = TimeDistributed(Dense(3,name='beacon_cls1_fc_pose_xyz'))(beacon_cls1_fc1_pose)
    
    beacon_cls1_fc_pose_wpqr = TimeDistributed(Dense(4,name='beacon_cls1_fc_pose_wpqr'))(beacon_cls1_fc1_pose)
    '''
    
    beacon_icp4_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp4_out1'))(beacon_icp1_out1)
    beacon_icp4_out1.trainable = False

    '''
    beacon_cls2_fc1_flat = TimeDistributed(Flatten())(beacon_icp4_out1)
    
    beacon_cls2_fc1 = TimeDistributed(Dense(1024,activation='relu',name='beacon_cls2_fc1'))(beacon_cls2_fc1_flat)
    
    beacon_cls2_fc_pose_xyz = TimeDistributed(Dense(3,name='beacon_cls2_fc_pose_xyz'))(beacon_cls2_fc1)
    
    beacon_cls2_fc_pose_wpqr = TimeDistributed(Dense(4,name='beacon_cls2_fc_pose_wpqr'))(beacon_cls2_fc1)
    '''
    
    beacon_icp7_out1 = TimeDistributed(Conv2D(16,(1,1),padding='same',activation='relu',name='beacon_icp7_out1'))(beacon_icp4_out1)
    beacon_icp7_out1.trainable = False

    beacon_cls3_fc1_flat = TimeDistributed(Flatten())(beacon_icp7_out1)
    beacon_cls3_fc1_flat.trainable = False
    
    beacon_cls3_fc1_pose = TimeDistributed(Dense(2048,activation='relu',name='beacon_cls3_fc1_pose'))(beacon_cls3_fc1_flat)
    beacon_cls3_fc1_pose.trainable = False
    
    beacon_lstm = LSTM(256,return_sequences=True,name='beacon_lstm')(beacon_cls3_fc1_pose)

    beacon_lstm_dense_xyz = TimeDistributed(Dense(128,activation='relu'),name='beacon_lstm_dense_xyz')(beacon_lstm)
    
    beacon_lstm_pose_xyz = TimeDistributed(Dense(3),name='beacon_lstm_pose_xyz')(beacon_lstm_dense_xyz)

    beacon_lstm_dense_wpqr = TimeDistributed(Dense(128,activation='relu'),name='beacon_lstm_dense_wpqr')(beacon_lstm)
    
    beacon_lstm_pose_wpqr = TimeDistributed(Dense(4),name='beacon_lstm_pose_wpqr')(beacon_lstm_dense_wpqr)
    
    beacon_posenet = Model(inputs=beacon_input, outputs=[beacon_lstm_pose_xyz, beacon_lstm_pose_wpqr])
    
    if weights_path:
	print("start load image network weights")
        beacon_posenet.load_weights(weights_path, by_name=True)
	print("finish load image network weights")

    if not trainable:
        for layer in beacon_posenet.layers:
            layer.trainable = False
    
    return beacon_posenet

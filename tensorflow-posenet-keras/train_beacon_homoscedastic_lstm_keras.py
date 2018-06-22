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

import argparse
import os
import sys
import numpy as np
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import posenet_geom_utils as posenet_geom_utils
import posenet_beacon_utils as posenet_beacon_utils
import posenet_data_utils as posenet_data_utils
import posenet_beacon_no_inception_shrink_lstm_keras
import posenet_homoscedastic_loss
from posenet_config import posenet_config
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import cv2
from tqdm import tqdm
import hulo_ibeacon.IBeaconUtils as IBeaconUtils
import math

def main():
        description = 'This script is for testing posenet'
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('input_txt_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path of input txt file in Cambridge Visual Landmark Dataset format.')
        parser.add_argument('input_beacon_setting_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path where beacon setting file is saved.')
        parser.add_argument('output_model_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where output models will be saved.')
        parser.add_argument('output_log_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where log files will be saved.')
        parser.add_argument('-w', '--input_beacon_weight_file', action='store', type=str, default=None, \
                            help='File path beacon posenet weiht is stored in numpy format.')
        parser.add_argument('-s', '--use_shrink_model', action='store_true', default=False, \
                            help='Use shrink model (default: False)')
        parser.add_argument('-f', '--use_fixed_input_mean_std', action='store_true', default=False, \
                            help='Use fixed input mean and std (default: False)')
        parser.add_argument('-a', '--use_augmentation_beacon', action='store_true', default=False, \
                            help='Use data augmentation for beacon data (default: False)')
        parser.add_argument('-e', '--epochs', action='store', type=int, default=posenet_config.epochs, \
                            help='Epochs (Default : ' + str(posenet_config.epochs))
        parser.add_argument('-b', '--batch_size', action='store', type=int, default=posenet_config.lstm_batch_size, \
                            help='Batch size (Default : ' + str(posenet_config.lstm_batch_size))
        args = parser.parse_args()
        input_txt_file = args.input_txt_file
        input_beacon_setting_file = args.input_beacon_setting_file
        output_model_dir = args.output_model_dir
        output_log_dir = args.output_log_dir
        input_beacon_weight_file = args.input_beacon_weight_file        
        use_shrink_model = args.use_shrink_model
        use_fixed_input_mean_std = args.use_fixed_input_mean_std
        use_augmentation_beacon = args.use_augmentation_beacon
        posenet_config.epochs = args.epochs
        posenet_config.lstm_batch_size = args.batch_size
        print "epochs : " + str(posenet_config.epochs)
        print "batch size : " + str(posenet_config.lstm_batch_size)
        print "use shrink model for training : " + str(use_shrink_model)
        print "use fixed input mean and std : " + str(use_fixed_input_mean_std)
        print "use beacon data augmentation : " + str(use_augmentation_beacon)
        if input_beacon_weight_file is None:
                print "please specify one of initial weight or restore directory"
                sys.exit()
        
        # parse beacon setting file
        beaconmap = IBeaconUtils.parseBeaconSetting(input_beacon_setting_file)
        beacon_num = len(beaconmap.keys())
        
        output_numpy_mean_beacon_file = os.path.join(output_model_dir, "mean_beacon.npy")
        output_numpy_model_file = os.path.join(output_model_dir, "model.npy")
        output_model_file = os.path.join(output_model_dir, "model.ckpt")
        
	datasource, mean_beacon = posenet_data_utils.get_beacon_data(input_txt_file, beaconmap, beacon_num, use_fixed_input_mean_std, use_augmentation_beacon)
        if use_fixed_input_mean_std:
            print("Skip save mean beacon")
        else:
            with open(output_numpy_mean_beacon_file, 'wb') as fw:
                np.save(fw, mean_beacon)
            print("Save mean beacon at: " + output_numpy_mean_beacon_file)
        
        # Set GPU options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        KTF.set_session(session)
        
        # Train model
        s_x = K.variable(value=0.0)
        s_q = K.variable(value=-3.0)
        if use_shrink_model:
                model = posenet_beacon_no_inception_shrink_lstm_keras.create_posenet(beacon_num, input_beacon_weight_file)
        else:
                print "Do not shrink model is not supported"
                sys.exit()
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001)
        euc_loss3x_s = posenet_homoscedastic_loss.euc_loss3x_s(s_x=s_x)
        euc_loss3q_s = posenet_homoscedastic_loss.euc_loss3q_s(s_q=s_q)
        model.compile(optimizer=adam, loss={'beacon_lstm_pose_xyz': euc_loss3x_s, 'beacon_lstm_pose_wpqr': euc_loss3q_s})
        model.summary()
        
        # Setup checkpointing
        checkpointer = ModelCheckpoint(filepath=os.path.join(output_model_dir, "checkpoint_weights.h5"), verbose=1, save_weights_only=True, period=1)
        
        # Save Tensorboard log
        logger = TensorBoard(log_dir=output_log_dir, histogram_freq=0, write_graph=True)
        
        # Adjust Epoch size depending on beacon data augmentation
        if use_augmentation_beacon:
                posenet_config.epochs = posenet_config.epochs/posenet_config.num_beacon_augmentation
        steps_per_epoch = int(len(datasource.poses_index)/float(posenet_config.lstm_batch_size))
        num_iterations = steps_per_epoch*posenet_config.epochs
        print("Number of epochs : " + str(posenet_config.epochs))
        print("Number of training data : " + str(len(datasource.poses_index)))
        print("Number of iterations : " + str(num_iterations))
        
	history = model.fit_generator(posenet_data_utils.gen_beacon_lstm_data_batch(datasource), steps_per_epoch=steps_per_epoch,
                                      epochs=posenet_config.epochs, callbacks=[checkpointer, logger])
        
        model.save_weights(os.path.join(output_model_dir, "trained_weights.h5"))

if __name__ == '__main__':
	main()

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
import posenet_image_utils as posenet_image_utils
import posenet_data_utils as posenet_data_utils
import posenet_lstm_keras
import posenet_homoscedastic_loss
from posenet_config import posenet_config
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import cv2
from tqdm import tqdm
import math

def main():
        description = 'This script is for testing posenet'
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('input_txt_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path of input txt file in Cambridge Visual Landmark Dataset format.')
        parser.add_argument('output_model_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where output models will be saved.')
        parser.add_argument('output_log_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where log files will be saved.')
        parser.add_argument('-i', '--input_image_weight_file', action='store', type=str, default=None, \
                            help='File path image posenet weiht is stored in numpy format.')
        parser.add_argument('-f', '--use_fixed_input_mean_std', action='store_true', default=False, \
                            help='Use fixed image mean and std (default: False)')
        parser.add_argument('-m', '--base_model', action='store', type=str, default=posenet_config.base_model, \
                            help='Base model : inception-v1/inception-v3/mobilenet-v1 (Default : ' + str(posenet_config.base_model))
        parser.add_argument('-e', '--epochs', action='store', type=int, default=posenet_config.epochs, \
                            help='Epochs (Default : ' + str(posenet_config.epochs))
        parser.add_argument('-b', '--batch_size', action='store', type=int, default=posenet_config.lstm_batch_size, \
                            help='Batch size (Default : ' + str(posenet_config.lstm_batch_size))
        args = parser.parse_args()
        input_txt_file = args.input_txt_file
        output_model_dir = args.output_model_dir
        output_log_dir = args.output_log_dir
        input_image_weight_file = args.input_image_weight_file
        use_fixed_input_mean_std = args.use_fixed_input_mean_std
        posenet_config.base_model = args.base_model
        posenet_config.epochs = args.epochs
        posenet_config.lstm_batch_size = args.batch_size
        print "base model : " + str(posenet_config.base_model)
        print "epochs : " + str(posenet_config.epochs)
        print "batch size : " + str(posenet_config.lstm_batch_size)
        print "use fixed input mean and std : " + str(use_fixed_input_mean_std)
        if posenet_config.base_model!="inception-v1" and posenet_config.base_model!="inception-v3" and posenet_config.base_model!="mobilenet-v1":
                print "invalid base model : " + posenet_config.base_model
                sys.exit()
        if input_image_weight_file is None:
                print "please specify initial weight for inception-v1"
                sys.exit()
        
        input_image_dir = os.path.dirname(input_txt_file)
        output_numpy_mean_image_file = os.path.join(output_model_dir, "mean_image.npy")
        output_numpy_model_file = os.path.join(output_model_dir, "model.npy")
        output_model_file = os.path.join(output_model_dir, "model.ckpt")

        if posenet_config.base_model=="inception-v1":
                image_size = 224
                output_auxiliary = True
        elif posenet_config.base_model=="inception-v3":
                image_size = 299
                output_auxiliary = False
        elif posenet_config.base_model=="mobilenet-v1":
                image_size = 224
                output_auxiliary = False
        else:
                print "invalid base model : " + posenet_config.base_model
                sys.exit()

        datasource, mean_image = posenet_data_utils.get_image_data(input_txt_file, input_image_dir, use_fixed_input_mean_std, image_size=image_size)
        if use_fixed_input_mean_std:
                print("Skip save mean image")
        else:
                with open(output_numpy_mean_image_file, 'wb') as fw:
                        np.save(fw, mean_image)
                print("Save mean image at: " + output_numpy_mean_image_file)
        
        # Set GPU options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        KTF.set_session(session)
        
        # Train model
        s_x = K.variable(value=0.0)
        s_q = K.variable(value=-3.0)
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001)
        euc_loss3x_s = posenet_homoscedastic_loss.euc_loss3x_s(s_x=s_x)
        euc_loss3q_s = posenet_homoscedastic_loss.euc_loss3q_s(s_q=s_q)
        if posenet_config.base_model=="inception-v1":
                model = posenet_lstm_keras.create_posenet_inception_v1(input_image_weight_file) # GoogLeNet (Trained on Places)
                model.compile(optimizer=adam, loss={'lstm_pose_xyz': euc_loss3x_s, 'lstm_pose_wpqr': euc_loss3q_s})
        elif posenet_config.base_model=="inception-v3":
                model = posenet_lstm_keras.create_posenet_inception_v3(input_image_weight_file)
                model.compile(optimizer=adam, loss={'lstm_pose_xyz': euc_loss3x_s, 'lstm_pose_wpqr': euc_loss3q_s})
        elif posenet_config.base_model=="mobilenet-v1":
                model = posenet_lstm_keras.create_posenet_mobilenet_v1(input_image_weight_file)
                model.compile(optimizer=adam, loss={'lstm_pose_xyz': euc_loss3x_s, 'lstm_pose_wpqr': euc_loss3q_s})
        else:
                print "invalid base model : " + posenet_config.base_model
                sys.exit()
        model.summary()
        
        # Setup checkpointing
        checkpointer = ModelCheckpoint(filepath=os.path.join(output_model_dir, "checkpoint_weights.h5"), verbose=1, save_weights_only=True, period=1)
        
        # Save Tensorboard log
        logger = TensorBoard(log_dir=output_log_dir, histogram_freq=0, write_graph=True)
        
        steps_per_epoch = int(len(datasource.poses_index)/float(posenet_config.lstm_batch_size))
        num_iterations = steps_per_epoch*posenet_config.epochs
        print("Number of epochs : " + str(posenet_config.epochs))
        print("Number of training data : " + str(len(datasource.poses_index)))
        print("Number of iterations : " + str(num_iterations))
        
	history = model.fit_generator(posenet_data_utils.gen_image_lstm_data_batch(datasource, batch_size=posenet_config.lstm_batch_size),
                                      steps_per_epoch=steps_per_epoch, epochs=posenet_config.epochs,
                                      callbacks=[checkpointer, logger])
        
        model.save_weights(os.path.join(output_model_dir, "trained_weights.h5"))

if __name__ == '__main__':
	main()

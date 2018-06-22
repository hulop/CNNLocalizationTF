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
import tensorflow as tf
import keras
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import posenet_beacon_no_inception_shrink_keras
import posenet_beacon_no_inception_shrink_lstm_keras
from tensorflow.python.platform import gfile
import hulo_ibeacon.IBeaconUtils as IBeaconUtils

def main():
        global num_dense_sample
        
        description = 'This script is for testing posenet'
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('input_beacon_setting_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path where beacon setting file is saved.')
        parser.add_argument('input_model_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where input model is saved.')
        parser.add_argument('output_graph_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path where exported graph def protobuf (.pb) file will be saved.')
        parser.add_argument('output_model_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where output model will be saved.')
        parser.add_argument('-s', '--use_shrink_model', action='store_true', default=False, \
                            help='Use shrink model (default: False)')
        parser.add_argument('-l', '--lstm_model', action='store_true', default=False, \
                            help='Export LSTM model (default: False)')
        args = parser.parse_args()
        input_beacon_setting_file = args.input_beacon_setting_file
        input_model_dir = args.input_model_dir
        output_graph_file = args.output_graph_file
        output_model_dir = args.output_model_dir
        output_model_file = os.path.join(output_model_dir, "model.ckpt")        
        use_shrink_model = args.use_shrink_model
        lstm_model = args.lstm_model
        print "use shrink model for training : " + str(use_shrink_model)
        
        # parse beacon setting file
        beaconmap = IBeaconUtils.parseBeaconSetting(input_beacon_setting_file)
        beacon_num = len(beaconmap.keys())
        
        # convert hd5 file to ckpt
        # https://github.com/keras-team/keras/issues/9040
        K.set_learning_phase(0)
        if use_shrink_model:
                if lstm_model:
                        model = posenet_beacon_no_inception_shrink_lstm_keras.create_posenet(beacon_num, trainable=False)
                else:
                        model = posenet_beacon_no_inception_shrink_keras.create_posenet(beacon_num, trainable=False)
        else:
                print "Do not shrink model is not supported"
                sys.exit()
        model.load_weights(os.path.join(input_model_dir, 'trained_weights.h5'))
        model.summary()

        #Save graph and checkpoint
        session = keras.backend.get_session()
        graph = session.graph
        graph_def = graph.as_graph_def()
        with gfile.GFile(output_graph_file, 'wb') as f:
            f.write(graph_def.SerializeToString())
        
        saver = tf.train.Saver()
        saver.save(session, output_model_file)
        
if __name__ == '__main__':
	main()

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
import subprocess
import tensorflow as tf
from posenet_config import posenet_config

tensorflow_dir = "~/opt/tensorflow-1.4.1"

def main():
        global tensorflow_dir
        
        description = 'This script is for testing posenet'
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('input_graph_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path where exported graph def protobuf (.pb) file will be saved.')
        parser.add_argument('checkpoint_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where trained model files are be saved.')
        parser.add_argument('output_graph_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where freezed, optimized, quantized files will be saved.')
        parser.add_argument('-m', '--base_model', action='store', type=str, default=posenet_config.base_model, \
                            help='Base model : inception-v1/inception-v3/mobilenet-v1 (Default : ' + str(posenet_config.base_model))
        args = parser.parse_args()
        input_graph_file = args.input_graph_file
        checkpoint_dir = args.checkpoint_dir
        output_graph_dir = args.output_graph_dir
        posenet_config.base_model = args.base_model
        print "base model : " + str(posenet_config.base_model)
        
        #Save frozon graph, optimized graph, and quantized graph from graph definition and checkpoint
        latest_checkpoint_filepath = tf.train.latest_checkpoint(checkpoint_dir)
        
        # you can check output node name by tensorflow/tools/graph_transforms::summarize_graph
        # https://github.com/tensorflow/models/tree/master/research/slim#Export
        if posenet_config.base_model=="inception-v1":
                output_node_names = "image_beacon_cls3_fc_pose_xyz/BiasAdd,image_beacon_cls3_fc_pose_wpqr/BiasAdd"
        elif posenet_config.base_model=="inception-v3" or posenet_config.base_model=="mobilenet-v1":
                output_node_names = "image_beacon_cls_fc_pose_xyz/BiasAdd,image_beacon_cls_fc_pose_wpqr/BiasAdd"
        else:
                print "invalid base model : " + posenet_config.base_model
                sys.exit()
        
        output_frozen_graph_filepath = os.path.join(output_graph_dir, 'frozen_graph.pb')
        freeze_graph_command_exec = os.path.join(tensorflow_dir, "bazel-bin/tensorflow/python/tools/freeze_graph")
        if not os.path.exists(freeze_graph_command_exec):
                print("fatal error, cannot find command : " + freeze_graph_command_exec)
                sys.exit()
        freeze_graph_command_env = os.environ.copy()
        freeze_graph_command_env["CUDA_VISIBLE_DEVICES"] = ''
        freeze_graph_command = []
        freeze_graph_command.append(freeze_graph_command_exec)
        freeze_graph_command.append("--input_graph=" + input_graph_file)
        freeze_graph_command.append("--input_checkpoint=" + latest_checkpoint_filepath)
        freeze_graph_command.append("--input_binary=true")
        freeze_graph_command.append("--output_graph=" + output_frozen_graph_filepath)
        freeze_graph_command.append("--output_node_names=" + output_node_names)
        print("start exec:" + " ".join(freeze_graph_command))
        proc = subprocess.Popen(freeze_graph_command, env=freeze_graph_command_env)
        print("freeze graph process ID=" + str(proc.pid))
        proc.communicate()
        print("finish exec:" + " ".join(freeze_graph_command))

        output_optimized_graph_filepath = os.path.join(output_graph_dir, 'optimized_graph.pb')
        optimize_graph_command_exec = os.path.join(tensorflow_dir, "bazel-bin/tensorflow/python/tools/optimize_for_inference")
        if not os.path.exists(optimize_graph_command_exec):
                print("fatal error, cannot find command : " + optimize_graph_command_exec)
                sys.exit()
        optimize_graph_command_env = os.environ.copy()
        optimize_graph_command_env["CUDA_VISIBLE_DEVICES"] = ''
        optimize_graph_command = []
        optimize_graph_command.append(optimize_graph_command_exec)
        optimize_graph_command.append("--input=" + output_frozen_graph_filepath)
        optimize_graph_command.append("--output=" + output_optimized_graph_filepath)
        optimize_graph_command.append("--input_names=input_1,input_2")
        optimize_graph_command.append("--output_names=" + output_node_names)
        optimize_graph_command.append("--frozen_graph=true")
        print("start exec:" + " ".join(optimize_graph_command))
        proc = subprocess.Popen(optimize_graph_command, env=optimize_graph_command_env)
        print("optimize graph process ID=" + str(proc.pid))
        proc.communicate()
        print("finish exec:" + " ".join(optimize_graph_command))
        
        output_quantized_graph_filepath = os.path.join(output_graph_dir, 'quantized_graph.pb')
        quantize_graph_command_exec = os.path.join(tensorflow_dir, "bazel-bin/tensorflow/tools/quantization/quantize_graph")
        if not os.path.exists(quantize_graph_command_exec):
                print("fatal error, cannot find command : " + quantize_graph_command_exec)
                sys.exit()
        quantize_graph_command_env = os.environ.copy()
        quantize_graph_command_env["CUDA_VISIBLE_DEVICES"] = ''
        quantize_graph_command = []
        quantize_graph_command.append(quantize_graph_command_exec)
        quantize_graph_command.append("--input=" + output_optimized_graph_filepath)
        quantize_graph_command.append("--output=" + output_quantized_graph_filepath)
        quantize_graph_command.append("--input_node_names=input_1,input_2")
        quantize_graph_command.append("--output_node_names=" + output_node_names)
        quantize_graph_command.append("--mode=eightbit")
        print("start exec:" + " ".join(quantize_graph_command))
        proc = subprocess.Popen(quantize_graph_command, env=quantize_graph_command_env)
        print("quantize graph process ID=" + str(proc.pid))
        proc.communicate()
        print("finish exec:" + " ".join(quantize_graph_command))
        
if __name__ == '__main__':
	main()

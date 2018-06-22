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
import numpy as np
import random
import tensorflow as tf
import posenet_beacon_utils as posenet_beacon_utils
import cv2
from tqdm import tqdm
import math
import hulo_ibeacon.IBeaconUtils as IBeaconUtils
import time

fixed_beacon_mean = 0.0
fixed_beacon_std = 1.0

input_beacon_layer_name = "input_1"
output_pos_layer_name = "beacon_cls3_fc_pose_xyz/BiasAdd"
output_rot_layer_name = "beacon_cls3_fc_pose_wpqr/BiasAdd"
        
# Refer this file how to run inference from graph file
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    # print imported graph for debug
    imported_graph_nodes = [n.name for n in graph_def.node]
    for node in imported_graph_nodes:
        print("Imported node : " + str(node))
    
    return graph

class datasource(object):
	def __init__(self, beacons, poses):
		self.beacons = beacons
		self.poses = poses

def preprocess(beacons, mean_beacon, use_fixed_input_mean_std):
	beacons_out = []
        
	# subtract mean beacon
	for X in tqdm(beacons):
            if use_fixed_input_mean_std:
                X = X - fixed_beacon_mean
                X = X / fixed_beacon_std
            else:
		X = X - mean_beacon
            Y = np.expand_dims(X, axis=0)
	    beacons_out.append(Y)
	return beacons_out

#
# This function is for making training easier
#
# set norm of quaternion as 1
# set quaternion to northern hemisphere
#
def preprocess_quaternion(quat):
        if len(quat)!=4:
                print "invalid input quaternion"
                sys.exit()
        np_quat = np.array(quat)
        norm_quat = np.linalg.norm(np_quat)
        np_quat = np_quat / norm_quat
        if quat[0]<0:
                np_quat = np_quat * -1.0
        return np_quat.tolist()

def get_data(input_txt_file, beaconmap, mean_beacon, use_fixed_input_mean_std):
	poses = []
	beacons = []
        
	with open(input_txt_file) as f:
		# skip header
		next(f)
		next(f)
		next(f)
		for line in f:
			beaconstr, p0,p1,p2,p3,p4,p5,p6 = line.split()
                        quat = preprocess_quaternion([float(p3),float(p4),float(p5),float(p6)])
                        
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = quat[0]
			p4 = quat[1]
			p5 = quat[2]
			p6 = quat[3]
			poses.append((p0,p1,p2,p3,p4,p5,p6))
                        beacon = posenet_beacon_utils.parse_beacon_string(beaconstr, beaconmap)
			beacons.append(beacon)
	beacons = preprocess(beacons, mean_beacon, use_fixed_input_mean_std)
	return datasource(beacons, poses)

def main():
        description = 'This script is for testing posenet'
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('input_txt_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path of input txt file in Cambridge Visual Landmark Dataset format.')
        parser.add_argument('input_beacon_setting_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path where beacon setting file is saved.')
        parser.add_argument('input_pb_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path of model pb file.')
        parser.add_argument('result_log_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where localization result files are saved.')
        parser.add_argument('-f', '--use_fixed_input_mean_std', action='store_true', default=False, \
                            help='Use fixed input mean and std (default: False)')
        args = parser.parse_args()
        input_txt_file = args.input_txt_file
        input_beacon_setting_file = args.input_beacon_setting_file
        input_pb_file = args.input_pb_file
        result_log_dir = args.result_log_dir
        use_fixed_input_mean_std = args.use_fixed_input_mean_std
        
        input_model_dir = os.path.dirname(input_pb_file)
        input_numpy_mean_beacon_file = os.path.join(input_model_dir, "mean_beacon.npy")
        if use_fixed_input_mean_std:
            input_numpy_mean_beacon = None
        else:
            input_numpy_mean_beacon = np.load(input_numpy_mean_beacon_file)
        output_summary_log_file = os.path.join(result_log_dir, "summary-log.txt")
        output_hist_log_file = os.path.join(result_log_dir, "hist-log.txt")

        # parse beacon setting file
        beaconmap = IBeaconUtils.parseBeaconSetting(input_beacon_setting_file)
        beacon_num = len(beaconmap.keys())
        
	beacons = tf.placeholder(tf.float32, [1, beacon_num, 1, 1])
	datasource = get_data(input_txt_file, beaconmap, input_numpy_mean_beacon, use_fixed_input_mean_std)
	results = np.zeros((len(datasource.beacons),2))
        
	# Set GPU options
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        
        time_array = np.array([])

        # Load model
        graph = load_graph(input_pb_file)
        input_beacon_name = "import/" + input_beacon_layer_name
        output_pos_name = "import/" + output_pos_layer_name
        output_rot_name = "import/" + output_rot_layer_name
        input_operation = graph.get_operation_by_name(input_beacon_name)
        output_pos_operation = graph.get_operation_by_name(output_pos_name)
        output_rot_operation = graph.get_operation_by_name(output_rot_name)
        
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as sess:
	    for i in range(len(datasource.beacons)):
			np_beacon = datasource.beacons[i]
			feed = {input_operation.outputs[0]: np_beacon}
                        
			pose_q= np.asarray(datasource.poses[i][3:7])
			pose_x= np.asarray(datasource.poses[i][0:3])
                        start_time = time.time()
			predicted_x, predicted_q = sess.run([output_pos_operation.outputs[0],
                                                             output_rot_operation.outputs[0]], feed_dict=feed)
                        elapsed_time = time.time() - start_time
                        time_array = np.append(time_array, elapsed_time)
                        
			pose_q = np.squeeze(pose_q)
			pose_x = np.squeeze(pose_x)
			predicted_q = np.squeeze(predicted_q)
			predicted_x = np.squeeze(predicted_x)

			# calculate error
			q1 = pose_q / np.linalg.norm(pose_q)
			q2 = predicted_q / np.linalg.norm(predicted_q)
			d = abs(np.sum(np.multiply(q1,q2)))
                        # fix floating point inaccuracy
                        if d<-1.0:
                                d = -1.0
                        if d>1.0:
                                d = 1.0
			theta = 2 * np.arccos(d) * 180/math.pi
			error_x = np.linalg.norm(pose_x-predicted_x)
			results[i,:] = [error_x,theta]
			print 'Index=', i, ' , Pos Error(m)=', error_x, ',  Rot Error(degrees)=', theta

        # write histgram results
        bin_edge = [0.01*float(x) for x in range(0,1001)]
        dist_errors = results[:,0]
        dist_hist, dist_hist_bins = np.histogram(dist_errors, bins=bin_edge)
        dist_hist_cum_ratio = np.cumsum(dist_hist) / float(len(datasource.beacons))
        print "Histogram of error: " + str(dist_hist)
        print "Cumulative ratio of error: " + str(dist_hist_cum_ratio)
        print "Total loc err larger than " + str(np.max(bin_edge)) + " meters: " + str(len(datasource.beacons)-np.sum(dist_hist))
        
        # write summary of results
	mean_result = np.mean(results,axis=0)
	std_result = np.std(results,axis=0)
	median_result = np.median(results,axis=0)
	max_result = np.max(results,axis=0)        
        percentile_80_result = np.percentile(results,80,axis=0)
        percentile_90_result = np.percentile(results,90,axis=0)
        percentile_95_result = np.percentile(results,95,axis=0)
	print 'Mean error ', mean_result[0], 'm  and ', mean_result[1], 'degrees.'
	print 'StdDev error ', std_result[0], 'm  and ', std_result[1], 'degrees.'
	print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'
	print 'Max error ', max_result[0], 'm  and ', max_result[1], 'degrees.'        
	print '80 percentile error ', percentile_80_result[0], 'm  and ', percentile_80_result[1], 'degrees.'
	print '90 percentile error ', percentile_90_result[0], 'm  and ', percentile_90_result[1], 'degrees.'
	print '95 percentile error ', percentile_95_result[0], 'm  and ', percentile_95_result[1], 'degrees.'
	print 'Mean time ', str(np.average(time_array))
	print 'StdDev time ', str(np.std(time_array))
	print 'Median time ', str(np.median(time_array))
        with open(output_summary_log_file, "w") as fw:
                fw.write("Number of test image = " + str(len(datasource.beacons)) + "\n")
                fw.write("Mean error = " + str(mean_result[0]) + " meters." + "\n")
                fw.write("StdDev error = " + str(std_result[0]) + " meters." + "\n")
                fw.write("Median error = " + str(median_result[0]) + " meters." + "\n")
                fw.write("Max error = " + str(max_result[0]) + " meters." + "\n")                
	        fw.write("80 percentile error = " + str(percentile_80_result[0]) + " meters." + "\n")
	        fw.write("90 percentile error = " + str(percentile_90_result[0]) + " meters." + "\n")
	        fw.write("95 percentile error = " + str(percentile_95_result[0]) + " meters." + "\n")                
                fw.write("\n")
                fw.write("Mean error = " + str(mean_result[1]) + " degrees." + "\n")
                fw.write("StdDev error = " + str(std_result[1]) + " degrees." + "\n")
                fw.write("Median error = " + str(median_result[1]) + " degrees." + "\n")
                fw.write("Max error = " + str(max_result[1]) + " degrees." + "\n")                
	        fw.write("80 percentile error = " + str(percentile_80_result[1]) + " degrees." + "\n")
                fw.write("90 percentile error = " + str(percentile_90_result[1]) + " degrees." + "\n")
                fw.write("95 percentile error = " + str(percentile_95_result[1]) + " degrees." + "\n")
                fw.write("\n")                
                fw.write("Histogram of error: " + str(dist_hist) + "\n")
                fw.write("Cumulative ratio: " + str(np.around(np.cumsum(dist_hist,dtype=float)/len(datasource.beacons),2)) + "\n")
                fw.write("Total loc err larger than " + str(np.max(bin_edge)) + " meters: " + str(len(datasource.beacons)-np.sum(dist_hist)) + "\n")
                fw.write("\n")
                fw.write("Mean time = " + str(np.average(time_array)) + "\n")
                fw.write("StdDev time = " + str(np.std(time_array)) + "\n")
                fw.write("Median time = " + str(np.median(time_array)) + "\n")
        # write error histgram
        np.savetxt(output_hist_log_file, zip(dist_hist_bins, dist_hist_cum_ratio), delimiter=',')
        # write error histgram
        np.savetxt(output_hist_log_file, zip(dist_hist_bins, dist_hist_cum_ratio), delimiter=',')

if __name__ == '__main__':
	main()

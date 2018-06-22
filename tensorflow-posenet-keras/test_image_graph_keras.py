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
import tensorflow as tf
from posenet_config import posenet_config
import posenet_image_utils as posenet_image_utils
import cv2
from tqdm import tqdm
import math
import transformations as transformations
import json
import time

num_dense_sample = 100
fixed_image_mean = 128.0
fixed_image_std = 1.0

input_image_layer_name = "input_1"
output_pos_layer_name = "cls3_fc_pose_xyz/BiasAdd"
output_rot_layer_name = "cls3_fc_pose_wpqr/BiasAdd"

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
	def __init__(self, image_filenames, poses):
		self.image_filenames = image_filenames
		self.poses = poses

def preprocess(image, mean_image, use_fixed_input_mean_std, num_sample, random_crop):
	images_out = []
        
	# resize and crop
	images_cropped = []

	X = cv2.imread(image)
	X = cv2.resize(X, (455, 256))
        
        for _ in range(num_sample):
                if random_crop:
	                X = posenet_image_utils.randomCrop(X, 224)
                else:
		        X = posenet_image_utils.centeredCrop(X, 224)
		images_cropped.append(X)
        
	# subtract mean image
	for X in images_cropped:
		X = np.transpose(X,(2,0,1))
                if use_fixed_input_mean_std:
                        X = X - fixed_image_mean
                        X = X / fixed_image_std
                else:
		        X = X - mean_image
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		images_out.append(X)
	return np.array(images_out)

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

def get_data(input_txt_file, input_image_dir):
	image_filenames = []
	poses = []
        
	with open(input_txt_file) as f:
		# skip header
		next(f)
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
                        quat = preprocess_quaternion([float(p3),float(p4),float(p5),float(p6)])
                        
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = quat[0]
			p4 = quat[1]
			p5 = quat[2]
			p6 = quat[3]
			poses.append((p0,p1,p2,p3,p4,p5,p6))
			image_filenames.append(os.path.join(input_image_dir, fname))
	return datasource(image_filenames, poses), image_filenames

def main():
        global num_dense_sample
        global output_pos_layer_name
        global output_rot_layer_name
        
        description = 'This script is for testing posenet'
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('input_txt_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path of input txt file in Cambridge Visual Landmark Dataset format.')
        parser.add_argument('input_pb_file', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='File path of model pb file.')
        parser.add_argument('result_log_dir', action='store', nargs=None, const=None, \
                            default=None, type=str, choices=None, metavar=None, \
                            help='Directory path where localization result files are saved.')
        parser.add_argument('-d', '--use_dense_sample', action='store_true', default=False, \
                            help='Use dense sample for testing (default: False)')
        parser.add_argument('-n', '--num_dense_sample', action='store', type=int, default=num_dense_sample, \
                            help='Number of dense sample (Default : ' + str(num_dense_sample))
        parser.add_argument('-f', '--use_fixed_input_mean_std', action='store_true', default=False, \
                            help='Use fixed input mean and std (default: False)')
        parser.add_argument('-m', '--base_model', action='store', type=str, default=posenet_config.base_model, \
                            help='Base model : inception-v1/inception-v3/mobilenet-v1 (Default : ' + str(posenet_config.base_model))
        args = parser.parse_args()
        input_txt_file = args.input_txt_file
        input_pb_file = args.input_pb_file
        result_log_dir = args.result_log_dir
        use_dense_sample = args.use_dense_sample
        num_dense_sample = args.num_dense_sample
        use_fixed_input_mean_std = args.use_fixed_input_mean_std
        posenet_config.base_model = args.base_model
        print "base model : " + str(posenet_config.base_model)

        # you can check output node name by tensorflow/tools/graph_transforms::summarize_graph
        # https://github.com/tensorflow/models/tree/master/research/slim#Export
        if posenet_config.base_model=="inception-v1":
            output_pos_layer_name = "cls3_fc_pose_xyz/BiasAdd"
            output_rot_layer_name = "cls3_fc_pose_wpqr/BiasAdd"
        elif posenet_config.base_model=="inception-v3" or posenet_config.base_model=="mobilenet-v1":
            output_pos_layer_name = "cls_fc_pose_xyz/BiasAdd"
            output_rot_layer_name = "cls_fc_pose_wpqr/BiasAdd"
        else:
            print "invalid base model : " + posenet_config.base_model
            sys.exit()
        
        input_image_dir = os.path.dirname(input_txt_file)
        input_model_dir = os.path.dirname(input_pb_file)
        input_numpy_mean_image_file = os.path.join(input_model_dir, "mean_image.npy")
        if use_fixed_input_mean_std:
                input_numpy_mean_image = None
        else:
                input_numpy_mean_image = np.load(input_numpy_mean_image_file)
        output_localize_txt_file = os.path.join(result_log_dir, "localize-poses.txt")
        output_localize_json_file = os.path.join(result_log_dir, "localize-poses.json")
        output_summary_log_file = os.path.join(result_log_dir, "summary-log.txt")
        output_hist_log_file = os.path.join(result_log_dir, "hist-log.txt")
        output_detail_log_file = os.path.join(result_log_dir, "detail-log.txt")
        
        if use_dense_sample:
	        image = tf.placeholder(tf.float32, [num_dense_sample, 224, 224, 3])
        else:
	        image = tf.placeholder(tf.float32, [1, 224, 224, 3])
	datasource, image_filenames = get_data(input_txt_file, input_image_dir)
        predicted_poses = np.zeros((len(datasource.image_filenames),7))
        groundtruth_poses = np.zeros((len(datasource.image_filenames),7))
	results = np.zeros((len(datasource.image_filenames),2))
        
	# Set GPU options
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        
        time_array = np.array([])

        # Load model
        graph = load_graph(input_pb_file)
        input_image_name = "import/" + input_image_layer_name
        output_pos_name = "import/" + output_pos_layer_name
        output_rot_name = "import/" + output_rot_layer_name
        input_operation = graph.get_operation_by_name(input_image_name)
        output_pos_operation = graph.get_operation_by_name(output_pos_name)
        output_rot_operation = graph.get_operation_by_name(output_rot_name)
        
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as sess:
		for i in range(len(datasource.image_filenames)):
                        if use_dense_sample:
                                np_images = preprocess(datasource.image_filenames[i], input_numpy_mean_image, use_fixed_input_mean_std, num_dense_sample, True)
			else:
                                np_images = preprocess(datasource.image_filenames[i], input_numpy_mean_image, use_fixed_input_mean_std, 1, False)
       		        feed = {input_operation.outputs[0]: np_images}
                        
			pose_q= np.asarray(datasource.poses[i][3:7])
			pose_x= np.asarray(datasource.poses[i][0:3])
                        start_time = time.time()
			sample_predicted_x, sample_predicted_q = sess.run([output_pos_operation.outputs[0],
                                                                           output_rot_operation.outputs[0]], feed_dict=feed)
                        elapsed_time = time.time() - start_time
                        time_array = np.append(time_array, elapsed_time)
                        
			pose_q = np.squeeze(pose_q)
			pose_x = np.squeeze(pose_x)
                        if use_dense_sample:
			        predicted_q = np.mean(sample_predicted_q, axis=0)
			        predicted_x = np.mean(sample_predicted_x, axis=0)
                        else:
			        predicted_q = np.squeeze(sample_predicted_q)
			        predicted_x = np.squeeze(sample_predicted_x)
                        
                        predicted_poses[i,0] = predicted_x[0]
                        predicted_poses[i,1] = predicted_x[1]
                        predicted_poses[i,2] = predicted_x[2]
                        predicted_poses[i,3] = predicted_q[0]
                        predicted_poses[i,4] = predicted_q[1]
                        predicted_poses[i,5] = predicted_q[2]
                        predicted_poses[i,6] = predicted_q[3]
                        
                        groundtruth_poses[i,0] = pose_x[0]
                        groundtruth_poses[i,1] = pose_x[1]
                        groundtruth_poses[i,2] = pose_x[2]
                        groundtruth_poses[i,3] = pose_q[0]
                        groundtruth_poses[i,4] = pose_q[1]
                        groundtruth_poses[i,5] = pose_q[2]
                        groundtruth_poses[i,6] = pose_q[3]
                        
		        # calcurate error
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

        # write localize poses
        with open(output_localize_txt_file, "w") as fw:
                fw.write("Localization Data V1\n")
                fw.write("ImageFile, Camera Position [X Y Z W P Q R]\n")
                fw.write("\n")
                for idx, filename in enumerate(image_filenames):
                        fw.write(os.path.basename(filename) + " " + ' '.join(['%f' % p for p in predicted_poses[idx,:]]) + "\n")
        locJsonObj = {}
        locJsonObj["locGlobal"] = []
        for idx, pose in enumerate(predicted_poses):
                Rh = transformations.quaternion_matrix([pose[3], pose[4], pose[5], pose[6]])
                groundtruth_pose = groundtruth_poses[idx]
                groundtruth_Rh = transformations.quaternion_matrix([groundtruth_pose[3], groundtruth_pose[4], groundtruth_pose[5], groundtruth_pose[6]])
                
                jsonLoc = {}
                jsonLoc["filename"] = os.path.basename(image_filenames[idx])
                jsonLoc["t"] = [pose[0], pose[1], pose[2]]
                jsonLoc["R"] = Rh[0:3,0:3].tolist()
                jsonLoc["groundtruth"] = [groundtruth_pose[0], groundtruth_pose[1], groundtruth_pose[2]]
                jsonLoc["groundtruthR"] = groundtruth_Rh[0:3,0:3].tolist()
                locJsonObj["locGlobal"].append(jsonLoc)
        with open(output_localize_json_file, "w") as fw:
                json.dump(locJsonObj, fw)
        
        # write histgram results
        bin_edge = [0.01*float(x) for x in range(0,1001)]
        dist_errors = results[:,0]
        dist_hist, dist_hist_bins = np.histogram(dist_errors, bins=bin_edge)
        dist_hist_cum_ratio = np.cumsum(dist_hist) / float(len(datasource.image_filenames))
        print "Histogram of error: " + str(dist_hist)
        print "Cumulative ratio of error: " + str(dist_hist_cum_ratio)
        print "Total loc err larger than " + str(np.max(bin_edge)) + " meters: " + str(len(datasource.image_filenames)-np.sum(dist_hist))
        
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
                fw.write("Number of test image = " + str(len(datasource.image_filenames)) + "\n")
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
                fw.write("Cumulative ratio: " + str(np.around(np.cumsum(dist_hist,dtype=float)/len(datasource.image_filenames),2)) + "\n")
                fw.write("Total loc err larger than " + str(np.max(bin_edge)) + " meters: " + str(len(datasource.image_filenames)-np.sum(dist_hist)) + "\n")
                fw.write("\n")
                fw.write("Mean time = " + str(np.average(time_array)) + "\n")
                fw.write("StdDev time = " + str(np.std(time_array)) + "\n")
                fw.write("Median time = " + str(np.median(time_array)) + "\n")
        # write error histgram
        np.savetxt(output_hist_log_file, zip(dist_hist_bins, dist_hist_cum_ratio), delimiter=',')
        # write error histgram
        np.savetxt(output_hist_log_file, zip(dist_hist_bins, dist_hist_cum_ratio), delimiter=',')

        # write detail results
        with open(output_detail_log_file, "w") as fw:
                for idx, filename in enumerate(image_filenames):
                        fw.write(os.path.basename(filename) + "," + str(results[idx,0]) + "," + str(results[idx,1]) + "\n")
        
if __name__ == '__main__':
	main()

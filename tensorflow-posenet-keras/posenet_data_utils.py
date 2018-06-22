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

import os
import math
import random
import numpy as np
import posenet_geom_utils as posenet_geom_utils
import posenet_image_utils as posenet_image_utils
import posenet_beacon_utils as posenet_beacon_utils
from posenet_config import posenet_config

class image_datasource(object):
    def __init__(self, images_index, poses_index, images_data, poses_data):
	self.images_index = images_index
	self.poses_index = poses_index
	self.images_data = images_data
	self.poses_data = poses_data

def get_image_data(input_txt_file, input_image_dir, use_fixed_input_mean_std, image_size=224):
    poses_index = []
    images_index = []
    poses_data = []
    images_data = []
    
    with open(input_txt_file) as f:
        # skip header
	next(f)
	next(f)
	next(f)
	for line in f:
	    fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
            quat = posenet_geom_utils.preprocess_quaternion([float(p3),float(p4),float(p5),float(p6)])
            
	    p0 = float(p0)
	    p1 = float(p1)
	    p2 = float(p2)
	    p3 = quat[0]
	    p4 = quat[1]
	    p5 = quat[2]
	    p6 = quat[3]
	    poses_data.append((p0,p1,p2,p3,p4,p5,p6))
            poses_index.append(len(poses_data)-1)
            
	    images_data.append(os.path.join(input_image_dir, fname))
            images_index.append(len(images_data)-1)
            
    images_data, mean_image = posenet_image_utils.preprocess(images_data, use_fixed_input_mean_std, image_size=image_size)
    return image_datasource(images_index, poses_index, images_data, poses_data), mean_image

def gen_image_data(source):
	while True:
		indices = range(len(source.images_index))
		random.shuffle(indices)
		for i in indices:
			image = source.images_data[source.images_index[i]]
			pose_x = source.poses_data[source.poses_index[i]][0:3]
			pose_q = source.poses_data[source.poses_index[i]][3:7]
			yield image, pose_x, pose_q

def gen_image_data_batch(source, output_auxiliary=True, batch_size=posenet_config.batch_size):
    data_gen = gen_image_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        if output_auxiliary:
            yield np.array(image_batch), [np.array(pose_x_batch), np.array(pose_q_batch), np.array(pose_x_batch), np.array(pose_q_batch), np.array(pose_x_batch), np.array(pose_q_batch)]
        else:
            yield np.array(image_batch), [np.array(pose_x_batch), np.array(pose_q_batch)]
        
def gen_image_lstm_data(source):
    while True:
	indices = range(posenet_config.lstm_step_size-1, len(source.images_index))
	random.shuffle(indices)
	for i in indices:
            found_gap = False
            image_seq = []
            pose_x_seq = []
            pose_q_seq = []
            
            for j in range(i-posenet_config.lstm_step_size+1, i+1):
		image = source.images_data[source.images_index[j]]
		pose_x = source.poses_data[source.poses_index[j]][0:3]
		pose_q = source.poses_data[source.poses_index[j]][3:7]
                image_seq.append(image)
                pose_x_seq.append(pose_x)
                pose_q_seq.append(pose_q)
                
                if j>0 :
                    pose1 = np.array(source.poses_data[source.poses_index[j]][0:3])
                    pose2 = np.array(source.poses_data[source.poses_index[j-1]][0:3])
                    if np.linalg.norm(pose1 - pose2) > posenet_config.detect_gap_threshold:
                        found_gap = True
                        
            if found_gap:
                continue
            
	    yield np.asarray(image_seq), np.asarray(pose_x_seq), np.asarray(pose_q_seq)

def gen_image_lstm_data_batch(source, batch_size=posenet_config.batch_size):
    data_gen = gen_image_lstm_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), [np.array(pose_x_batch), np.array(pose_q_batch)]

class beacon_datasource(object):
    def __init__(self, beacons_index, poses_index, beacons_data, poses_data):
        self.beacons_index = beacons_index
	self.poses_index = poses_index
        self.beacons_data = beacons_data
	self.poses_data = poses_data

def get_beacon_data(input_txt_file, beaconmap, beacon_num, use_fixed_input_mean_std, use_augmentation_beacon):
    poses_index = []
    beacons_index = []
    poses_data = []
    beacons_data = []
        
    with open(input_txt_file) as f:
        # skip header
	next(f)
	next(f)
	next(f)
	for line in f:
	    beaconstr, p0,p1,p2,p3,p4,p5,p6 = line.split()
            quat = posenet_geom_utils.preprocess_quaternion([float(p3),float(p4),float(p5),float(p6)])
            
	    p0 = float(p0)
	    p1 = float(p1)
	    p2 = float(p2)
	    p3 = quat[0]
	    p4 = quat[1]
	    p5 = quat[2]
	    p6 = quat[3]
	    poses_data.append((p0,p1,p2,p3,p4,p5,p6))
            poses_index.append(len(poses_data)-1)
            
            beacon = posenet_beacon_utils.parse_beacon_string(beaconstr, beaconmap)
	    beacons_data.append(beacon)
            beacons_index.append(len(beacons_data)-1)
            
            if use_augmentation_beacon:
                nonzero_idxs = np.nonzero(beacon)[0]
                num_augment_beacon = int(math.ceil(len(nonzero_idxs) * posenet_config.ratio_beacon_augmentation))
                for idx in range(0,posenet_config.num_beacon_augmentation):
                    poses_index.append(len(poses_data)-1)
                    
                    augment_beacon = np.copy(beacon)
                    for idx_augment in range(0,num_augment_beacon):
                        random_idx = int(np.random.uniform(0.0, 1.0) * (len(nonzero_idxs)-1))
                        augment_beacon_idx = nonzero_idxs[random_idx]
                        augment_beacon[augment_beacon_idx][0][0] = math.floor(augment_beacon[augment_beacon_idx][0][0] * np.random.uniform(0.0, 1.0))
		    beacons_data.append(augment_beacon)
                    beacons_index.append(len(beacons_data)-1)
                    
    beacons_data, mean_beacon = posenet_beacon_utils.preprocess_beacons(beacons_data, beacon_num, use_fixed_input_mean_std)
    return beacon_datasource(beacons_index, poses_index, beacons_data, poses_data), mean_beacon

def gen_beacon_data(source):
    while True:
	indices = range(len(source.beacons_index))
	random.shuffle(indices)
	for i in indices:
	    beacon = source.beacons_data[source.beacons_index[i]]
	    pose_x = source.poses_data[source.poses_index[i]][0:3]
	    pose_q = source.poses_data[source.poses_index[i]][3:7]
	    yield beacon, pose_x, pose_q

def gen_beacon_data_batch(source, batch_size=posenet_config.batch_size):
    data_gen = gen_beacon_data(source)
    while True:
        beacon_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            beacon, pose_x, pose_q = next(data_gen)
            beacon_batch.append(beacon)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(beacon_batch), [np.array(pose_x_batch), np.array(pose_q_batch), np.array(pose_x_batch), np.array(pose_q_batch), np.array(pose_x_batch), np.array(pose_q_batch)]

def gen_beacon_lstm_data(source):
    while True:
        indices = range(posenet_config.lstm_step_size-1, len(source.beacons_index))
	random.shuffle(indices)
	for i in indices:
            found_gap = False
            beacon_seq = []
            pose_x_seq = []
            pose_q_seq = []
            
            for j in range(i-posenet_config.lstm_step_size+1, i+1):
		beacon = source.beacons_data[source.beacons_index[j]]
		pose_x = source.poses_data[source.poses_index[j]][0:3]
		pose_q = source.poses_data[source.poses_index[j]][3:7]
                beacon_seq.append(beacon)
                pose_x_seq.append(pose_x)
                pose_q_seq.append(pose_q)
                
                if j>0 :
                    pose1 = np.array(source.poses_data[source.poses_index[j]][0:3])
                    pose2 = np.array(source.poses_data[source.poses_index[j-1]][0:3])
                    if np.linalg.norm(pose1 - pose2) > posenet_config.detect_gap_threshold:
                        found_gap = True
                        
            if found_gap:
                continue
                        
	    yield np.asarray(beacon_seq), np.asarray(pose_x_seq), np.asarray(pose_q_seq)

def gen_beacon_lstm_data_batch(source, batch_size=posenet_config.batch_size):
    data_gen = gen_beacon_lstm_data(source)
    while True:
        beacon_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            beacon, pose_x, pose_q = next(data_gen)
            beacon_batch.append(beacon)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(beacon_batch), [np.array(pose_x_batch), np.array(pose_q_batch)]

class image_beacon_datasource(object):
    def __init__(self, images_index, beacons_index, poses_index, images_data, beacons_data, poses_data):
	self.images_index = images_index
        self.beacons_index = beacons_index
	self.poses_index = poses_index
	self.images_data = images_data
        self.beacons_data = beacons_data
	self.poses_data = poses_data
    
def get_image_beacon_data(input_txt_file, input_image_dir, beaconmap, beacon_num, use_fixed_input_mean_std, use_augmentation_beacon, image_size=224):
    poses_index = []
    images_index = []
    beacons_index = []
    poses_data = []
    images_data = []
    beacons_data = []
    
    with open(input_txt_file) as f:
        # skip header
	next(f)
	next(f)
	next(f)
	for line in f:
	    fname, beaconstr, p0,p1,p2,p3,p4,p5,p6 = line.split()
            quat = posenet_geom_utils.preprocess_quaternion([float(p3),float(p4),float(p5),float(p6)])
            
	    p0 = float(p0)
	    p1 = float(p1)
	    p2 = float(p2)
	    p3 = quat[0]
	    p4 = quat[1]
	    p5 = quat[2]
	    p6 = quat[3]
	    poses_data.append((p0,p1,p2,p3,p4,p5,p6))
            poses_index.append(len(poses_data)-1)
            
	    images_data.append(os.path.join(input_image_dir, fname))
            images_index.append(len(images_data)-1)
            
            beacon = posenet_beacon_utils.parse_beacon_string(beaconstr, beaconmap)
	    beacons_data.append(beacon)
            beacons_index.append(len(beacons_data)-1)
            
            if use_augmentation_beacon:
                nonzero_idxs = np.nonzero(beacon)[0]
                num_augment_beacon = int(math.ceil(len(nonzero_idxs) * posenet_config.ratio_beacon_augmentation))
                for idx in range(0,posenet_config.num_beacon_augmentation):
                    poses_index.append(len(poses_data)-1)
                    images_index.append(len(images_data)-1)
                    
                    augment_beacon = np.copy(beacon)
                    for idx_augment in range(0,num_augment_beacon):
                        random_idx = int(np.random.uniform(0.0, 1.0) * (len(nonzero_idxs)-1))
                        augment_beacon_idx = nonzero_idxs[random_idx]
                        augment_beacon[augment_beacon_idx][0][0] = math.floor(augment_beacon[augment_beacon_idx][0][0] * np.random.uniform(0.0, 1.0))
		    beacons_data.append(augment_beacon)
                    beacons_index.append(len(beacons_data)-1)
        
    images_data, mean_image = posenet_image_utils.preprocess(images_data, use_fixed_input_mean_std, image_size=image_size)
    beacons_data, mean_beacon = posenet_beacon_utils.preprocess_beacons(beacons_data, beacon_num, use_fixed_input_mean_std)
    return image_beacon_datasource(images_index, beacons_index, poses_index, images_data, beacons_data, poses_data), mean_image, mean_beacon

def gen_image_beacon_data(source):
    while True:
	indices = range(len(source.images_index))
	random.shuffle(indices)
	for i in indices:
	    image = source.images_data[source.images_index[i]]
	    beacon = source.beacons_data[source.beacons_index[i]]
	    pose_x = source.poses_data[source.poses_index[i]][0:3]
	    pose_q = source.poses_data[source.poses_index[i]][3:7]
	    yield image, beacon, pose_x, pose_q

def gen_image_beacon_data_batch(source, output_auxiliary=True, batch_size=posenet_config.batch_size):
    data_gen = gen_image_beacon_data(source)
    while True:
        image_batch = []
        beacon_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, beacon, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            beacon_batch.append(beacon)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        if output_auxiliary:
            yield [np.array(image_batch), np.array(beacon_batch)], [np.array(pose_x_batch), np.array(pose_q_batch), np.array(pose_x_batch), np.array(pose_q_batch), np.array(pose_x_batch), np.array(pose_q_batch)]
        else:
            yield [np.array(image_batch), np.array(beacon_batch)], [np.array(pose_x_batch), np.array(pose_q_batch)]
    
def gen_image_beacon_lstm_data(source):
    while True:
	indices = range(posenet_config.lstm_step_size-1, len(source.images_index))
	random.shuffle(indices)
	for i in indices:
            found_gap = False
            image_seq = []
            beacon_seq = []
            pose_x_seq = []
            pose_q_seq = []
            
            for j in range(i-posenet_config.lstm_step_size+1, i+1):
		image = source.images_data[source.images_index[j]]
		beacon = source.beacons_data[source.beacons_index[j]]
		pose_x = source.poses_data[source.poses_index[j]][0:3]
		pose_q = source.poses_data[source.poses_index[j]][3:7]
                image_seq.append(image)
                beacon_seq.append(beacon)
                pose_x_seq.append(pose_x)
                pose_q_seq.append(pose_q)
                
                if j>0 :
                    pose1 = np.array(source.poses_data[source.poses_index[j]][0:3])
                    pose2 = np.array(source.poses_data[source.poses_index[j-1]][0:3])
                    if np.linalg.norm(pose1 - pose2) > posenet_config.detect_gap_threshold:
                        found_gap = True
            
            if found_gap:
                continue
            
	    yield np.asarray(image_seq), np.asarray(beacon_seq), np.asarray(pose_x_seq), np.asarray(pose_q_seq)

def gen_image_beacon_lstm_data_batch(source, batch_size=posenet_config.batch_size):
    data_gen = gen_image_beacon_lstm_data(source)
    while True:
        image_batch = []
        beacon_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, beacon, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            beacon_batch.append(beacon)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield [np.array(image_batch), np.array(beacon_batch)], [np.array(pose_x_batch), np.array(pose_q_batch)]

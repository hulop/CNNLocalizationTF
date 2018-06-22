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

import numpy as np
import random
import cv2
from tqdm import tqdm

fixed_image_mean = 128.0
fixed_image_std = 1.0

def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
        
	height_offset = (height - output_side_length) / 2
	width_offset = (width - output_side_length) / 2
        
	cropped_img = img[height_offset:height_offset + output_side_length,
			  width_offset:width_offset + output_side_length]
	return cropped_img

def randomCrop(img, output_side_length):
        RANDOM_SHIFT = 10

	height, width, depth = img.shape
        
	height_offset = (height - output_side_length) / 2
	width_offset = (width - output_side_length) / 2
        
        height_offset += random.randint(-RANDOM_SHIFT,RANDOM_SHIFT)
        width_offset += random.randint(-RANDOM_SHIFT,RANDOM_SHIFT)
        if height_offset<0:
                height_offset = 0
        if height_offset>=height-output_side_length:
                height_offset = height-output_side_length
        if width_offset<0:
                width_offset = 0
        if width_offset>=width-output_side_length:
                width_offset = width-output_side_length
        
	cropped_img = img[height_offset:height_offset + output_side_length,
						width_offset:width_offset + output_side_length]
	return cropped_img

def preprocess(images, use_fixed_input_mean_std, image_size=224):
	images_out = []

        # resize and crop
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		#X = cv2.resize(X, (455, 256))
                X = cv2.resize(X, (int(image_size*2.031), int(image_size*1.143)))
		X = centeredCrop(X, image_size)
		images_cropped.append(X)

        # calculate mean image
	N = 0
        mean = None
        if not use_fixed_input_mean_std:
	        mean = np.zeros((1, 3, image_size, image_size))
	        for X in tqdm(images_cropped):
		        mean[0][0] += X[:,:,0]
		        mean[0][1] += X[:,:,1]
		        mean[0][2] += X[:,:,2]
		        N += 1
	        mean[0] /= N

	# subtract mean image
	for X in tqdm(images_cropped):
		X = np.transpose(X,(2,0,1))
                if use_fixed_input_mean_std:
                        X = X - fixed_image_mean
                        X = X / fixed_image_std
                else:
		        X = X - mean
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		images_out.append(X)
	return images_out, mean

def preprocess_test_image(image, mean_image, use_fixed_input_mean_std, num_sample, random_crop, image_size=224):
	images_out = []

        # resize and crop        
	images_cropped = []
	X = cv2.imread(image)
	#X = cv2.resize(X, (455, 256))
        X = cv2.resize(X, (int(image_size*2.031), int(image_size*1.143)))        
        for _ in range(num_sample):
                if random_crop:
	                X = randomCrop(X, image_size)
                else:
		        X = centeredCrop(X, image_size)
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

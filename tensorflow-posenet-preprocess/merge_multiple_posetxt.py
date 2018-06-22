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
import csv
import os
import sys
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='append', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='File path of input text files in Visual Landmark Dataset format.')
    parser.add_argument('-o', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='File path where output text file in Visual Landmark Dataset format will be saved.')
    args = parser.parse_args()
    input_pose_txt_files = args.i
    output_pose_txt = args.o
    print "input_pose_txt_files : " + str(input_pose_txt_files)
    print "output_pose_txt : " + str(output_pose_txt)
    
    pose_txt_lines = []
    for input_pose_txt in input_pose_txt_files:
	with open(input_pose_txt) as f:
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
                    pose_txt_lines.append(line)
    
    with open(output_pose_txt, 'w') as f:
        f.write("Localization Data V1\n")
        f.write("ImageFile, Camera Position [X Y Z W P Q R]\n")
        f.write("\n")

        for line in pose_txt_lines:
            f.write(line)

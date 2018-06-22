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
    parser.add_argument('input_image_csv', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='File path of input csv file sampled by LiDAR.')
    parser.add_argument('input_image_dir', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Directory path where input images are saved.')
    parser.add_argument('output_pose_txt', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='File path where output text file in Visual Landmark Dataset format will be saved.')
    args = parser.parse_args()
    input_image_csv_file = args.input_image_csv
    input_image_dir = args.input_image_dir
    output_pose_txt = args.output_pose_txt
        
    photoTimeFileDict = {}
    lidarTimePoseDict = {}
    with open(input_image_csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[1]=="Misc" and row[2]=="Photo":
                print "image file : " + row[3]
                photoTimeFileDict[int(row[0])] = row[3]
            elif row[1]=="LiDAR" and row[2]=="7":
                print "LiDAR pose : " + row[3] + ", " + row[4] + ", " + row[5] \
                 + ", " + row[6]  + ", " + row[7]  + ", " + row[8]  + ", " + row[9]
                lidarTimePoseDict[int(row[0])] = [float(row[3]), float(row[4]), float(row[5]),
                                                  float(row[6]), float(row[7]), float(row[8]), float(row[9])]
    
    relative_image_dir_path = os.path.relpath(input_image_dir, os.path.dirname(output_pose_txt))    
    
    with open(output_pose_txt, 'w') as f:
        f.write("Localization Data V1\n")
        f.write("ImageFile, Camera Position [X Y Z W P Q R]\n")
        f.write("\n")

        for key in sorted(photoTimeFileDict.keys()):
            lidarInd = np.argmin(abs(key-np.array(lidarTimePoseDict.keys())))                        
            f.write(os.path.join(relative_image_dir_path, photoTimeFileDict[key]) + " " + " ".join(str(x) for x in lidarTimePoseDict[lidarTimePoseDict.keys()[lidarInd]]) + "\n")        
    

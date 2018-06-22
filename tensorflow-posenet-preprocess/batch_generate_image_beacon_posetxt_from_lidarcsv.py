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

def find_sfm_projects(input_dir):
    sfm_image_dirs = []
    sfm_beacon_files = []
    
    for dirpath, dirnames, files in os.walk(input_dir):
        bFoundBeaconSetting = False
        for filename in files:
            if filename=="listbeacon.txt":
                bFoundBeaconSetting = True
        bSfMProject = True
        if bFoundBeaconSetting:
            for dirname in dirnames:
                child_dirpath = os.path.join(dirpath, dirname)
                child_dirfiles = os.listdir(child_dirpath)
                if "csv" in child_dirfiles and "inputImg" in child_dirfiles:
                    child_csvfiles = os.listdir(os.path.join(child_dirpath, "csv"))
                    if len(child_csvfiles)!=1:
                        print("invalid SfM project created by LiDAR, each SfM project should have one beacon setting CSV")
                        sys.exit()
                    sfm_image_dirs.append(os.path.join(child_dirpath, "inputImg"))
                    sfm_beacon_files.append(os.path.join(child_dirpath, "csv", child_csvfiles[0]))
    
    return sfm_image_dirs, sfm_beacon_files

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_projects_dir', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='Directory path where SfM projects directories are located.')
    parser.add_argument('output_txt', action='store', nargs=None, const=None, \
                        default=None, type=str, choices=None, metavar=None, \
                        help='File path where Visual Landmark Dataset format will be saved.')
    args = parser.parse_args()
    input_projects_dir = args.input_projects_dir
    output_txt = args.output_txt
    output_dir = os.path.dirname(output_txt)
    
    sfm_image_dirs, sfm_beacon_files = find_sfm_projects(input_projects_dir)
    if len(sfm_image_dirs)==0 or len(sfm_beacon_files)==0 or len(sfm_image_dirs)!=len(sfm_beacon_files):
        print("invalid input project directory")
    
    sfm_output_files = []
    for i in range(len(sfm_image_dirs)):
        print("image directory : " + sfm_image_dirs[i])
        print("beacon file : " + sfm_beacon_files[i])
        data_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sfm_image_dirs[i]))))))
        phone_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sfm_image_dirs[i])))))
        print("data id : " + data_id)
        print("phone id : " + phone_id)
        sfm_output_txt = os.path.join(output_dir, data_id + "-" + phone_id + "-" + str(i) + ".txt")
        command = "python generate_image_beacon_posetxt_from_lidarcsv.py " + sfm_beacon_files[i] + " " + sfm_image_dirs[i] + " " + sfm_output_txt
        print("execute : " + command)
        os.system(command)
        sfm_output_files.append(sfm_output_txt)

    command = "python merge_multiple_posetxt.py"
    for sfm_output_file in sfm_output_files:
        command += " -i " + sfm_output_file
    command += " -o " + output_txt
    print("execute : " + command)
    os.system(command)

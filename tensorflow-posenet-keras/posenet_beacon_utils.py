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
import hulo_ibeacon.IBeaconUtils as IBeaconUtils
from tqdm import tqdm

fixed_beacon_mean = 0.0
fixed_beacon_std = 1.0

# parse beacon signal string and create numpy tensor
def parse_beacon_string(beaconstr,beaconmap):
    splitline = beaconstr.split(",")
    
    timestamp = int(splitline[0])
    
    # parse by type
    if splitline[1] != "Beacon":
        print "invalid beacon signal"
        sys.exit()

    rssi = IBeaconUtils.parseBeaconList(splitline[6:],beaconmap)
    beacon = rssi.astype(np.float32).reshape(len(rssi),1,1)
    return beacon

def preprocess_beacons(beacons, num_beacons, use_fixed_input_mean_std):
    beacons_out = []
    
    # calculate mean beacon
    N = 0
    mean = None
    if not use_fixed_input_mean_std:
	mean = np.zeros((num_beacons, 1, 1))
	for X in tqdm(beacons):
	    mean += X[:,:,:]
	    N += 1
	mean /= N
    
    # subtract mean beacon
    for X in tqdm(beacons):
        if use_fixed_input_mean_std:
            X = X - fixed_beacon_mean
            X = X / fixed_beacon_std
        else:
	    X = X - mean
	beacons_out.append(X)
    return beacons_out, mean

def preprocess_test_beacons(beacons, mean_beacon, use_fixed_input_mean_std):
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

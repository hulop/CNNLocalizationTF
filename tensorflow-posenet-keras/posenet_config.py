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

class posenet_config:
    ###############################
    # Basic settings
    ###############################
    base_model = "inception-v1"
    batch_size = 64
    epochs = 800

    ###############################
    # PoseNet (ICCV 2015) settings
    ###############################
    loss_beta = 500
    
    ###############################
    # LSTM settings
    ###############################
    lstm_batch_size = 16
    lstm_step_size = 3
    # threshold to detect gap for LSTM training data
    detect_gap_threshold = 3.0
    
    ###############################
    # Data augmentation settings
    ###############################
    # number of data augmentation for each beacon
    num_beacon_augmentation = 5
    # ratio of observed beacons that will be augmented
    ratio_beacon_augmentation = 0.1
    

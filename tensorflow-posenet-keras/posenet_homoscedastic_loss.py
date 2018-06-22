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
from keras import backend as K

# https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras
def euc_loss1x_s(s_x):
    def euc_loss1x(y_true, y_pred):
        lx = K.sum(K.abs(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True)
        return (lx * K.exp(-1.0 * s_x) + s_x) * 0.3
    return euc_loss1x
    
def euc_loss1q_s(s_q):
    def euc_loss1q(y_true, y_pred):
        lq = K.sum(K.abs(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True)
        return (lq * K.exp(-1.0 * s_q) + s_q) * 0.3
    return euc_loss1q

def euc_loss2x_s(s_x):
    def euc_loss2x(y_true, y_pred):
        lx = K.sum(K.abs(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True)
        return (lx * K.exp(-1.0 * s_x) + s_x) * 0.3
    return euc_loss2x

def euc_loss2q_s(s_q):
    def euc_loss2q(y_true, y_pred):
        lq = K.sum(K.abs(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True)
        return (lq * K.exp(-1.0 * s_q) + s_q) * 0.3
    return euc_loss2q

def euc_loss3x_s(s_x):
    def euc_loss3x(y_true, y_pred):
        lx = K.sum(K.abs(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True)
        return (lx * K.exp(-1.0 * s_x) + s_x)
    return euc_loss3x

def euc_loss3q_s(s_q):
    def euc_loss3q(y_true, y_pred):
        lq = K.sum(K.abs(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True)
        return (lq * K.exp(-1.0 * s_q) + s_q)
    return euc_loss3q

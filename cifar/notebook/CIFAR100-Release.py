import gzip
import os
import sys
import time

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../tuner_utils')
from robust_region_v2 import *
sys.path.append('../scripts')
from resnet_utils import *

sys.path.append('../scripts')
import resnet_model
from resnet_main import train
import cifar_input


# define single step function
def train_single_step_frame(sess, model_train, lr_val, mom_val, clip_norm_base):      
    output_results = sess.run( [model_train.cost] + model_train.grads +[model_train.train_op, ] )    
    loss = output_results[0]
    grads = output_results[1:(1 + len(model_train.grads) ) ]
    return grads, loss


# set up path and other parameters
NUM_CLASSES = 100
TRAIN_DATA_PATH = '../../data/cifar-100-binary/train.bin'
TEST_DATA_PATH = '../../data/cifar-100-binary/test.bin'
MODE = 'train'
LOG_ROOT='../results/resnet_model'
DATASET='cifar100'
DEV = '/gpu:0'
tf.reset_default_graph()


# construct train model and session
batch_size_train = 128
batch_size_test = 100
tf.set_random_seed(1)
hps_train = resnet_model.HParams(batch_size=batch_size_train,
                         num_classes=NUM_CLASSES,
                         min_lrn_rate=0.0001,
                         lrn_rate=0.1,
                         mom=0.9,
                         clip_norm_base=10.0,
                         num_residual_units=5,
                         use_bottleneck=True,
                         weight_decay_rate=0.0002,
                         relu_leakiness=0.1,
                         optimizer='YF', model_scope='train')
hps_eval = resnet_model.HParams(batch_size=batch_size_test,
                         num_classes=NUM_CLASSES,
                         min_lrn_rate=0.0001,
                         lrn_rate=0.1,
                         mom=0.9,
                         clip_norm_base=10.0,
                         num_residual_units=5,
                         use_bottleneck=True,
                         weight_decay_rate=0.0002,
                         relu_leakiness=0.1,
                         optimizer='YF', model_scope='train')

gpu_mem_portion=0.5
model_train, model_eval, init_op, mon_sess = setup(hps_train, hps_eval, gpu_mem_portion, DEV, DATASET, TRAIN_DATA_PATH, TEST_DATA_PATH)


# In[10]:

general_log_dir = "../results"
num_step = 70001
display_interval=500
test_interval=1000 # no test
clip_norm_base = 100.0
# the following a few params are all dummy and placeholder
train_batch_size = 128
grad_beta = 0.9
dist_beta = 0.9
curv_beta = 0.999
param_beta = 0.999
sliding_win_width=10


# In[12]:

lr_val = 1.0
mom_val = 0.0
do_pred = True
log_dir = general_log_dir + "/test-release-cifar100"
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

tf.set_random_seed(1)
mon_sess.run(init_op)

train_single_step = lambda lr_val, mom_val, clip_norm_base: train_single_step_frame(mon_sess, model_train, lr_val, mom_val, clip_norm_base)

loss_list, g_norm_list = train_rave_noisy_acc(mon_sess, model_eval, num_step, lr_val, mom_val, 
                     clip_norm_base, do_pred, curv_beta, param_beta, train_single_step, sliding_win_width, display_interval, test_interval, log_dir=log_dir)
np.savetxt(log_dir + "/loss_full.txt", np.array(loss_list) )
np.savetxt(log_dir + "/g_norm_full.txt", np.array(g_norm_list) )

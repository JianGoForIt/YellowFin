
# coding: utf-8

# ### load generic libraries

# In[1]:

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


# In[2]:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


# In[3]:

sys.path.append('../tuner_utils')
from robust_region_v2 import *
sys.path.append('../scripts')
from resnet_utils import *


# ### load model specific libraries

# In[4]:

sys.path.append('../scripts')
# from resnet import resnet_model
# from resnet.resnet_main import train
# from resnet import cifar_input
import resnet_model
from resnet_main import train
import cifar_input


# ### define function for constructing model

# In[5]:

def get_model(hps, dataset, train_data_path):
    images, labels = cifar_input.build_input(
      dataset, train_data_path, hps.batch_size, 'train')
    model = resnet_model.ResNet(hps, images, labels, 'train')
    model.build_graph()
    return model


# ### define function to get a session for training

# In[6]:

# def GetTrainingSession(model_train):
#     mon_sess = tf.train.MonitoredTrainingSession(
#       config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#     return mon_sess


# ### define single step function

# In[7]:

def train_single_step_frame(sess, model_train, lr_val, mom_val, clip_norm_base):
    feed_dict={model_train.lrn_rate: lr_val,
               model_train.mom: mom_val,
               model_train.clip_norm: clip_norm_base / lr_val}  # Sets learning rate        
    output_results = sess.run( [model_train.cost, model_train.global_norm] + model_train.grads +[model_train.train_op, ], feed_dict=feed_dict)    
    loss = output_results[0]
    grad_norm = output_results[1]
    grads = output_results[2:(2 + len(model_train.grads) ) ]
    return grads, grad_norm, loss


# ### set up path and other parameters

# In[8]:

NUM_CLASSES = 100
TRAIN_DATA_PATH = '../../../cifar-100-binary/train.bin'
TEST_DATA_PATH = '../../../cifar-100-binary/test.bin'
MODE = 'train'
LOG_ROOT='../results/resnet_model'
DATASET='cifar100'
DEV = '/gpu:0'
tf.reset_default_graph()


# ### construct train model and session

# In[9]:

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
                         optimizer='mom',
                         model_scope='train')
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
                         optimizer='mom',
                         model_scope='train')


# with tf.variable_scope("train"), tf.device(DEV):
#     model_train = get_model(hps_train, DATASET, TRAIN_DATA_PATH)

#     init_op = tf.global_variables_initializer()
#     mon_sess = GetTrainingSession(model_train)
gpu_mem_portion=0.4
model_train, model_eval, init_op, mon_sess = setup(hps_train, hps_eval, gpu_mem_portion, DEV, DATASET, TRAIN_DATA_PATH, TEST_DATA_PATH)


# In[10]:

general_log_dir = "../results"


# ### exp smooth 0.99

# In[11]:

num_step = 40001
display_interval=5000
test_interval=50000 # no test
clip_norm_base = 100.0
# the following a few params are all dummy and placeholder
train_batch_size = 128
grad_beta = 0.9
dist_beta = 0.9
curv_beta = 0.9
param_beta = 0.999
sliding_win_width=10


# In[12]:

lr_val = 0.1
mom_val = 0.9
do_pred = True
log_dir = general_log_dir + "/two_term_lr_" + str(lr_val) + "_mom_" + str(mom_val) + "_exp_0.9"
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

tf.set_random_seed(1)
mon_sess.run(init_op)

train_single_step = lambda lr_val, mom_val, clip_norm_base: train_single_step_frame(mon_sess, model_train, lr_val, mom_val, clip_norm_base)

loss_list, g_norm_list = train_rave_noisy_acc(mon_sess, model_eval, num_step, lr_val, mom_val, 
                     clip_norm_base, do_pred, curv_beta, param_beta, train_single_step, sliding_win_width, display_interval, test_interval, log_dir=log_dir)
np.savetxt(log_dir + "/loss_full.txt", np.array(loss_list) )
np.savetxt(log_dir + "/g_norm_full.txt", np.array(g_norm_list) )


# In[ ]:

#lr_val = 0.1
#mom_val = 0.9
#do_pred = True
#log_dir = general_log_dir + "/two_term_lr_" + str(lr_val) + "_mom_" + str(mom_val) + "_exp_0.9"
#if not os.path.isdir(log_dir):
#    os.mkdir(log_dir)
#
#tf.set_random_seed(1)
#mon_sess.run(init_op)
#
#train_single_step = lambda lr_val, mom_val, clip_norm_base: train_single_step_frame(mon_sess, model_train, lr_val, mom_val, clip_norm_base)
#
#loss_list, g_norm_list = train_rave_noisy_acc(mon_sess, model_eval, num_step, lr_val, mom_val, 
#                     clip_norm_base, do_pred, curv_beta, param_beta, train_single_step, sliding_win_width, display_interval, test_interval, log_dir=log_dir)
#np.savetxt(log_dir + "/loss_full.txt", np.array(loss_list) )
#np.savetxt(log_dir + "/g_norm_full.txt", np.array(g_norm_list) )
#

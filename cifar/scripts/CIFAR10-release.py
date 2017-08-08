from __future__ import print_function
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

sys.path.append('../model')
import resnet_model
from resnet_utils import *
import cifar_input


# set up path and other parameters
NUM_CLASSES = 10
TRAIN_DATA_PATH = '../../data/cifar10/data_batch*'
TEST_DATA_PATH = '../../data/cifar10/test_batch.bin'
MODE = 'train'
DATASET='cifar10'
DEV = '/gpu:0'
tf.reset_default_graph()


# construct train model and session
batch_size_train = 128
batch_size_test = 100
hps_train = resnet_model.HParams(batch_size=batch_size_train,
                                num_classes=NUM_CLASSES,
                                # note these dummy params lr, mom and clip are just for adaptation of the model implementation, it is not relevant to the optimizer
                                min_lrn_rate=0.0001,
                                lrn_rate=0.1,
                                mom=0.9,
                                clip_norm_base=10.0,
                                num_residual_units=5,
                                use_bottleneck=False,
                                weight_decay_rate=0.0002,
                                relu_leakiness=0.1,
                                optimizer='YF',
                                model_scope='train')
# specify how much memory to use on each GPU
gpu_mem_portion=0.5
n_core = 16
with tf.variable_scope("train"), tf.device(DEV):
  model_train = get_model(hps_train, DATASET, TRAIN_DATA_PATH, mode='train')
init_op = tf.global_variables_initializer()
sess = GetTrainingSession(model_train, gpu_mem_portion=gpu_mem_portion)


# run steps
general_log_dir = "../results"
if not os.path.isdir(general_log_dir):
  os.mkdir(general_log_dir)
log_dir = general_log_dir + "/test-release-cifar10-test-clip"
num_step = 40001
display_interval=2500

if not os.path.isdir(log_dir):
  os.mkdir(log_dir)

sess.run(init_op)

loss_list = []
for i in range(num_step):
  loss, _ = sess.run( [model_train.cost, model_train.train_op ] )
  loss_list.append(loss)
  if (i % display_interval == 0 or i == 50) and (i != 0):
    print("plotting for iteration ", i)
    plot_loss(loss_list, log_dir, i)
    np.savetxt(log_dir + "/loss_full.txt", np.array(loss_list) )

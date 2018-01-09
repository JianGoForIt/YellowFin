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

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', type=float, default=1.0,
                     help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--opt_method', type=str, default="YF", help='optimizer')
parser.add_argument('--log_dir', type=str, default="results/", help="log folder")
parser.add_argument('--h_max_log_smooth', action='store_true')

args = parser.parse_args()

# set up path and other parameters
NUM_CLASSES = 100
TRAIN_DATA_PATH = '../../data/cifar-100-binary/train.bin'
TEST_DATA_PATH = '../../data/cifar-100-binary/test.bin'
MODE = 'train'
#LOG_ROOT='../results/resnet_model'
DATASET='cifar100'
DEV = '/gpu:0'
tf.reset_default_graph()

# set random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)
print("using random seed", args.seed)

# construct train model and session
batch_size_train = 128
batch_size_test = 100
hps_train = resnet_model.HParams(batch_size=batch_size_train,
                                num_classes=NUM_CLASSES,
                                # note these dummy params lr, mom and clip are just for adaptation of the model implementation, it is not relevant to the optimizer
                                min_lrn_rate=0.0001,
                                lrn_rate=args.lr,
                                mom=0.9,
                                clip_norm_base=10.0,
                                num_residual_units=5,
                                use_bottleneck=True,
                                weight_decay_rate=0.0002,
                                relu_leakiness=0.1,
                                optimizer=args.opt_method,
                                model_scope='train',
                                h_max_log_smooth=args.h_max_log_smooth)
hps_eval = resnet_model.HParams(batch_size=batch_size_test,
                               num_classes=NUM_CLASSES,
                               # note these dummy params lr, mom and clip are just for adaptation of the model implementation, it is not relevant to the optimizer
                               min_lrn_rate=0.0001,
                               lrn_rate=args.lr,
                               mom=0.9,
                               clip_norm_base=10.0,
                               num_residual_units=5,
                               use_bottleneck=True,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer=args.opt_method,
                               model_scope='train',
                               h_max_log_smooth=args.h_max_log_smooth)

# specify how much memory to use on each GPU
gpu_mem_portion=0.45
n_core = 16
#with tf.variable_scope("train"), tf.device(DEV):
#  model_train = get_model(hps_train, DATASET, TRAIN_DATA_PATH, mode='train')
#init_op = tf.global_variables_initializer()
#sess = GetTrainingSession(model_train, gpu_mem_portion=gpu_mem_portion)
model_train, model_eval, init_op, sess = setup(hps_train, hps_eval, gpu_mem_portion, DEV, DATASET, TRAIN_DATA_PATH, TEST_DATA_PATH)

# run steps
#general_log_dir = "../results"
#if not os.path.isdir(general_log_dir):
#  os.mkdir(general_log_dir)
#log_dir = general_log_dir + "/test-release-cifar100-final"
log_dir = args.log_dir
num_step = 150001
display_interval=2500
test_interval = 1000

if not os.path.isdir(log_dir):
  os.makedirs(log_dir)

sess.run(init_op)

loss_list = []
precision_list = []
for i in range(num_step):
  loss, _ = sess.run( [model_train.cost, model_train.train_op ] )
  loss_list.append(loss)
  if (i % display_interval == 0 or i == 50) and (i != 0):
    print("plotting for iteration ", i)
    plot_loss(loss_list, log_dir, i)
    np.savetxt(log_dir + "/loss_full.txt", np.array(loss_list) )

  if (i % test_interval == 0) and (i != 0):
    print("start test ")
    # do evaluation on whole test set
    precision = evaluate(sess, model_eval, n_batch=100)
    precision_list.append(precision)
    print("precision %.6f" % precision)

    plt.figure()
    plt.plot(test_interval * np.arange(len(precision_list) ), np.array(precision_list) )
    plt.title("Test precision " + str(precision) )
    plt.ylim( [0, 1] )
    plt.savefig(log_dir + "/fig_acc.png")
    plt.close()

    with open(log_dir + "/test_acc.txt", "w") as f:
        np.savetxt(f, np.array(precision_list) )

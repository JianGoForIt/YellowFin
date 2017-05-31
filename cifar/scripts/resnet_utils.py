import gzip
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

import resnet_model
from resnet_main import train
import cifar_input

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_model(hps, dataset, train_data_path, mode='train'):
    images, labels = cifar_input.build_input(
      dataset, train_data_path, hps.batch_size, mode)
    model = resnet_model.ResNet(hps, images, labels, mode)
    model.build_graph()
    return model


def GetTrainingSession(model_train, n_core=16, gpu_mem_portion=0.99):
    mon_sess = tf.train.MonitoredTrainingSession(
      config=tf.ConfigProto(intra_op_parallelism_threads=n_core,
                            inter_op_parallelism_threads=n_core,
                            allow_soft_placement=True, log_device_placement=True,
                            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_portion)))
    return mon_sess


def plot_loss(loss_list, log_dir, iter_id):
    def running_mean(x, N):
      cumsum = np.cumsum(np.insert(x, 0, 0)) 
      return (cumsum[N:] - cumsum[:-N]) / N 
    plt.figure()
    plt.semilogy(loss_list, '.', alpha=0.2, label="Loss")
    plt.semilogy(running_mean(loss_list,100), label="Average Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
    print "figure plotted"
    plt.close()
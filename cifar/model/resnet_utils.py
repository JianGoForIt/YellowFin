from __future__ import print_function
import gzip
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

import resnet_model
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


def setup(hps_train, hps_eval, gpu_mem_portion, DEV, DATASET, TRAIN_DATA_PATH, TEST_DATA_PATH):
    with tf.variable_scope("train"), tf.device(DEV):
        model_train = get_model(hps_train, DATASET, TRAIN_DATA_PATH, mode='train')
        
    # use the train for the scope name just for reuse the variable
    with tf.variable_scope("train", reuse=True), tf.device(DEV):
        model_eval = get_model(hps_eval, DATASET, TEST_DATA_PATH, mode='eval')

    init_op = tf.global_variables_initializer()
    mon_sess = GetTrainingSession(model_train, gpu_mem_portion=gpu_mem_portion)
    return model_train, model_eval, init_op, mon_sess


def GetTrainingSession(model_train, n_core=16, gpu_mem_portion=0.99):
  mon_sess = tf.train.MonitoredTrainingSession(
    config=tf.ConfigProto(intra_op_parallelism_threads=n_core,
                          inter_op_parallelism_threads=n_core,
                          allow_soft_placement=True, log_device_placement=True,
                          gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_portion)))
  return mon_sess

def evaluate(sess, model, n_batch=100):
  start_time = time.time()
  total_prediction, correct_prediction = 0, 0
  for _ in xrange(n_batch):
    (loss, predictions, truth, train_step) = sess.run(
          [model.cost, model.predictions,
           model.labels, model.global_step])

    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]

  precision = 1.0 * correct_prediction / total_prediction
  #best_precision = max(precision, best_precision)
  end_time = time.time()
  print("eval time ", end_time - start_time)
  return precision


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
  print("figure plotted")
  plt.close()

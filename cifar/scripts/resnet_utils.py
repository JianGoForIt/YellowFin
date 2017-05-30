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

def running_mean(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N 


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


def setup(hps_train, hps_eval, gpu_mem_portion, DEV, DATASET, TRAIN_DATA_PATH, TEST_DATA_PATH):
    with tf.variable_scope("train"), tf.device(DEV):
        model_train = get_model(hps_train, DATASET, TRAIN_DATA_PATH, mode='train')
        
    # use the train for the scope name just for reuse the variable
    with tf.variable_scope("train", reuse=True), tf.device(DEV):
        model_eval = get_model(hps_eval, DATASET, TEST_DATA_PATH, mode='eval')
    # model_eval = None

    init_op = tf.global_variables_initializer()
    mon_sess = GetTrainingSession(model_train, gpu_mem_portion=gpu_mem_portion)
    return model_train, model_eval, init_op, mon_sess


def evaluate(sess, model, best_precision, n_batch=50):
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
    best_precision = max(precision, best_precision)
    end_time = time.time()
    print "eval time ", end_time - start_time
    return best_precision


def training(model_train, model_eval, mon_sess, num_step, lr_vals, mu_vals, clip_thresh_vals, init_op, display_interval, log_dir, test_interval, use_meta=True):
  mon_sess.run(init_op)
  loss_list = []
  precision_list = []

  if use_meta:
    model_train.optimizer.assign_hyper_param_value(lr_vals, mu_vals, clip_thresh_vals)
    relu_rate_list = []
    for i in range(len(model_train.relu_output) ):
      relu_rate_list.append( [] )

  iter_id = 0
  best_precision = 0
  while iter_id < num_step:
      
      if use_meta:
        feed_dict = model_train.optimizer.get_hyper_feed_dict()
        
        output_results = mon_sess.run( [model_train.cost, model_train.grads, model_train.train_op, model_train.relu_output], feed_dict=feed_dict)
      else:
        output_results = mon_sess.run( [model_train.cost, model_train.grads, model_train.train_op] )

      loss = output_results[0]
      grad_vals = output_results[1]
      loss_list.append(loss)
      
      if use_meta:
        relu_output = output_results[3]
        for i, act in enumerate(relu_output):
            relu_rate_list[i].append(1.0 - (act[act < 0].size / float(act.size) ) )
      
      if use_meta:
        model_train.optimizer.on_iter_finish(mon_sess, [ [g, ] for g in grad_vals], loss)
        # model_train.optimizer.on_iter_finish( [ [g, ] for g in grad_vals], loss)

      
      if (iter_id % display_interval == 0 or iter_id==50 or iter_id == 250 or iter_id == 500 or iter_id == 1000 or iter_id == 2500) and iter_id != 0:
          # evaluate bundle        
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
          plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".png")
          plt.close()
          
          if use_meta:    
            # save relu rate
            for i, relu_rate in enumerate(relu_rate_list):
                with open(log_dir + "/relu_rate_" + str(i) + ".txt", "w") as f:
                    np.savetxt(f, np.array(relu_rate) )
            
            model_train.optimizer.plot_curv(log_dir=log_dir)

          with open(log_dir + "/loss.txt", "w") as f:
            np.savetxt(f, np.array(loss_list) )
          
      if iter_id % test_interval == 0:
          print "start test "
          # do evaluation on whole test set
          best_precision = evaluate(mon_sess, model_eval, best_precision, n_batch=50)
          precision_list.append(best_precision)
          print "best precision %.6f" % best_precision
          
          plt.figure()
          plt.plot(test_interval * np.arange(len(precision_list) ), np.array(precision_list) )
          plt.title("Test precision " + str(best_precision) )
          plt.ylim( [0, 1] )
          plt.savefig(log_dir + "/fig_acc.png")
          plt.close()
          
          with open(log_dir + "/test_acc.txt", "w") as f:
              np.savetxt(f, np.array(precision_list) )
          
      iter_id += 1
        
    
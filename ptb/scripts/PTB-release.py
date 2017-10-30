from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
import cPickle as pickle

sys.path.append('../model')
from ptb_word_lm import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append("../tuner_utils/")
from debug_plot import plot_func

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', type=float, default=1.0,
                     help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--opt_method', type=str, default="YF", help='optimizer')
parser.add_argument('--log_dir', type=str, default="results/", help="log folder")
parser.add_argument('--h_max_log_smooth', action='store_true')

args = parser.parse_args()
#print("use log smooth h_max ", args.h_max_log_smooth)

def construct_model(config, eval_config, raw_data, opt_method):
  train_data, valid_data, test_data, _ = raw_data

  eval_config.batch_size = 1
  eval_config.num_steps = 1

  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  with tf.name_scope("Train"):
    train_input = PTBInput(config=config, data=train_data, name="TrainInput")
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config, input_=train_input, opt_method=opt_method)

  with tf.name_scope("Valid"):
    valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config, input_=valid_input, opt_method=opt_method)

  with tf.name_scope("Test"):
    test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mtest = PTBModel(is_training=False, config=eval_config, input_=test_input, opt_method=opt_method)

  return m, mvalid, mtest

loss_list_visual = []
local_curv_list = [] 
max_curv_list = []
min_curv_list = []
lr_g_norm_list = [] 
lr_g_norm_squared_list = [] 
lr_list = [] 
lr_t_list = []
dr_list = [] 
mu_list = [] 
mu_t_list = [] 
grad_avg_norm_list = []
dist_list = []
grad_var_list = []
move_lr_g_norm_list = [] 
move_lr_g_norm_squared_list = [] 
fast_view_act_list = [] 
lr_grad_norm_clamp_act_list = []


def train_single_step(sess, model, model_eval, model_test, eval_op, iter_id, test_int=1000):
  global state
  global iters
  global costs

  if iter_id % model.input.epoch_size == 0:
    iters = 0
    costs = 0
    state = sess.run(m.initial_state)

  fetches = {
    "cost": model.cost,
    "final_state": model.final_state,
    "grads": model.grads,
    "grad_norm": model.grad_norm,
    "model": model.tvars,
    # for visualization
    "h_max": model.optimizer._h_max,
    "h_min": model.optimizer._h_min,
    "grad_var": model.optimizer._grad_var,
    "dist_to_opt": model.optimizer._dist_to_opt_avg,
    "lr": model.optimizer._lr,
    "mu": model.optimizer._mu
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  feed_dict = {}
  for i, (c, h) in enumerate(model.initial_state):
    feed_dict[c] = state[i].c
    feed_dict[h] = state[i].h
  vals = sess.run(fetches, feed_dict)
  cost = vals["cost"]
  state = vals["final_state"]
  grads = vals["grads"]
  grad_norm = vals["grad_norm"]
  w = vals["model"]


  # for printing issue
  loss_list_visual.append(cost)
  local_curv_list.append(vals["grad_norm"]**2)
  max_curv_list.append(vals["h_max"])
  min_curv_list.append(vals["h_min"])
  lr_g_norm_list.append(vals["lr"] * vals["grad_norm"]) 
  lr_g_norm_squared_list.append(vals["lr"] * vals["grad_norm"]**2) 
  lr_list.append(vals["lr"])
  lr_t_list.append(vals["lr"])
  dr_list.append((vals["h_max"] + 1e-6) / (vals["h_min"] + 1e-6)) 
  mu_list.append(vals["mu"]) 
  mu_t_list.append(vals["mu"]) 
  # grad_avg_norm_list = []
  dist_list.append(vals["dist_to_opt"])
  grad_var_list.append(vals["grad_var"])
  # move_lr_g_norm_list = [] 
  # move_lr_g_norm_squared_list = [] 
  # fast_view_act_list = [] 
  # lr_grad_norm_clamp_act_list = []

  costs += cost
  iters += model.input.num_steps

  train_perp = np.exp(cost / model.input.num_steps)
  val_perp = None
  test_perp = None

  if iter_id == 10 or iter_id == 30 or iter_id == 100 or iter_id %1000 == 0:
    plot_func(args.log_dir, iter_id, loss_list_visual, local_curv_list, max_curv_list, min_curv_list,
             lr_g_norm_list, lr_g_norm_squared_list, lr_list, lr_t_list, dr_list, 
             mu_list, mu_t_list, grad_avg_norm_list,
             dist_list, grad_var_list, move_lr_g_norm_list, move_lr_g_norm_squared_list, 
             fast_view_act_list, lr_grad_norm_clamp_act_list)
    print("figure plotted")


  if iter_id % (model.input.epoch_size // 10) == 10:
    print("%.3f perplexity: %.3f speed: %.0f wps" %
    (iter_id * 1.0 / model.input.epoch_size, np.exp(costs * model.input.num_steps / iters), 0))

  if iter_id % test_int == 0 and iter_id != 0:
      print("test interval ", test_int)
      val_perp = run_epoch(sess, model_eval)
      print("Valid Perplexity: %.3f" % val_perp)
  #     test_perp = run_epoch(sess, model_test)
  #     print("Test Perplexity: %.3f" % test_perp)

  return cost, train_perp, val_perp, test_perp


# load data and generate config
data_path = "../../data/ptb/data"
train_config = SmallConfig()
eval_config = SmallConfig()
raw_data = reader.ptb_raw_data(data_path)

# construct models
tf.reset_default_graph()
opt_method = 'YF'
train_config.h_max_log_smooth = args.h_max_log_smooth
m, m_val, m_test = construct_model(train_config, eval_config, raw_data, args.opt_method)

lr_as = tf.assign(m._lr, args.lr)
mu_as = tf.assign(m._mu, 0.9)
# set clipping thresh to super large so that there is no clipping
thresh_as = tf.assign(m._grad_norm_thresh, 1e10 )

init_op = tf.global_variables_initializer()
os.system("rm -r ./tmp")
sv = tf.train.Supervisor(logdir='./tmp')


# set trainining parameters
num_step = 2323 * 13
train_batch_size = 64
display_interval=1000
test_int = 1000
n_core=20
#general_log_dir = "../results"
#if not os.path.isdir(general_log_dir):
#  os.mkdir(general_log_dir)
#log_dir = general_log_dir + "/test-release-ptb"
log_dir = args.log_dir
if not os.path.isdir(log_dir):
  os.makedirs(log_dir)

np.random.seed(args.seed)
tf.set_random_seed(args.seed)
print("using random seed", args.seed)

loss_list = []
train_perp_list = []
val_perp_list = []
test_perp_list = []
with sv.managed_session(config=tf.ConfigProto(inter_op_parallelism_threads=n_core,
          intra_op_parallelism_threads=n_core,
          gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45))) as sess:
  sess.run(init_op)
  state = sess.run(m.initial_state)
  # costs and iters are for calculating perplexity
  costs = 0
  iters = 0
  if args.opt_method != "YF":
     sess.run([lr_as, mu_as, thresh_as])
     print("test lr outside ", sess.run([m._lr, m._mu, m._grad_norm_thresh]))

  for iter_id in range(num_step):
    loss, train_perp, val_perp, test_perp = \
      train_single_step(sess, m, m_val, m_test, m.train_op, iter_id)

    loss_list.append(loss)
    if train_perp is not None:
      train_perp_list.append(train_perp)
    if val_perp is not None:
      val_perp_list.append(val_perp)
    if test_perp is not None:
      test_perp_list.append(test_perp)

    if iter_id % display_interval == 0 and iter_id != 0:
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
      with open(log_dir + "/loss.txt", "w") as f:
	np.savetxt(f, np.array(loss_list) )
      with open(log_dir + "/val_perp.txt", "w") as f:
        np.savetxt(f, np.array(val_perp_list) )

      plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".jpg")
      plt.close()

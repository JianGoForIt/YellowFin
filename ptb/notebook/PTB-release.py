
# coding: utf-8

# In[1]:

import os, sys
import numpy as np
import tensorflow as tf
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


# In[2]:

sys.path.append('../scripts')
from ptb_word_lm_wo_thresh import *
sys.path.append('../tuner_utils')
from robust_region_v2 import *


# In[3]:

def construct_model(config, eval_config, raw_data, dev, opt_method):
    train_data, valid_data, test_data, _ = raw_data

#     config = get_config()
#     eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

#       with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input, dev=dev, opt_method=opt_method)
#         tf.scalar_summary("Training Loss", m.cost)
#         tf.scalar_summary("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input, dev=dev, opt_method=opt_method)
#         tf.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input, dev=dev, opt_method=opt_method)
    
    return m, mvalid, mtest


# In[4]:

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 


# In[5]:

def train_single_step_frame(sess, model, model_eval, model_test, eval_op, lr_val, mom_val, clip_norm_base, iter_id, verbose=True, test_int=500):    
    global state
    global iters
    global costs
    
#     model.assign_lr(sess, lr_val)
    model.assign_hyper_param(sess, lr_val, mom_val, clip_norm_base)
    
    if iter_id % model.input.epoch_size == 0:
        iters = 0
        costs = 0
        state = sess.run(m.initial_state)
    
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "grads": model.grads,
        "grad_norm": model.grad_norm,
        "model": model.tvars
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

    costs += cost
    iters += model.input.num_steps

    train_perp = np.exp(cost / model.input.num_steps)
    val_perp = None
    test_perp = None
    
#     print "iter id ", iter_id
    
    
    if iter_id % (model.input.epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" %
        (iter_id * 1.0 / model.input.epoch_size, np.exp(costs * model.input.num_steps / iters), 0))     
        
    if iter_id % test_int == 0 and iter_id != 0:
        print("test interval ", test_int)
#         val_perp = run_epoch(sess, model_eval)
#         print("Valid Perplexity: %.3f" % val_perp)
        test_perp = run_epoch(sess, model_test)
        print("Test Perplexity: %.3f" % test_perp)
        
#     if iter_id % model.input.epoch_size == 10:
#         file_name = './model_ckpt_adam/iter_' + str(iter_id)
#         with open(file_name, "wb") as outfile:
#             pickle.dump(w, outfile, pickle.HIGHEST_PROTOCOL)
#         print "model saved for iter ", iter_id
        
    return grads, grad_norm, cost, train_perp, val_perp, test_perp 


# ### load data and generate config

# In[6]:

data_path = "../../data/ptb/data"
train_config = SmallConfig()
eval_config = SmallConfig()
raw_data = reader.ptb_raw_data(data_path)
dev='cpu:0'


# ### construct models

# In[7]:

tf.reset_default_graph()
opt_method = 'YF'
# with tf.device(dev):
m, m_val, m_test = construct_model(train_config, eval_config, raw_data, dev, opt_method)
init_op = tf.global_variables_initializer()
sv = tf.train.Supervisor(logdir='./tmp_init_thresh_inf')


# In[8]:

num_step = 2323 * 13
# num_step = 20
lr_val = 1.0
mom_val = 0.9
clip_norm_base = 10.0
train_batch_size = 64
do_pred = True
grad_beta = 0.9
dist_beta = 0.9
curv_beta = 0.999
param_beta = 0.999
sliding_win_width=10
display_interval=1000
test_int = 1000
n_core=20

general_log_dir = "../results"


# ### run experiments lr = 0.1

# In[9]:

os.system("rm -r ./tmp")
lr_val = 1.0 
mom_val = 0.0
do_pred = True
log_dir = general_log_dir + "/test-release-ptb"
# log_dir = general_log_dir + "/noisy_lr_" + str(lr_val) + "_mom_" + str(mom_val) + "_exp_0.999_test_perp_init_thresh_10_percent_0_100_no_thresh-seed-2"
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

with sv.managed_session(config=tf.ConfigProto(inter_op_parallelism_threads=n_core,
                   intra_op_parallelism_threads=n_core, gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5))) as sess:
    tf.set_random_seed(2)
    sess.run(init_op)
    state = sess.run(m.initial_state)
    costs = 0
    iters = 0
    m.assign_hyper_param(sess, lr_val, mom_val, clip_norm_base)
    train_single_step = lambda lr_val, mom_val, clip_norm_base, iter_id: train_single_step_frame(sess, m, m_val, m_test, m.train_op, lr_val, mom_val, clip_norm_base, iter_id=iter_id, test_int=test_int)
    loss_list, g_norm_list, perp_train_list, perp_val_list, perp_test_list = train_rave_noisy_ptb(num_step, lr_val, mom_val, 
                         clip_norm_base, do_pred, curv_beta, param_beta, train_single_step, sliding_win_width, display_interval, log_dir=log_dir)
    
np.savetxt(log_dir + "/loss_full.txt", np.array(loss_list) )
np.savetxt(log_dir + "/g_norm_full.txt", np.array(g_norm_list) )
np.savetxt(log_dir + "/train_perp_full.txt", np.array(perp_train_list) )
np.savetxt(log_dir + "/val_perp_full.txt", np.array(perp_val_list) )
np.savetxt(log_dir + "/test_perp_full.txt", np.array(perp_test_list) )


# In[ ]:




# In[ ]:




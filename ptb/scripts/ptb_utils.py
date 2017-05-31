import os, sys
import numpy as np
import tensorflow as tf
import cPickle as pickle

import matplotlib.pyplot as plt
from ptb_word_lm import *
sys.path.append('../tuner_utils')
from robust_region_adagrad_per_layer import *


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 


def construct_model(config, eval_config, raw_data, dev, opt_method):
    train_data, valid_data, test_data, _ = raw_data

    eval_config.batch_size = 1
    eval_config.num_steps = 1

    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input, dev=dev, opt_method=opt_method)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input, dev=dev, opt_method=opt_method)

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input, dev=dev, opt_method=opt_method)
    
    return m, mvalid, mtest



def training(model, model_eval, model_test, sess, num_step, lr_vals, mu_vals, clip_thresh_vals, init_op, display_interval, log_dir, test_interval, use_meta=True):
    sess.run(init_op)
    loss_list = []
    r_loss_list = []
    train_perp_list = []
    r_train_perp_list = []
    val_perp_list = []
    test_perp_list = []
    grad_norm_list = []

    if use_meta:
        model.optimizer.assign_hyper_param_value(lr_vals, mu_vals, clip_thresh_vals)
    else:
        model.assign_hyper_param(sess, lr_vals, mu_vals, clip_thresh_vals)
    
    iter_id = 0
    costs = 0
    iters = 0
    while iter_id < num_step:      
        if iter_id % model.input.epoch_size == 0:
            iters = 0
            costs = 0
            state = sess.run(model.initial_state)
        
        # setup fetch
        fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
            "grads": model.grads,
            "grad_norm": model.grad_norm,
            "model": model.tvars,
            "train_op": model._train_op
        }

        # setup feed
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        
        # TODO recover after sanity check
        if use_meta:
            feed_dict_hyper = model.optimizer.get_hyper_feed_dict()
            feed_dict.update(feed_dict_hyper)
                  
        vals = sess.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        grads = vals["grads"]
        grad_norm = vals["grad_norm"]
        w = vals["model"]
          
        costs += cost
        iters += model.input.num_steps
            
        loss_list.append(cost)
        r_loss_list.append(costs/float(iters) )
        train_perp_list.append(np.exp(cost * model.input.num_steps /model.input.num_steps) )
        r_train_perp_list.append(np.exp(costs * model.input.num_steps / iters) )
        grad_norm_list.append(grad_norm)
        
        if use_meta:
            model.optimizer.on_iter_finish( [ [g, ] for g in grads], cost)

        if iter_id % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
            (iter_id * 1.0 / model.input.epoch_size, np.exp(costs * model.input.num_steps / iters), 0))     

        if (iter_id % display_interval == 0 or iter_id == 250 or iter_id == 500 or iter_id == 1000 or iter_id == 2500) and iter_id != 0:
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
            plt.savefig(log_dir + "/fig_loss.png")
            plt.close()
            
            
            if use_meta:
                model.optimizer.plot_curv(log_dir=log_dir)
            else:
                # evaluate bundle        
                plt.figure()
                plt.semilogy(grad_norm_list, '.', alpha=0.2, label="grad norm")
#                 plt.semilogy(running_mean(loss_list,100), label="Average Loss")
                plt.xlabel('Iterations')
                plt.ylabel('gradient_norm')
                plt.legend()
                plt.grid()
                ax = plt.subplot(111)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                    ncol=3, fancybox=True, shadow=True)
                plt.savefig(log_dir + "/fig_grad_norm.png")
                plt.close()
                
                with open(log_dir + "/loss.txt", "w") as f:
                    np.savetxt(f, np.array(loss_list) )

        if iter_id % test_interval == 0 and iter_id != 0:
            start_time = time.time()
            # val_perp = run_epoch(sess, model_eval)
            # print("Valid Perplexity: %.3f" % val_perp)
            # end_time = time.time()
            # print("val done in ", end_time - start_time)
            # val_perp_list.append(val_perp)
            
            start_time = time.time()
            test_perp = run_epoch(sess, model_test)
            print("Test Perplexity: %.3f" % test_perp)
            end_time = time.time()
            print("test done in ", end_time - start_time)
            test_perp_list.append(test_perp)

            plt.figure()
            plt.plot(test_interval * np.arange(len(test_perp_list) ), np.array(test_perp_list), label="test perp")
            plt.plot(test_interval * np.arange(len(val_perp_list) ), np.array(val_perp_list), label="val perp")
            plt.title("test perplexity: " + str(test_perp_list[-1] ) )# + "val perplexity: " + str(val_perp_list[-1] ))
            plt.grid()
            plt.savefig(log_dir + "/fig_perp.png")
            plt.close()

            with open(log_dir + "/test_perp.txt", "w") as f:
                np.savetxt(f, np.array(test_perp_list) )
                
            # with open(log_dir + "/val_perp.txt", "w") as f:
            #     np.savetxt(f, np.array(val_perp_list) )

        iter_id += 1
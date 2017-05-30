import numpy as np
# import matplotlib.pyplot as plt
from math import ceil, floor
from scipy.optimize import minimize
from tensorflow.python.training import momentum
import tensorflow as tf
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def running_mean(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N 


class YFOptimizerUnit(object):
  def __init__(self, lr_val, mu_val, clip_thresh_val, alpha,
               high_pct=99.5, low_pct=0.5, gamma=0.3, 
               mu_update_interval=10, use_placeholder=False, use_async=False, slow_start_iters=200):
    # use placeholder if the graph complain assign op can not be used 
    # after the graph is finalized.
    if use_placeholder:
      self.lr_var = tf.placeholder(tf.float32, shape=() )
      self.mu_var = tf.placeholder(tf.float32, shape=() )
      self.clip_thresh_var = tf.placeholder(tf.float32, shape=() )
      self.lr_val = lr_val
      self.mu_val = mu_val
      self.clip_thresh_val = clip_thresh_val
    else:
      print("using variable style hyper parameter")
      self.lr_var = tf.Variable(lr_val, trainable=False)
      self.mu_var = tf.Variable(mu_val, trainable=False)
      self.clip_thresh_var = tf.Variable(clip_thresh_val, trainable=False)

    self._optimizer = tf.train.MomentumOptimizer(self.lr_var, self.mu_var)

    self._alpha = alpha
    self._gamma = gamma
    self._mu_update_interval = mu_update_interval
    self._curv_list = []
    self._max_curv = None
    self._grad_sum_square = None
    self._iter_id = 0
    self._high_pct = high_pct
    self._low_pct = low_pct
    
    self._global_step = tf.Variable(0, trainable=False)

    # monitoring code
    self._max_curv_list = []
    self._lr_list = []
    self._mu_list = []
    self._lr_grad_list = []
    # clip_list monitor thresh over lr * clip_thresh
    self._clip_list = []
    self._dr_list = []
    self._id = None
    self._slow_start_iters = slow_start_iters

    self.use_async = use_async
    
    # TODO remove for debug
    self._accum_grad_squared_list = []


  def setup_slots(self):
    for tvar in self._tvars:
      # we setup momentum slot in advance, and the original momentum creation will
      # check the existence and keep the one here
      self._optimizer._zeros_slot(tvar, "grad_squared", self._optimizer._name)
      self._optimizer._zeros_slot(tvar, "grad_squared_accum", self._optimizer._name)
      if self.use_async:
        self._optimizer._zeros_slot(tvar, "momentum", self._optimizer._name)
        self._optimizer._zeros_slot(tvar, "momentum_delay", self._optimizer._name)
        self._optimizer._zeros_slot(tvar, "momentum_diff", self._optimizer._name)
        self._optimizer._zeros_slot(tvar, "grad", self._optimizer._name)
    return


  def before_apply(self):
    # copy old momentum before the underlying momentum optimizer get the old momentum
    tune_prep_ops = []
    if self.use_async:
      for tvar in self._tvars:
        tune_prep_ops.append(
          tf.assign(self._optimizer.get_slot(tvar, "momentum_delay"), 
          self._optimizer.get_slot(tvar, "momentum") ) )
    return tf.group(*tune_prep_ops)


  def after_apply(self):
    self._moving_averager = tf.train.ExponentialMovingAverage(decay=self._gamma)
    assert self._grads != None and len(self._grads) > 0
    # setup moving average for sync mu estimation part
    after_apply_ops = []
    self._grad_squared = []
    for v in self._grads:
      self._grad_squared.append(v**2)
    moving_ave_op = self._moving_averager.apply(self._grad_squared)
    self._moving_grad_squared = []
    with tf.control_dependencies([moving_ave_op, ]):
      for v in self._grad_squared:
        self._moving_grad_squared.append(self._moving_averager.average(v) )

      if len(self._moving_grad_squared) == 1:
        self._moving_grad_squared = tf.reshape(self._moving_grad_squared[0], [-1] )
      else:
        self._moving_grad_squared = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._moving_grad_squared] )
    after_apply_ops.append(moving_ave_op)

    # calculate the gradient squared accumulator for learning rate
    accum_add_ops = [] 
    for v, grad_squared in zip(self._tvars, self._grad_squared):  
      accumulator = self._optimizer.get_slot(v, "grad_squared_accum")
      accum_add_ops.append(tf.assign(accumulator, accumulator + grad_squared) )
    after_apply_ops += accum_add_ops  

    return tf.group(*after_apply_ops)


  def after_apply_async(self):
    '''
    It has to be called after after_apply()
    '''
    # keep moving average momentum
    after_apply_ops = []
    self._momentum = [self._optimizer.get_slot(tvar, "momentum") for tvar in self._tvars]
    ave_op = self._moving_averager.apply(self._momentum)
    self._moving_momentum = [self._moving_averager.average(v) for v in self._momentum]
    with tf.control_dependencies( [ave_op] ):
      if len(self._moving_momentum) == 1:
        self._moving_momentum = tf.reshape(self._moving_momentum[0], [-1] )
      else:
        self._moving_momentum = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._moving_momentum] )
    after_apply_ops.append(ave_op)

    # keep track of the momentum difference
    diff_ops = []
    for tvar in self._tvars:
      momentum = self._optimizer.get_slot(tvar, "momentum")
      momentum_delay = self._optimizer.get_slot(tvar, "momentum_delay")
      momentum_diff = self._optimizer.get_slot(tvar, "momentum_diff")
      diff_op = tf.assign(momentum_diff, momentum - momentum_delay)
      diff_ops.append(diff_op)
    after_apply_ops += diff_ops
    self._momentum_diff = [self._optimizer.get_slot(tvar, "momentum_diff") for tvar in self._tvars]
    ave_op = self._moving_averager.apply(self._momentum_diff)
    self._moving_momentum_diff = [self._moving_averager.average(v) for v in self._momentum_diff] 
    with tf.control_dependencies( [ave_op] ):
      if len(self._moving_momentum_diff) == 1:
        self._moving_momentum_diff = tf.reshape(self._moving_momentum_diff[0], [-1] )
      else:
        self._moving_momentum_diff = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._moving_momentum_diff] )
    after_apply_ops.append(ave_op)

    # keep track of gradient, the moving grad to server is on server
    # _gradient to distinguish from self._grad
    grad_assign_ops = [tf.assign(self._optimizer.get_slot(tvar, "grad"), grad) for tvar, grad in zip(self._tvars, self._grads) ]
    with tf.control_dependencies(grad_assign_ops):
      self._gradients = [self._optimizer.get_slot(tvar, "grad") for tvar in self._tvars]
      moving_grad_op = self._moving_averager.apply(self._gradients)
      self._moving_grad = [self._moving_averager.average(v) for v in self._gradients]
    with tf.control_dependencies( [moving_grad_op] ):
      if len(self._moving_grad) == 1:
        self._moving_grad = tf.reshape(self._moving_grad, [-1] )
      else:
        self._moving_grad = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._moving_grad] )
    after_apply_ops += grad_assign_ops
    after_apply_ops.append(moving_grad_op)
      
    return tf.group(*after_apply_ops)


  def get_lr_tensor(self):
    grad_squared_per_var_sum = [tf.reduce_sum(self._optimizer.get_slot(tvar, "grad_squared_accum") ) for tvar in self._tvars]
    grad_squared_sum = tf.add_n(grad_squared_per_var_sum)
    
    # # DEBUG
    # self.grad_squared_sum = grad_squared_sum
    # grad_squared_sum = tf.Print(grad_squared_sum, [grad_squared_sum], message="inside denominator")
    # self.grad_squared_sum = grad_squared_sum
    # print "check alpha ", self._alpha
    
    lr = self._alpha * tf.minimum(1.0/tf.constant(self._slow_start_iters, tf.float32) \
       + 1.0/tf.constant(self._slow_start_iters, tf.float32) * tf.cast(self._global_step, tf.float32),
       tf.constant(1.0, tf.float32) ) / tf.sqrt(grad_squared_sum + 1e-6) 
    return lr


  def get_mu_tensor(self):
    '''
    It has to be called after setup_moving_ave
    '''
    moving_grad_squared = self._moving_grad_squared

    n_dim = moving_grad_squared.get_shape().as_list()[0]
    high_rank = ceil(self._high_pct * n_dim / 100)
    low_rank = floor(self._low_pct * n_dim / 100)

    # # DEBUG
    # moving_grad_squared = tf.Print(moving_grad_squared, [moving_grad_squared], message="before zero-out")

    where = tf.not_equal(moving_grad_squared, tf.constant(0, dtype=tf.float32) )
    grad_squared_non_zero = tf.gather_nd(moving_grad_squared, tf.where(where) )

    # ## DEBUG
    # grad_squared_non_zero = tf.Print(grad_squared_non_zero, [grad_squared_non_zero], message="check non-zerp")

    high_val_all, _ = tf.nn.top_k(grad_squared_non_zero, k=max(int(n_dim - high_rank), 1), sorted=True)
    low_val_all, _ = tf.nn.top_k(-grad_squared_non_zero, k=max(int(low_rank), 1), sorted=True)

    high_val = high_val_all[min(int(n_dim - high_rank) - 1, n_dim) ]
    low_val = -low_val_all[max(int(low_rank) - 1, 0) ]

    # high_val = tf.Print(high_val, [high_val, ], message="high_val")
    # low_val = tf.Print(low_val, [low_val, ], message="low_val")

    dr = tf.sqrt(high_val / (low_val + 1e-9) )
    return ( (tf.sqrt(dr) - 1) / (tf.sqrt(dr) + 1) )**2
    

  def get_async_mu_tensor(self):
    '''
    Called after setup_moving_average
    '''
    moving_momentum = self._moving_momentum
    moving_momentum_diff = self._moving_momentum_diff
    moving_grad = self._moving_grad
    n_dim = moving_momentum.get_shape().as_list()[0]
    median_id = n_dim / 2
    mu_array = (moving_momentum + self.lr_var * moving_grad) / (moving_momentum - moving_momentum_diff + 1e-9)

    median_all, _ = tf.nn.top_k(mu_array, k=max(median_id, 1) , sorted=True)
    mu_async = median_all[max(median_id - 1, 0) ]
    
    return mu_async


  def apply_gradients(self, grads_tvars):
    self._grads, self._tvars = zip(*grads_tvars)
    self.setup_slots()

    with tf.variable_scope("before_apply"):
      before_apply_op = self.before_apply()

    with tf.variable_scope("apply_updates"):
      with tf.control_dependencies( [before_apply_op] ):
        self._grads_clip, self._grads_norm = tf.clip_by_global_norm(self._grads, self.clip_thresh_var)
        apply_grad_op = \
          self._optimizer.apply_gradients(zip(self._grads_clip, self._tvars) )

    with tf.variable_scope("after_apply"):
      # with tf.control_dependencies( [apply_grad_op] ):
      # we calculate the gradeint squared here which can
      # be paralleled with apply_grad_op
      after_apply_op = self.after_apply()

    if self.use_async:
      with tf.variable_scope("after_apply_async"):
        with tf.control_dependencies( [apply_grad_op] ):
          after_apply_op_async = self.after_apply_async()
          after_apply_op = tf.group(after_apply_op, after_apply_op_async)

    with tf.control_dependencies( [after_apply_op] ):
      lr = self.get_lr_tensor()
    
      assign_lr_op = tf.assign(self.lr_var, lr)
      mu_sync = self.get_mu_tensor()
    
      if self.use_async:
          mu_async = self.get_async_mu_tensor()
          assign_mu_op = tf.assign(self.mu_var, self.mu_var + mu_sync - mu_async)
      else:
        assign_mu_op = tf.assign(self.mu_var, mu_sync)

    with tf.control_dependencies( [assign_mu_op] ):
      self._increment_global_step_op = tf.assign(self._global_step, self._global_step + 1)

    return tf.group(before_apply_op, apply_grad_op, after_apply_op, 
        assign_mu_op, assign_lr_op, self._increment_global_step_op)
    

  def assign_hyper_param(self, lr_val, mu_val, clip_thresh_val):
    lr_op = self.lr_var.assign(lr_val)
    mu_op = self.mu_var.assign(mu_val)
    clip_thresh_op = self.clip_thresh_var.assign(clip_thresh_val / float(lr_val) )
    self.lr_val = lr_val
    self.mu_val = mu_val
    self.clip_thresh_val = clip_thresh_val
    return tf.group(lr_op, mu_op, clip_thresh_op)


  def assign_hyper_param_value(self, lr_val, mu_val, clip_thresh_val):
    self.lr_val = lr_val
    # TODO change back
    self.mu_val = mu_val
    # self.mu_val = 0.0
    self.clip_thresh_val = clip_thresh_val
    return 


  def get_min_max_curvatures(self):
    all_curvatures = self._curv_list
    t=len(all_curvatures)
    W=10
    start = max([0,t-W])
    max_curv=max(all_curvatures[start:t])
    min_curv=min(all_curvatures[start:t])
    return max_curv, min_curv


  def set_alpha(self, alpha):
    self._alpha = alpha


  def set_slow_start_iters(self, slow_start_iters):
    self._slow_start_iters = slow_start_iters

    
  def get_lr(self):
    # lr = self._alpha / np.sqrt(sum(self._curv_list) + 1e-6)
    # lr = self._alpha *(min([1.0, 1/float(np.sqrt(self._slow_start_iters) )**2+1/float(np.sqrt(self._slow_start_iters) )**2*self._iter_id] ) ) / np.sqrt(sum(self._curv_list) + 1e-6)
    lr = self._alpha *(min([1.0, 1/float(self._slow_start_iters)+1/float(self._slow_start_iters)*self._iter_id] ) ) / np.sqrt(sum(self._curv_list) + 1e-6)
    return lr


  def get_mu(self):
    high_pct = self._high_pct
    low_pct = self._low_pct
    pct_max = np.percentile(self._grad_sum_square[self._grad_sum_square != 0], high_pct)
    pct_min = np.percentile(self._grad_sum_square[self._grad_sum_square != 0], low_pct) 
    dr = np.sqrt(pct_max / (pct_min + 1e-9) ) 
    mu = ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2
    return dr, mu


  def on_iter_finish(self, sess, grad_vals, use_hyper_op=False):
        
    res = sess.run( [self.lr_var, self.mu_var] )
    self._lr_list.append(res[0] )
    self._mu_list.append(res[1] )
    # self._accum_grad_squared_list.append(res[2] )
#     # print "inside ", res[2]
    
    
#     # i = self._iter_id + 1
#     # w = i**(-1.0)
#     # beta_poly = w
#     # gamma = self._gamma

#     # in case there are sparse gradient strutures
#     for i, item in enumerate(grad_vals):
#       if type(item) is not np.ndarray:
#         tmp = np.zeros(item.dense_shape)
#         # note here we have duplicated vectors corresponding to the same word
#         np.add.at(tmp, item.indices, item.values)
#         grad_vals[i] = tmp.copy()

#     grad_sum_square_new = np.hstack( [val.ravel()**2 for val in grad_vals] )
#     self._curv_list.append(np.sum(grad_sum_square_new) )    

    # # this max_curv_new is for clipping
    # max_curv_new, _ = self.get_min_max_curvatures()
    # # update 
    # if self._max_curv == None:
    #     self._max_curv = max_curv_new
    # else:
    #     self._max_curv = (beta_poly**gamma)*max_curv_new + (1-beta_poly**gamma)*self._max_curv
    # if self._grad_sum_square is None:
    #   self._grad_sum_square = grad_sum_square_new 
    # else:
    #   self._grad_sum_square += grad_sum_square_new
    
    # if self._iter_id % self._mu_update_interval == 0 and self._iter_id > 0:
    #   dr, mu_val = self.get_mu()
    #   self._dr_list.append(dr)
    # else:
    #   mu_val = self.mu_val
    # if self._iter_id >= 0:
    #     lr_val = self.get_lr()
    # else:
    #   lr_val = self.lr_val

    # clip_thresh_val = lr_val * np.sqrt(self._max_curv)
            
    # # TODO tidy up capping operation
    # if use_hyper_op:
    #   hyper_op = self.assign_hyper_param(lr_val, min(mu_val, 0.9), min(clip_thresh_val, 1.0) )
    # else:
    #   self.assign_hyper_param_value(lr_val, min(mu_val, 0.9), min(clip_thresh_val, 1.0) )

    # self._max_curv_list.append(self._max_curv)
    # self._lr_list.append(lr_val)
    # self._mu_list.append(min(mu_val, 0.9) )
    # self._lr_grad_list.append(lr_val * np.sqrt(self._curv_list[-1] ) )
    # # clip_list monitor thresh over lr * clip_thresh
    # self._clip_list.append(min(clip_thresh_val, 1.0) )

    # self._iter_id += 1
    
    # if use_hyper_op:
    #   return hyper_op
    # else:
    #   return
    # pass
    return
    

  def plot_curv(self, log_dir='./'):            
    plt.figure()
    # # plt.semilogy(self._lr_grad_list, label="lr * grad")
    # # plt.semilogy(self._max_curv_list, label="max curv for clip")
    # # plt.semilogy(self._clip_list, label="clip thresh")
    # plt.semilogy(self._accum_grad_squared_list, label="denominator")
    # plt.semilogy(self._curv_list, label="clip curvature")
    plt.semilogy(self._lr_list, label="lr")
    plt.semilogy(self._mu_list, label="mu")
    plt.title('LR='+str(self._lr_list[-1] )+' mu='+str(self._mu_list[-1] ) )
    plt.xlabel("iteration")
    plt.grid()
    ax = plt.subplot(111)
    ax.legend(loc='lower left', 
            ncol=2, fancybox=True, shadow=True)
    plt.savefig(log_dir + "/fig_" + str(self._id) + ".png")
    plt.close()
    # save monitoring quantities
    with open(log_dir + "/lr_grad_" + str(self._id) + ".txt", "w") as f:
        np.savetxt(f, np.array(self._lr_grad_list) )
    with open(log_dir + "/max_curv_" + str(self._id) + ".txt", "w") as f:
        np.savetxt(f, np.array(self._max_curv_list) )
    with open(log_dir + "/clip_thresh_" + str(self._id) + ".txt", "w") as f:
        np.savetxt(f, np.array(self._clip_list) )
    with open(log_dir + "/lr_" + str(self._id) + ".txt", "w") as f:
        np.savetxt(f, np.array(self._lr_list) )
    with open(log_dir + "/mu_" + str(self._id) + ".txt", "w") as f:
        np.savetxt(f, np.array(self._mu_list) )  
    with open(log_dir + "/mu_" + str(self._id) + ".txt", "w") as f:
        np.savetxt(f, np.array(self._dr_list) )


class MetaOptimizer(object):
  def __init__(self, lr_vals, mu_vals, clip_thresh_vals, 
               alpha=10, high_pct=0.99, low_pct=0.5,
               gamma=0.5, use_hyper_op=False):
    assert len(lr_vals) == len(mu_vals)
    assert len(mu_vals) == len(clip_thresh_vals)
    self._use_hyper_op = use_hyper_op
    self.clip_thresh_vals = clip_thresh_vals[:]
    self._optimizers = [YFOptimizerUnit(lr_val, mu_val, clip_thresh_val,
                        # use_placeholder=(not use_hyper_op) ) \
                        alpha=alpha/np.sqrt(float(len(lr_vals) ) ), use_placeholder=False) \
                        for lr_val, mu_val, clip_thresh_val in zip(lr_vals, mu_vals, clip_thresh_vals) ]
    for i in range(len(self._optimizers) ):
        self._optimizers[i]._id = i
    self.lr_vars = [optimizer.lr_var for optimizer in self._optimizers]
    self.mu_vars = [optimizer.mu_var for optimizer in self._optimizers]
    self.clip_thresh_vars = [optimizer.clip_thresh_var for optimizer in self._optimizers]
    self._iter_id = 0
    self._loss_list = [] 


  def set_alpha(self, alpha):
    for optimizer in self._optimizers:
        optimizer.set_alpha(alpha)


  def set_slow_start_iters(self, slow_start_iters):
    for optimizer in self._optimizers:
        optimizer.set_slow_start_iters(slow_start_iters)

        
  def apply_gradients(self, grad_tvar_list):
    '''
    Grad_tvar_list is a list. Each list is a list of tuples.
    Each tuple is a (grad, tvar) pair
    '''    
    assert len(grad_tvar_list) == len(self._optimizers)
    assert len(grad_tvar_list) == len(self._optimizers)
    assert len(grad_tvar_list) == len(self._optimizers)
    self._apply_grad_ops = [opt.apply_gradients(grad_tvar) \
      for opt, grad_tvar in zip(self._optimizers, grad_tvar_list) ]
    self.apply_grad_op = tf.group(*self._apply_grad_ops)
    return self.apply_grad_op


  def on_iter_finish(self, sess, grad_vals_list, loss):
    lr_vals = []
    mu_vals = []
    clip_thresh_vals = []
    hyper_ops = []
    use_hyper_op = self._use_hyper_op
    self._loss_list.append(loss)
    for optimizer, grad_vals in zip(self._optimizers, grad_vals_list):
      if use_hyper_op:
        hyper_op = optimizer.on_iter_finish(sess, grad_vals, use_hyper_op)
        hyper_ops.append(hyper_op)
      else:
        optimizer.on_iter_finish(sess, grad_vals, use_hyper_op)
      lr_vals.append(optimizer.lr_val)
      mu_vals.append(optimizer.mu_val)
      clip_thresh_vals.append(optimizer.clip_thresh_val)
      # TODO append monitoring measurements
    self.clip_thresh_vals = clip_thresh_vals[:]
    self._iter_id += 1
    
    if use_hyper_op:
        return tf.group(*hyper_ops)
    else:
        return

    
  def assign_hyper_param(self, lr_vals, mu_vals, clip_thresh_vals):
    assert len(lr_vals) == len(self._optimizers)
    assert len(mu_vals) == len(self._optimizers)
    assert len(clip_thresh_vals) == len(self._optimizers)
    assign_ops = []
    for optimizer, lr_val, mu_val, clip_thresh_val in \
      zip(self._optimizers, lr_vals, mu_vals, clip_thresh_vals):
      assign_ops.append(optimizer.assign_hyper_param(lr_val, mu_val, clip_thresh_val) )
    return tf.group(*assign_ops)


  def assign_hyper_param_value(self, lr_vals, mu_vals, clip_thresh_vals):
    assert len(lr_vals) == len(self._optimizers)
    assert len(mu_vals) == len(self._optimizers)
    assert len(clip_thresh_vals) == len(self._optimizers)
    for optimizer, lr_val, mu_val, clip_thresh_val in \
      zip(self._optimizers, lr_vals, mu_vals, clip_thresh_vals):
      optimizer.assign_hyper_param_value(lr_val, mu_val, clip_thresh_val)
    return


  def plot_curv(self, plot_id=None, log_dir='./'):
    if plot_id == None:
      for i, optimizer in enumerate(self._optimizers):
        print("plot for optimizer " + str(i) )
        optimizer.plot_curv(log_dir=log_dir)
    else:
      for i in plot_id:
        print("plot for optimizer " + str(i) )
        self._optimizers[i].plot_curv(log_dir=log_dir)
    # save the loss
    with open(log_dir + "/loss.txt", "w") as f:
        np.savetxt(f, np.array(self._loss_list) )
    print("ckpt done for iter ", self._iter_id)
    return
            
    
  def get_hyper_feed_dict(self):
    # lr_pair = [ (optimizer.lr_var, optimizer.lr_val) for optimizer in self._optimizers]
    # mu_pair = [ (optimizer.mu_var, optimizer.mu_val) for optimizer in self._optimizers]
    # clip_thresh_pair = [ (optimizer.clip_thresh_var, optimizer.clip_thresh_val) for optimizer in self._optimizers]
    # feed_dict = dict(lr_pair + mu_pair + clip_thresh_pair)
    # return feed_dict
    return dict()


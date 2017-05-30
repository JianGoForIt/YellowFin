import numpy as np
from math import ceil, floor
import tensorflow as tf
from tensorflow.python.training import momentum
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def running_mean(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N 


class YFOptimizerUnit(object):
  def __init__(self, lr=1.0, mu=0.0, clip_thresh=10000.0, beta=0.999, curv_win_width=20,
    mu_update_interval=1, use_async=False):
    # clip thresh is the threshold value on ||lr * gradient||
    self._lr = lr
    self._mu = mu
    self._clip_thresh_ = clip_thresh

    self._lr_var = tf.Variable(lr, dtype=tf.float32, name="YF_lr", trainable=False)
    self._mu_var = tf.Variable(mu, dtype=tf.float32, name="YF_mu", trainable=False)
    self._clip_thresh_var = tf.Variable(clip_thresh, dtype=tf.float32, name="YF_clip_thresh", trainable=False)

    # the underlying momentum optimizer
    self._optimizer = tf.train.MomentumOptimizer(self._lr_var, self._mu_var)

    # moving average for statistics
    self._beta = beta
    self._moving_averager = None
    
    # for global step counting    
    self._global_step = tf.Variable(0, trainable=False)

    # for conditional tuning
    self._do_tune = tf.greater(self._global_step, tf.constant(0) )

    # option for async version
    self._use_async = use_async

    self._tvars = None

    # for curvature range
    self._curv_win_width = curv_win_width
    self._curv_win = None


  def setup_slots(self):
    for tvar in self._tvars:
      # self._optimizer._zeros_slot(tvar, "grad squared t", self._optimizer._name)
      if self._use_async:
        # we setup momentum slot in advance, and the original momentum creation will
        # check the existence and keep the one here
        self._optimizer._zeros_slot(tvar, "momentum", self._optimizer._name)
        self._optimizer._zeros_slot(tvar, "momentum_delay", self._optimizer._name)
        self._optimizer._zeros_slot(tvar, "momentum_diff", self._optimizer._name)
        self._optimizer._zeros_slot(tvar, "grad", self._optimizer._name)
    return


  def before_apply(self):
    # copy old momentum before the underlying momentum optimizer get the old momentum
    tune_prep_ops = []
    if self._use_async:
      for tvar in self._tvars:
        tune_prep_ops.append(
          tf.assign(self._optimizer.get_slot(tvar, "momentum_delay"), 
          self._optimizer.get_slot(tvar, "momentum") ) )
    return tf.group(*tune_prep_ops)


  def curvature_range(self):
    # set up the curvature window
    self._curv_win = \
      tf.Variable(np.zeros( [self._curv_win_width, ] ), dtype=tf.float32, name="curv_win", trainable=False)
    self._curv_win = tf.scatter_update(self._curv_win, 
      self._global_step % self._curv_win_width, self._grad_norm_squared)
    # note here the iterations start from iteration 0
    valid_window = tf.slice(self._curv_win, tf.constant( [0, ] ), 
      tf.expand_dims(tf.minimum(tf.constant(self._curv_win_width), self._global_step + 1), dim=0) )
    self._h_min_t = tf.reduce_min(valid_window)
    self._h_max_t = tf.reduce_max(valid_window)

    curv_range_ops = []
    with tf.control_dependencies([self._h_min_t, self._h_max_t] ):
      avg_op = self._moving_averager.apply([self._h_min_t, self._h_max_t] )
      with tf.control_dependencies([avg_op] ):
        self._h_min = tf.identity(self._moving_averager.average(self._h_min_t) )
        self._h_max = tf.identity(self._moving_averager.average(self._h_max_t) )
    curv_range_ops.append(avg_op)
    return curv_range_ops


  def grad_variance(self):
    grad_var_ops = []
    avg_op = self._moving_averager.apply(self._grads)
    grad_var_ops.append(avg_op)
    with tf.control_dependencies([avg_op] ):
      self._grad_avg = [self._moving_averager.average(val) for val in self._grads]
      self._grad_avg_squared = [val**2 for val in self._grad_avg]
    self._grad_var = self._grad_norm_squared_avg - tf.add_n( [tf.reduce_sum(val) for val in self._grad_avg_squared] )
    return grad_var_ops


  def dist_to_opt(self):
    dist_to_opt_ops = []
    # running average of the norm of gradeint
    self._grad_norm = tf.sqrt(self._grad_norm_squared)
    avg_op = self._moving_averager.apply([self._grad_norm,] )
    dist_to_opt_ops.append(avg_op)
    # TODO verify if the exe order is right in here
    with tf.control_dependencies([avg_op] ):
      self._grad_norm_avg = self._moving_averager.average(self._grad_norm)
      # single iteration distance estimation, note here self._grad_norm_avg is per variable
      self._dist_to_opt = self._grad_norm_avg / self._grad_norm_squared_avg
    # running average of distance
    avg_op = self._moving_averager.apply([self._dist_to_opt] )
    dist_to_opt_ops.append(avg_op)
    # TODO verify if the exe order is right in here
    with tf.control_dependencies([avg_op]):
      self._dist_to_opt_avg = tf.identity(self._moving_averager.average(self._dist_to_opt) )
    return dist_to_opt_ops


  def after_apply(self):
    # print "decay ", self._beta
    self._moving_averager = tf.train.ExponentialMovingAverage(decay=self._beta)
    assert self._grads != None and len(self._grads) > 0
    after_apply_ops = []

    # get per var g**2 and norm**2
    self._grad_squared = []
    self._grad_norm_squared = []
    for g in self._grads:
      with ops.colocate_with(g):
        self._grad_squared.append(g**2)
    self._grad_norm_squared = [tf.reduce_sum(grad_squared) for grad_squared in self._grad_squared]

    # the following running average on squared norm of gradient is shared by grad_var and dist_to_opt
    avg_op = self._moving_averager.apply(self._grad_norm_squared)
    with tf.control_dependencies([avg_op] ):
      self._grad_norm_squared_avg = [self._moving_averager.average(val) for val in self._grad_norm_squared]
      self._grad_norm_squared = tf.add_n(self._grad_norm_squared)
      self._grad_norm_squared_avg = tf.add_n(self._grad_norm_squared_avg)
    after_apply_ops.append(avg_op)

    with tf.control_dependencies([avg_op] ):
      curv_range_ops = self.curvature_range()
      after_apply_ops += curv_range_ops
      grad_var_ops = self.grad_variance()
      after_apply_ops += grad_var_ops
      dist_to_opt_ops = self.dist_to_opt() 
      after_apply_ops += dist_to_opt_ops

    return tf.group(*after_apply_ops)


  def get_lr_tensor(self):
    lr = (1.0 - tf.sqrt(self._mu) )**2 / self._h_min
    return lr


  def get_mu_tensor(self):
    const_fact = self._dist_to_opt_avg**2 * self._h_min**2 / 2 / self._grad_var
    coef = tf.Variable([-1.0, 3.0, 0.0, 1.0], dtype=tf.float32, name="cubic_solver_coef")
    coef = tf.scatter_update(coef, tf.constant(2), -(3 + const_fact) )        
    roots = tf.py_func(np.roots, [coef], Tout=tf.complex64, stateful=False)
    
    # filter out the correct root
    root_idx = tf.logical_and(tf.logical_and(tf.greater(tf.real(roots), tf.constant(0.0) ),
      tf.less(tf.real(roots), tf.constant(1.0) ) ), tf.less(tf.abs(tf.imag(roots) ), 1e-5) )
    # in case there are two duplicated roots satisfying the above condition
    root = tf.reshape(tf.gather(tf.gather(roots, tf.where(root_idx) ), tf.constant(0) ), shape=[] )
    tf.assert_equal(tf.size(root), tf.constant(1) )

    dr = self._h_max / self._h_min
    mu = tf.maximum(tf.real(root)**2, ( (tf.sqrt(dr) - 1)/(tf.sqrt(dr) + 1) )**2)    
    return mu


  def update_hyper_param(self):
    assign_hyper_ops = []
    # TODO verify whether I need to get a control_dependency here 
    self._mu = tf.identity(tf.cond(self._do_tune, lambda: self.get_mu_tensor(),
      lambda: self._mu_var) )
    with tf.control_dependencies([self._mu] ):
      self._lr = tf.identity(tf.cond(self._do_tune, lambda: self.get_lr_tensor(),
        lambda: self._lr_var) )
    
    with tf.control_dependencies([self._mu, self._lr] ):
      self._mu = self._beta * self._mu_var + (1 - self._beta) * self._mu
      self._lr = self._beta * self._lr_var + (1 - self._beta) * self._lr       
      assign_hyper_ops.append(tf.assign(self._mu_var, self._mu) )
      assign_hyper_ops.append(tf.assign(self._lr_var, self._lr) )
    assign_hyper_op = tf.group(*assign_hyper_ops)
    return assign_hyper_op


  def apply_gradients(self, grads_tvars):
    self._grads, self._tvars = zip(*grads_tvars)
    self.setup_slots()

    # with tf.control_dependencies(pop_ops):
    with tf.variable_scope("before_apply"):
      before_apply_op = self.before_apply()

    with tf.variable_scope("apply_updates"):
      with tf.control_dependencies( [before_apply_op] ):
        self._grads_clip, self._grads_norm = tf.clip_by_global_norm(self._grads, self._clip_thresh_var)
        apply_grad_op = \
          self._optimizer.apply_gradients(zip(self._grads_clip, self._tvars) )

    with tf.variable_scope("after_apply"):
      # we calculate the gradeint squared here which can
      # be paralleled with apply_grad_op. 
      after_apply_op = self.after_apply()

    with tf.variable_scope("update_hyper"):
      with tf.control_dependencies( [after_apply_op] ):
        update_hyper_op = self.update_hyper_param()

    with tf.control_dependencies([update_hyper_op] ):
      self._increment_global_step_op = tf.assign(self._global_step, self._global_step + 1)

    return tf.group(before_apply_op, apply_grad_op, after_apply_op, update_hyper_op, self._increment_global_step_op)
    

#   def after_apply_async(self):
#     '''
#     It has to be called after after_apply()
#     '''
#     if self._moving_averager == None:
#         self._moving_averager = tf.train.ExponentialMovingAverage(decay=self._gamma)
#     # keep moving average momentum
#     after_apply_ops = []
#     # note here we get the negative of our definition of momentum
#     self._momentum = [self._optimizer.get_slot(tvar, "momentum") for tvar in self._tvars]
#     ave_op = self._moving_averager.apply(self._momentum)
#     self._moving_momentum = [self._moving_averager.average(v) for v in self._momentum]
#     with tf.control_dependencies( [ave_op] ):
#       if len(self._moving_momentum) == 1:
#         self._moving_momentum = tf.reshape(self._moving_momentum[0], [-1] )
#       else:
#         self._moving_momentum = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._moving_momentum] )
#       # if len(self._moving_momentum) == 1:
#       #   self._moving_momentum = tf.reshape(self._momentum[0], [-1] )
#       # else:
#       #   self._moving_momentum = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._momentum] )


#     after_apply_ops.append(ave_op)

#     # DEBUG
#     self.test_momentum = []
#     self.test_momentum_delay = []
#     self.test_momentum_diff = []

#     # keep track of the momentum difference
#     diff_ops = []
#     for tvar in self._tvars:
#       momentum = self._optimizer.get_slot(tvar, "momentum")

#       # # DEBUG
#       # if tvar.name == self._tvars[0].name:
#       #   momentum = tf.Print(momentum, [momentum, ], message="momentum[0]")

#       momentum_delay = self._optimizer.get_slot(tvar, "momentum_delay")
#       momentum_diff = self._optimizer.get_slot(tvar, "momentum_diff")


#       # if tvar.name == self._tvars[0].name:
#       #   momentum = tf.Print(momentum, [momentum, ], message="momentum")
#       #   momentum_delay = tf.Print(momentum_delay, [momentum_delay, ], message="momentum_delay")

#       # note here, we are using the tf definition of momentum which is the 
#       # negative of our definition of momentum
#       diff_op = tf.assign(momentum_diff, momentum - momentum_delay)


#       # DEBUG
#       self.test_momentum.append(momentum)
#       self.test_momentum_delay.append(momentum_delay)
#       self.test_momentum_diff.append(momentum_diff)

#       diff_ops.append(diff_op)
#     after_apply_ops += diff_ops
#     self._momentum_diff = [self._optimizer.get_slot(tvar, "momentum_diff") for tvar in self._tvars]
#     with tf.control_dependencies(diff_ops):
#       ave_op = self._moving_averager.apply(self._momentum_diff)
#     self._moving_momentum_diff = [self._moving_averager.average(v) for v in self._momentum_diff] 
#     with tf.control_dependencies( [ave_op] ):
#       if len(self._moving_momentum_diff) == 1:
#         self._moving_momentum_diff = tf.reshape(self._moving_momentum_diff[0], [-1] )
#       else:
#         self._moving_momentum_diff = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._moving_momentum_diff] )
      
#       # if len(self._moving_momentum_diff) == 1:
#       #   self._moving_momentum_diff = tf.reshape(self._momentum_diff[0], [-1] )
#       # else:
#       #   self._moving_momentum_diff = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._momentum_diff] )


#     after_apply_ops.append(ave_op)

#     # keep track of gradient, the moving grad to server is on server
#     # _gradient to distinguish from self._grad
#     grad_assign_ops = [tf.assign(self._optimizer.get_slot(tvar, "grad"), grad) for tvar, grad in zip(self._tvars, self._grads) ]
#     with tf.control_dependencies(grad_assign_ops):
#       self._gradients = [self._optimizer.get_slot(tvar, "grad") for tvar in self._tvars]
#       moving_grad_op = self._moving_averager.apply(self._gradients)
#       self._moving_grad = [self._moving_averager.average(v) for v in self._gradients]
#     with tf.control_dependencies( [moving_grad_op] ):
#       if len(self._moving_grad) == 1:
#         self._moving_grad = tf.reshape(self._moving_grad, [-1] )
#       else:
#         self._moving_grad = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._moving_grad] )
#       # if len(self._moving_grad) == 1:
#       #   self._moving_grad = tf.reshape(self._grad, [-1] )
#       # else:
#       #   self._moving_grad = tf.concat(0, [tf.reshape(v, [-1] ) for v in self._gradients] )

#     after_apply_ops += grad_assign_ops
#     after_apply_ops.append(moving_grad_op)
      
#     return tf.group(*after_apply_ops)


#   # def get_lr_tensor(self):
#   #   grad_squared_per_var_sum = [tf.reduce_sum(self._optimizer.get_slot(tvar, "grad_squared_accum") ) for tvar in self._tvars]
#   #   grad_squared_sum = tf.add_n(grad_squared_per_var_sum)
    
#   #   # # DEBUG
#   #   # self.grad_squared_sum = grad_squared_sum
#   #   # grad_squared_sum = tf.Print(grad_squared_sum, [grad_squared_sum], message="inside denominator")
#   #   self.grad_squared_sum = grad_squared_sum
#   #   print "check alpha ", self._alpha
    
#   #   lr = self._alpha * tf.minimum(1.0/tf.constant(self._slow_start_iters, tf.float32) \
#   #      + 1.0/tf.constant(self._slow_start_iters, tf.float32) * tf.cast(self._global_step, tf.float32),
#   #      tf.constant(1.0, tf.float32) ) / tf.sqrt(grad_squared_sum + 1e-6) 
    
#   #   # lr = self._alpha / tf.sqrt(grad_squared_sum + 1e-6) 

#   #   return lr


#   def get_async_mu_tensor(self):
#     '''
#     Called after setup_moving_average
#     '''
#     moving_momentum = self._moving_momentum
#     moving_momentum_diff = self._moving_momentum_diff
#     moving_grad = self._moving_grad


#     # self.test_moving_momentum = moving_momentum

#     # #DEBUG
#     # moving_momentum = tf.Print(moving_momentum, [moving_momentum, ], message="moving mom")
#     # moving_momentum_diff = tf.Print(moving_momentum_diff, [moving_momentum_diff, ], message="moving mom diff")
#     # moving_grad = tf.Print(moving_grad, [moving_grad, ], message="moving grad")


#     n_dim = moving_momentum.get_shape().as_list()[0]
#     median_id = n_dim / 2
#     # note here we negate the moving average of momentum and momentum difference
#     # In addition, the momentum here is actually momentum / lr
#     # Thus the denominator  (-moving_momentum + self.lr_var * moving_grad) in our definition
#     # is  (-moving_momentum + moving_grad) using tensorflow momentum definition
#     mu_array = (-moving_momentum + moving_grad) / (-moving_momentum + moving_momentum_diff + 1e-9)

#     # DEBUG
#     # self.mu_array = tf.Print(mu_array, [mu_array, ], message="mu_array")
#     self.mu_array = mu_array

#     median_all, _ = tf.nn.top_k(mu_array, k=max(median_id, 1) , sorted=True)
#     mu_async = median_all[max(median_id - 1, 0) ]
    
#     # DEBUG
#     mu_async = tf.Print(mu_async, [mu_async, ], message="mu_async")

#     return mu_async


#   def assign_hyper_param(self, lr_val, mu_val, clip_thresh_val):
#     lr_op = self.lr_var.assign(lr_val)
#     mu_op = self.mu_var.assign(mu_val)
#     clip_thresh_op = self.clip_thresh_var.assign(clip_thresh_val / float(lr_val) )
#     self.lr_val = lr_val
#     self.mu_val = mu_val
#     self.clip_thresh_val = clip_thresh_val
#     return tf.group(lr_op, mu_op, clip_thresh_op)


#   def assign_hyper_param_value(self, lr_val, mu_val, clip_thresh_val):
#     self.lr_val = lr_val
#     # TODO change back
#     self.mu_val = mu_val
#     # self.mu_val = 0.0
#     self.clip_thresh_val = clip_thresh_val
#     return 


#   def get_min_max_curvatures(self):
#     all_curvatures = self._curv_list
#     t=len(all_curvatures)
#     W=10
#     start = max([0,t-W])
#     max_curv=max(all_curvatures[start:t])
#     min_curv=min(all_curvatures[start:t])
#     return max_curv, min_curv


#   def set_alpha(self, alpha):
#     self._alpha = alpha


#   def set_slow_start_iters(self, slow_start_iters):
#     self._slow_start_iters = slow_start_iters

    
#   def get_lr(self):
#     # lr = self._alpha / np.sqrt(sum(self._curv_list) + 1e-6)
#     # lr = self._alpha *(min([1.0, 1/float(np.sqrt(self._slow_start_iters) )**2+1/float(np.sqrt(self._slow_start_iters) )**2*self._iter_id] ) ) / np.sqrt(sum(self._curv_list) + 1e-6)
#     lr = self._alpha *(min([1.0, 1/float(self._slow_start_iters)+1/float(self._slow_start_iters)*self._iter_id] ) ) / np.sqrt(sum(self._curv_list) + 1e-6)
#     return lr


#   def get_mu(self):
#     high_pct = self._high_pct
#     low_pct = self._low_pct
#     pct_max = np.percentile(self._grad_sum_square[self._grad_sum_square != 0], high_pct)
#     pct_min = np.percentile(self._grad_sum_square[self._grad_sum_square != 0], low_pct) 
#     dr = np.sqrt(pct_max / (pct_min + 1e-9) ) 
#     mu = ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2
#     return dr, mu


#   def on_iter_finish(self, sess, grad_vals, use_hyper_op=False):
        
#     # res = sess.run( [self.lr_var, self.mu_var, self._mu_async, self._mu_sync] )
#     res = sess.run( [self.lr_var, self.mu_var] )
#     self._lr_list.append(res[0] )
#     self._mu_list.append(res[1] )
#     # self._mu_async_list.append(res[2] )
#     # self._mu_sync_list.append(res[3] )
#     self.lr_val = res[0]
#     self.mu_val = res[1]

#     # self._accum_grad_squared_list.append(res[2] )
# #     # print "inside ", res[2]
    
    
#     # i = self._iter_id + 1
#     # w = i**(-1.0)
#     # beta_poly = w
#     # gamma = self._gamma

#     # # in case there are sparse gradient strutures
#     # for i, item in enumerate(grad_vals):
#     #   if type(item) is not np.ndarray:
#     #     tmp = np.zeros(item.dense_shape)
#     #     # note here we have duplicated vectors corresponding to the same word
#     #     np.add.at(tmp, item.indices, item.values)
#     #     grad_vals[i] = tmp.copy()

#     # grad_sum_square_new = np.hstack( [val.ravel()**2 for val in grad_vals] )
#     # self._curv_list.append(np.sum(grad_sum_square_new) )    

#     # # this max_curv_new is for clipping
#     # max_curv_new, _ = self.get_min_max_curvatures()
#     # # update 
#     # if self._max_curv == None:
#     #     self._max_curv = max_curv_new
#     # else:
#     #     self._max_curv = (beta_poly**gamma)*max_curv_new + (1-beta_poly**gamma)*self._max_curv
#     # if self._grad_sum_square is None:
#     #   self._grad_sum_square = grad_sum_square_new 
#     # else:
#     #   self._grad_sum_square += grad_sum_square_new
    
#     # if self._iter_id % self._mu_update_interval == 0 and self._iter_id > 0:
#     #   dr, mu_val = self.get_mu()
#     #   self._dr_list.append(dr)
#     # else:
#     #   mu_val = self.mu_val
#     # if self._iter_id >= 0:
#     #     lr_val = self.get_lr()
#     # else:
#     #   lr_val = self.lr_val

#     # clip_thresh_val = lr_val * np.sqrt(self._max_curv)
            
#     # # TODO tidy up capping operation
#     # if use_hyper_op:
#     #   hyper_op = self.assign_hyper_param(lr_val, min(mu_val, 0.9), min(clip_thresh_val, 1.0) )
#     # else:
#     #   self.assign_hyper_param_value(lr_val, min(mu_val, 0.9), min(clip_thresh_val, 1.0) )

#     # self._max_curv_list.append(self._max_curv)
#     # self._lr_list.append(lr_val)
#     # self._mu_list.append(min(mu_val, 0.9) )
#     # self._lr_grad_list.append(lr_val * np.sqrt(self._curv_list[-1] ) )
#     # # clip_list monitor thresh over lr * clip_thresh
#     # self._clip_list.append(min(clip_thresh_val, 1.0) )

#     self._iter_id += 1
    
#     # if use_hyper_op:
#     #   return hyper_op
#     # else:
#     #   return
    
#     # # pass
#     return
    

  def plot_curv(self, log_dir='./'):            
    plt.figure()
    # # plt.semilogy(self._lr_grad_list, label="lr * grad")
    # # plt.semilogy(self._max_curv_list, label="max curv for clip")
    # # plt.semilogy(self._clip_list, label="clip thresh")
    # plt.semilogy(self._accum_grad_squared_list, label="denominator")
    # plt.semilogy(self._curv_list, label="clip curvature")
    plt.semilogy(self._lr_list, label="lr")
    plt.semilogy(self._mu_list, label="mu")
    plt.semilogy(self._mu_async_list, label="async_mu")
    plt.semilogy(self._mu_sync_list, label="sync_mu")

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

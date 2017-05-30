import numpy as np
from math import ceil, floor
import tensorflow as tf
from tensorflow.python.training import momentum
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops


class YFOptimizer(object):
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
        

  # def plot_curv(self, log_dir='./'):            
  #   plt.figure()
  #   # # plt.semilogy(self._lr_grad_list, label="lr * grad")
  #   # # plt.semilogy(self._max_curv_list, label="max curv for clip")
  #   # # plt.semilogy(self._clip_list, label="clip thresh")
  #   # plt.semilogy(self._accum_grad_squared_list, label="denominator")
  #   # plt.semilogy(self._curv_list, label="clip curvature")
  #   plt.semilogy(self._lr_list, label="lr")
  #   plt.semilogy(self._mu_list, label="mu")
  #   plt.semilogy(self._mu_async_list, label="async_mu")
  #   plt.semilogy(self._mu_sync_list, label="sync_mu")

  #   plt.title('LR='+str(self._lr_list[-1] )+' mu='+str(self._mu_list[-1] ) )
  #   plt.xlabel("iteration")
  #   plt.grid()
  #   ax = plt.subplot(111)
  #   ax.legend(loc='lower left', 
  #           ncol=2, fancybox=True, shadow=True)
  #   plt.savefig(log_dir + "/fig_" + str(self._id) + ".png")
  #   plt.close()
  #   # save monitoring quantities
  #   with open(log_dir + "/lr_grad_" + str(self._id) + ".txt", "w") as f:
  #       np.savetxt(f, np.array(self._lr_grad_list) )
  #   with open(log_dir + "/max_curv_" + str(self._id) + ".txt", "w") as f:
  #       np.savetxt(f, np.array(self._max_curv_list) )
  #   with open(log_dir + "/clip_thresh_" + str(self._id) + ".txt", "w") as f:
  #       np.savetxt(f, np.array(self._clip_list) )
  #   with open(log_dir + "/lr_" + str(self._id) + ".txt", "w") as f:
  #       np.savetxt(f, np.array(self._lr_list) )
  #   with open(log_dir + "/mu_" + str(self._id) + ".txt", "w") as f:
  #       np.savetxt(f, np.array(self._mu_list) )  
  #   with open(log_dir + "/mu_" + str(self._id) + ".txt", "w") as f:
  #       np.savetxt(f, np.array(self._dr_list) )

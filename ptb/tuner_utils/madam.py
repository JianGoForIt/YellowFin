"""
Madam optimizer for Tensorflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import momentum

import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil


class MadamOptimizer(momentum.MomentumOptimizer):
  """
  Optimizer that implements the Madam algorithm
  """

  def __init__(self, learning_rate=0.001, momentum=0.9, clip_thresh=100.0, 
               use_locking=False, name="Madam", window_size=10, 
               beta_m=0.9, beta_p=0.999):
    super(MadamOptimizer, self).__init__(tf.Variable(learning_rate, trainable=False), 
                                         tf.Variable(momentum, trainable=False), 
                                         use_locking, name, use_nesterov=False)
    self._learning_rate_val = learning_rate
    self._momentum_val = momentum
    self._clip_thresh = clip_thresh
    self._max_curv = None
    self._min_curv = None
    self._curv_window = []
    self._window_size = window_size
    self._beta_m = beta_m
    self._beta_p = beta_p
    self._iter_id = 0
    self._dr = 0
    self._dist_to_opt = 0
    self._grad_var = 0
    self._grad_rave = None

    # for debugging
    self.max_curv_list = []
    self.min_curv_list = []
    self.curv_list = []
    self.clip_thresh_list = []
    self.update_norm_list = []
    self.grad_var_list = []
    self.dist_to_opt_list = []
    self.lr_list = []
    self.mom_list = []
    self.dr_list = []

    
  def on_iter_finish(self, grads, display_interval=500):
    '''
    update learning rate and momentum
    '''
    # get gradient norm and projected gradient norm
    val_list = []
    for item in grads:
      if type(item) is not np.ndarray:
        tmp = np.zeros(item.dense_shape)
        tmp[item.indices, :] = item.values
        val_list += tmp.flatten().tolist()
      else:
        val_list += item.flatten().tolist()
    val_list = np.array(val_list)
    grad_norm = np.linalg.norm(val_list)
    
    if self._iter_id == 0:
      self._grad_rave = val_list.copy()
      self._grad_var = 0
      self._dist_to_opt = 1 / np.linalg.norm(self._grad_rave)
    else:
      self._grad_rave = self._grad_rave * self._beta_m + (1 - self._beta_m) * val_list
      self._grad_var = self._grad_var * self._beta_m + (1 - self._beta_m) * np.sum( (val_list - self._grad_rave)**2)       
      self._dist_to_opt = self._dist_to_opt * self._beta_m + (1 - self._beta_m) * (1.0 / np.linalg.norm(self._grad_rave) )
        
    self._curv_window.append(grad_norm**2)
    learning_rate, momentum, clip_thresh = self._get_lr_and_mu_rave_noisy(self._curv_window)
    beta_p = self._beta_p
    self._learning_rate_val = beta_p * self._learning_rate_val + (1 - beta_p) * learning_rate
    self._momentum_val = beta_p * self._momentum_val + (1 - beta_p) * momentum
    self._clip_thresh = beta_p * self._clip_thresh + (1 - beta_p) * clip_thresh
    self._iter_id += 1
    
    # evaluate the value to the hyper parameter tensor
#     print "inside ", sess.run( [self._learning_rate] )
#     sess.run( [self._learning_rate.assign(self._learning_rate_val),
#                                 self._momentum.assign(self._momentum_val) ] )\
    hyper_op = tf.group(self._learning_rate.assign(self._learning_rate_val),
                        self._momentum.assign(self._momentum_val) )
    
    # for debugging
    self.max_curv_list.append(self._max_curv)
    self.min_curv_list.append(self._min_curv)
    self.curv_list.append(grad_norm**2)
    self.clip_thresh_list.append(self._clip_thresh)
#     self.momentum_list.append(self._momentum_val)
#     self.learning_rate_list.append(self._learning_rate_val)
    self.update_norm_list.append(self._learning_rate_val * grad_norm)
    self.grad_var_list.append(self._grad_var)
    self.dist_to_opt_list.append(self._dist_to_opt)
    # smoothed prediction of lr and mu
    self.lr_list.append(self._learning_rate_val)
    self.mom_list.append(self._momentum_val)
    # noisy prediction of lr and mu
#     self.lr_list.append(learning_rate)
#     self.mom_list.append(momentum)

    self.dr_list.append(self._dr)
    
    if self._iter_id % display_interval == 0 and self._iter_id != 0:
      plt.figure()
      plt.semilogy(self.curv_list, label="local curvature")
      plt.semilogy(self.max_curv_list, label="max curv in win")
      plt.semilogy(self.min_curv_list, label="min curv in win")
      plt.semilogy(self.clip_thresh_list, label="clip thresh")
      plt.semilogy(self.update_norm_list, label="lr * grad norm")
      plt.semilogy(self.grad_var_list, label="grad var")
      plt.semilogy(self.dist_to_opt_list, label="dist to opt")
      plt.title("On local curvature")
      plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
      plt.grid()
      plt.show()

      plt.figure()
      plt.semilogy(self.lr_list, label="lr min")
      plt.semilogy(self.dr_list, label="dynamic range")
      plt.semilogy(self.mom_list, label="mu")
      plt.grid()
      plt.legend(loc="lower left")
      plt.show()
    
    return hyper_op
    

  def _get_lr_and_mu_rave_noisy(self, H_curv, percentile_lower=0.0, percentile_upper=1.0, thresh_fac=5.0):
    '''
    currently 1/ median_curvature is used as the learning rate rule.
    The function calls this function guarantee there are at least 2 elements in H_curv
    and the largest element is different from the smallest value.
    '''
    window_size = self._window_size
    max_curv = self._max_curv
    min_curv = self._min_curv
    dr = self._dr
    beta = self._beta_m
    dist_to_opt = self._dist_to_opt
    grad_var = self._grad_var

    start = max(0, len(H_curv) - window_size)
    H_curv_orig = H_curv[start:]
    # sort for Percentile 
    H_curv = sorted(H_curv_orig)
        
    # this is the case we first call this function
    if self._iter_id == 0:
      cutoff_min = len(H_curv) - 1
      cutoff_max = len(H_curv)
      max_curv = np.nanmax(H_curv)
      min_curv = np.nanmin(H_curv)
      
      root = 0
    else:
      if len(H_curv) < window_size:
        # force the estimation to follow trend at the begining
        cutoff_min = len(H_curv) - 1
        cutoff_max = len(H_curv)
        
        min_curv = np.nanmin(H_curv_orig[cutoff_min:cutoff_max] )
        max_curv = np.nanmax(H_curv_orig[cutoff_min:cutoff_max] )
      else:
        # cut outlier
        if np.any(H_curv > (max_curv * thresh_fac) ):
            H_curv = np.where(H_curv > (max_curv * thresh_fac), H_curv, max_curv * thresh_fac)
        if np.any(H_curv < (min_curv / thresh_fac) ):
            H_curv = np.where(H_curv < (min_curv / thresh_fac), H_curv, min_curv / thresh_fac)

        cutoff_min = int(floor(percentile_lower * window_size) )
        cutoff_max = int(ceil(percentile_upper * window_size) )
        
        max_curv = beta * max_curv + (1 - beta) * np.nanmax(H_curv[cutoff_min:cutoff_max])
        min_curv = beta * min_curv + (1 - beta) * np.nanmin(H_curv[cutoff_min:cutoff_max])
  
      const_fact = dist_to_opt**2 * min_curv**2 / 2 / grad_var
      coef = [-1, 3, -(3 + const_fact), 1]
      roots = np.roots(coef)
      roots = roots[np.real(roots) > 0]
      roots = roots[np.real(roots) < 1]
      root = roots[np.argmin(np.imag(roots) ) ]
      assert root > 0 and root < 1 and np.absolute(root.imag) < 1e-6
    
    dr = max_curv / min_curv
    assert max_curv >= min_curv
    mu = max( ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2, root**2)
        
    lr_min = (1 - np.sqrt(mu) )**2 / min_curv
    lr_max = (1 + np.sqrt(mu) )**2 / max_curv
#     assert np.abs(lr_max - lr_min) / np.abs(lr_max) <= 1e-9
    lr = lr_min
    clip_norm_base = lr * np.sqrt(max_curv)

    self._max_curv = max_curv
    self._min_curv = min_curv
    self._dr = dr

    return lr, mu, clip_norm_base



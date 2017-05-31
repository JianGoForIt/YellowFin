import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
from scipy.optimize import minimize
import tensorflow as tf

def get_min_max_curvatures(all_curvatures):
    t=len(all_curvatures)
    W=20
    start = max([0,t-W])

    wpct99=max(all_curvatures[start:t])
    wpct01=min(all_curvatures[start:t])
    
    return wpct99, wpct01


def tune_adagradier(alpha, local_curv_list, gmin, gmax):
    # First tune based on dynamic range
    dr=gmax/gmin
    mustar=((np.sqrt(dr)-1)/(np.sqrt(dr)+1))**2
    alpha_star = (1+np.sqrt(mustar))**2/gmax
    
    alpha_star = alpha/np.sqrt(sum(local_curv_list)+1e-6)
 
    return alpha_star, mustar


def get_lr_and_mu_rave(list_curv, gmin, gmax):
    '''
    currently 1/ median_curvature is used as the learning rate rule.
    The function calls this function guarantee there are at least 2 elements in H_curv
    and the largest element is different from the smallest value.
    '''
    max_curv, min_curv = gmax, gmin
    
    dr = max_curv / min_curv
    # assert max_curv >= min_curv
    alpha = 10.0
    lr, mu = tune_adagradier(alpha, list_curv, min_curv, max_curv)
    clip_norm_base = lr * np.sqrt(max_curv)
    
    
    # clip_norm_base *= 0.01
    # print "inside ", lr, mu, min_curv, max_curv, clip_norm_base
    

    return lr, mu, clip_norm_base, dr, max_curv, min_curv, 0.0, \
        0.0, 0.0
    
    
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 


def train_rave(num_iter, lr_val, mom_val, 
               clip_norm_base, do_pred, curv_beta, param_beta, 
               single_step_func, sliding_win_width=10, display_interval=200, log_dir='./', tune_interval=10):
    '''
    single_step_func is a single step for forward backward.
    '''
        
    loss_list = []
    g_norm_list = []
    local_curv_list = []
    
    lr_list = []
    mu_list = []
    dr_list = []
    max_curv_list = []
    min_curv_list = []

    clip_norm_base_list = []
    rprod_mu_list = []
    
    # init variables 
    offset = 0
    max_curv = 0.0
    min_curv = 0.0
    med_curv = None
    dr = 1.0
    mu = 0.0
    
    grad_sum_square = None
    
    train_var = tf.trainable_variables()
    
    if False: #iter_id == 0:
       with open("for_lr_tuning4000", "r") as f:
           var_val = pickle.load(f)
       eval_ops = []
       for v, v_val in zip(train_var, var_val):
           eval_ops.append(v.assign(v_val) )
       sess.run(eval_ops)
    
    for iter_id in range(num_iter):
        # run a single step
        grads, g_norm, loss = single_step_func(lr_val, mom_val, clip_norm_base, iter_id)
            
        # get gradient norm and projected gradient norm
        val_list = []
        for item in grads:
            val_list += item.flatten().tolist()
        local_curv_list.append(np.sum( (np.array(val_list)**2) ) )
        
        i = len(local_curv_list)
        w = i**(-1.0)
        gamma2=0.7

        beta_poly = w
        
        if grad_sum_square is None:
            grad_sum_square = np.array(val_list)**2
        else:
            grad_sum_square += np.array(val_list)**2

        loss_list.append(loss)
        g_norm_list.append(g_norm * lr_val)
        
        if iter_id != 0:
            max_curv_val, min_curv_val = get_min_max_curvatures(local_curv_list)
            LB_GUARD=20
            UB_GUARD=1

            max_curv = (beta_poly**gamma2)*max_curv_val*UB_GUARD + (1-beta_poly**gamma2)*max_curv
            min_curv = (beta_poly**gamma2)*min_curv_val/LB_GUARD + (1-beta_poly**gamma2)*min_curv

        
            # mu, DR is drawn from elmtwise stats in the end
            lr, _, clip_base, _, max_curv, min_curv, _, _, _ = \
                get_lr_and_mu_rave(local_curv_list, min_curv, max_curv)                
                
            lr_list.append(lr)
            dr_list.append(dr)
            mu_list.append(mu)
            max_curv_list.append(max_curv)
            min_curv_list.append(min_curv)
            
            clip_norm_base_list.append(clip_base)
            if mom_val == 0:
                rprod_mu_list.append(0)
            else:
                rprod_mu_list.append(np.log10(np.sqrt(mom_val) ) )
            if len(rprod_mu_list) > 1:
                rprod_mu_list[-1] = rprod_mu_list[-1] + rprod_mu_list[-2]
            if do_pred:
                lr_val = lr
                mom_val = mu
                clip_norm_base = clip_base
        
        if iter_id % display_interval == 0 and iter_id != 0:
            display_interval = int(np.ceil(2*display_interval))

            plt.figure()
            plt.semilogy(loss_list, '.', alpha=0.2, label="Loss")
            plt.semilogy(running_mean(loss_list,100), label="Average Loss")
#            plt.semilogy(np.cumprod(mu_list), label='mu product')
            plt.ylim( [None, 13] )
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            ax = plt.subplot(111)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=3, fancybox=True, shadow=True)
            # plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
            # plt.close()
            plt.show()
            
            plt.figure()
            plt.semilogy(local_curv_list, label="local curvature")
            plt.semilogy(max_curv_list, label="max curv in win")
            plt.semilogy(min_curv_list, label="min curv in win")
            plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
            plt.semilogy(g_norm_list, label="lr * grad norm")
            plt.grid()
            ax = plt.subplot(111)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=2, fancybox=True, shadow=True)
            # plt.savefig(log_dir + "/fig_curv_iter_" + str(iter_id) + ".pdf")
            # plt.close()
            plt.show()
            
            plt.figure()
            plt.semilogy(lr_list, label="LR")
            #plt.semilogy(dr_list, label="dynamic range")
            plt.semilogy(mu_list, label="mu")
            plt.title('LR='+str(lr_val)+' mu='+str(mom_val))
            plt.grid()
            ax = plt.subplot(111)
            ax.legend(loc='lower left', 
                  ncol=2, fancybox=True, shadow=True)
            # plt.savefig(log_dir + "/fig_hyper_iter_" + str(iter_id) + ".pdf")
            # plt.close()
            plt.show()
            
#             plt.figure()
#             plt.hist(np.log10(grad_sum_square),bins=50)
#             plt.show()
        if iter_id % tune_interval == 0 and iter_id != 0:
            elmt_pct_high=np.percentile(grad_sum_square[grad_sum_square != 0], 99.5)
            elmt_pct_low=np.percentile(grad_sum_square[grad_sum_square != 0], 0.5)
            #elmt_pct_high=max(grad_sum_square)
            #elmt_pct_low=min(grad_sum_square)
            dr = np.sqrt(elmt_pct_high/(elmt_pct_low + 1e-6) )
            mu=((np.sqrt(dr)-1)/(np.sqrt(dr)+1))**2
            #print 'DR=', dr
            #print 'mu=', mu
            
    return loss_list, g_norm_list#, grad_avg_norm_list, grad_var2_list


def train_rave_ptb(num_iter, lr_val, mom_val, 
               clip_norm_base, do_pred, curv_beta, param_beta, 
               single_step_func, sliding_win_width=10, display_interval=200, log_dir='./', tune_interval=10):
    '''
    single_step_func is a single step for forward backward.
    '''
    
    print "entry point new"
        
    loss_list = []
    g_norm_list = []
    local_curv_list = []
    
    lr_list = []
    mu_list = []
    dr_list = []
    max_curv_list = []
    min_curv_list = []

    clip_norm_base_list = []
    rprod_mu_list = []
    
    # init variables 
    offset = 0
    max_curv = 0.0
    min_curv = 0.0
    med_curv = None
    dr = 1.0
    mu = 0.0
    
    grad_sum_square = None
    
    train_var = tf.trainable_variables()
    
    if False: #iter_id == 0:
       with open("for_lr_tuning4000", "r") as f:
           var_val = pickle.load(f)
       eval_ops = []
       for v, v_val in zip(train_var, var_val):
           eval_ops.append(v.assign(v_val) )
       sess.run(eval_ops)
    
    for iter_id in range(num_iter):
        # run a single step
        grads, g_norm, loss, _, _, _ = single_step_func(lr_val, mom_val, clip_norm_base, iter_id)
            
        # get gradient norm and projected gradient norm
        val_list = []
        for item in grads:
            if type(item) is not np.ndarray:
                tmp = np.zeros(item.dense_shape)
                tmp[item.indices, :] = item.values
                val_list += tmp.flatten().tolist()
            else:
                val_list += item.flatten().tolist()
        local_curv_list.append(np.sum( (np.array(val_list)**2) ) )
        
        i = len(local_curv_list)
        w = i**(-1.0)
        gamma2=0.7

        beta_poly = w
        
        if grad_sum_square is None:
            grad_sum_square = np.array(val_list)**2
        else:
            grad_sum_square += np.array(val_list)**2

        loss_list.append(loss)
        g_norm_list.append(g_norm * lr_val)
        
        if iter_id != 0:
            max_curv_val, min_curv_val = get_min_max_curvatures(local_curv_list)
            LB_GUARD=20
            UB_GUARD=1

            max_curv = (beta_poly**gamma2)*max_curv_val*UB_GUARD + (1-beta_poly**gamma2)*max_curv
            min_curv = (beta_poly**gamma2)*min_curv_val/LB_GUARD + (1-beta_poly**gamma2)*min_curv

        
            # mu, DR is drawn from elmtwise stats in the end
            lr, _, clip_base, _, max_curv, min_curv, _, _, _ = \
                get_lr_and_mu_rave(local_curv_list, min_curv, max_curv)                
                
            lr_list.append(lr)
            dr_list.append(dr)
            mu_list.append(mu)
            max_curv_list.append(max_curv)
            min_curv_list.append(min_curv)
            
            clip_norm_base_list.append(clip_base)
            if mom_val == 0:
                rprod_mu_list.append(0)
            else:
                rprod_mu_list.append(np.log10(np.sqrt(mom_val) ) )
            if len(rprod_mu_list) > 1:
                rprod_mu_list[-1] = rprod_mu_list[-1] + rprod_mu_list[-2]
            if do_pred:
                lr_val = lr
                mom_val = mu
                clip_norm_base = clip_base
        
        if iter_id % display_interval == 0 and iter_id != 0:
            display_interval = int(np.ceil(2*display_interval))

            plt.figure()
            plt.semilogy(loss_list, '.', alpha=0.2, label="Loss")
            plt.semilogy(running_mean(loss_list,100), label="Average Loss")
#            plt.semilogy(np.cumprod(mu_list), label='mu product')
#             plt.ylim( [None, 13] )
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            ax = plt.subplot(111)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=3, fancybox=True, shadow=True)
            plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
#             plt.close()
            plt.show()
            
            plt.figure()
            plt.semilogy(local_curv_list, label="local curvature")
            plt.semilogy(max_curv_list, label="max curv in win")
            plt.semilogy(min_curv_list, label="min curv in win")
            plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
            plt.semilogy(g_norm_list, label="lr * grad norm")
            plt.grid()
            ax = plt.subplot(111)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=2, fancybox=True, shadow=True)
            plt.savefig(log_dir + "/fig_curv_iter_" + str(iter_id) + ".pdf")
#             plt.close()
            plt.show()
            
            plt.figure()
            plt.semilogy(lr_list, label="LR")
            #plt.semilogy(dr_list, label="dynamic range")
            plt.semilogy(mu_list, label="mu")
            plt.title('LR='+str(lr_val)+' mu='+str(mom_val))
            plt.grid()
            ax = plt.subplot(111)
            ax.legend(loc='lower left', 
                  ncol=2, fancybox=True, shadow=True)
            plt.savefig(log_dir + "/fig_hyper_iter_" + str(iter_id) + ".pdf")
#             plt.close()
            plt.show()
            
#             plt.figure()
#             plt.hist(np.log10(grad_sum_square),bins=50)
#             plt.show()
        if iter_id % tune_interval == 0 and iter_id != 0:
            elmt_pct_high=np.percentile(grad_sum_square[grad_sum_square != 0], 99.5)
            elmt_pct_low=np.percentile(grad_sum_square[grad_sum_square != 0], 0.5)
            #elmt_pct_high=max(grad_sum_square)
            #elmt_pct_low=min(grad_sum_square)
            dr = np.sqrt(elmt_pct_high/(elmt_pct_low + 1e-6) )
            mu=((np.sqrt(dr)-1)/(np.sqrt(dr)+1))**2
            #print 'DR=', dr
            #print 'mu=', mu
            
    return loss_list, g_norm_list, [], [], []#, grad_avg_norm_list, grad_var2_list






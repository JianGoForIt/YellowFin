import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor

def get_lr_and_mu_rave(H_curv, window_size, max_curv, min_curv, dr, beta, 
                       percentile_lower=0.0, percentile_upper=1.0, thresh_fac=10.0):
    '''
    currently 1/ median_curvature is used as the learning rate rule.
    The function calls this function guarantee there are at least 2 elements in H_curv
    and the largest element is different from the smallest value.
    '''
    start = max(0, len(H_curv) - window_size)
    H_curv_orig = H_curv[start:]
    # sort for Percentile 
    H_curv = sorted(H_curv_orig)
        
    # this is the case we first call this function
    if max_curv == None or min_curv == None:
        cutoff_min = len(H_curv) - 1
        cutoff_max = len(H_curv)
        max_curv = np.nanmax(H_curv)
        min_curv = np.nanmin(H_curv)
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
    
    dr = max_curv / min_curv
    assert max_curv >= min_curv
    mu = ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2
    lr_min = (1 - np.sqrt(mu) )**2 / min_curv
    lr_max = (1 + np.sqrt(mu) )**2 / max_curv
    assert np.abs(lr_max - lr_min) / np.abs(lr_max) <= 1e-9
    lr = lr_max
    clip_norm_base = lr * np.sqrt(max_curv)
#     print("max-H ", np.max(H_curv), " min-H ", np.min(H_curv), " med-H ", np.median(H_curv),
#           "d range ", dr, "lr_min ", lr_min, "lr_max ", lr_max, "lr_med ", lr_med, " mu ", mu)
    return lr, mu, clip_norm_base, dr, max_curv, min_curv, np.nanmax(H_curv[cutoff_min:cutoff_max] ), \
        np.nanmin(H_curv[cutoff_min:cutoff_max] ), np.nanmedian(H_curv)


def get_lr_and_mu_rave_noisy(H_curv, window_size, max_curv, min_curv, dr, beta, dist_to_opt, grad_var,  
                       percentile_lower=0.01, percentile_upper=0.99, thresh_fac=5.0):
    '''
    currently 1/ median_curvature is used as the learning rate rule.
    The function calls this function guarantee there are at least 2 elements in H_curv
    and the largest element is different from the smallest value.
    '''
    
    t=len(H_curv)
    start = int(np.floor(0.5*t))
    start = max([start, max([0,t-window_size])])    
    max_curv_new = np.percentile(H_curv[start:t], percentile_upper * 100)
    min_curv_new = np.percentile(H_curv[start:t], percentile_lower * 100)
    
#    start = max(0, len(H_curv) - window_size)
#    H_curv_orig = H_curv[start:]
#    # sort for Percentile 
#    H_curv = sorted(H_curv_orig)
        
    # this is the case we first call this function
#    if max_curv == None or min_curv == None:
#        cutoff_min = len(H_curv) - 1
#        cutoff_max = len(H_curv)
#        max_curv = np.nanmax(H_curv)
#        min_curv = np.nanmin(H_curv)
#    else:
#         if len(H_curv) < window_size:
#             # force the estimation to follow trend at the begining
#             cutoff_min = len(H_curv) - 1
#             cutoff_max = len(H_curv)
            
#             min_curv = np.nanmin(H_curv_orig[cutoff_min:cutoff_max] )
#             max_curv = np.nanmax(H_curv_orig[cutoff_min:cutoff_max] )
#         else:
#             # cut outlier
#             if np.any(H_curv > (max_curv * thresh_fac) ):
#                 H_curv = np.where(H_curv > (max_curv * thresh_fac), H_curv, max_curv * thresh_fac)
#             if np.any(H_curv < (min_curv / thresh_fac) ):
#                 H_curv = np.where(H_curv < (min_curv / thresh_fac), H_curv, min_curv / thresh_fac)

#             cutoff_min = int(floor(percentile_lower * window_size) )
#             cutoff_max = int(ceil(percentile_upper * window_size) )
            
#             max_curv = beta * max_curv + (1 - beta) * np.nanmax(H_curv[cutoff_min:cutoff_max])
#             min_curv = beta * min_curv + (1 - beta) * np.nanmin(H_curv[cutoff_min:cutoff_max])
    if max_curv != None and min_curv != None:
        max_curv = beta * max_curv + (1 - beta) * max_curv_new
        min_curv = beta * min_curv + (1 - beta) * min_curv_new
    else:
        max_curv = max_curv_new
        min_curv = min_curv_new
    
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
    
#     print "test root ", ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2, root, root**2
    
    lr_min = (1 - np.sqrt(mu) )**2 / min_curv
    lr_max = (1 + np.sqrt(mu) )**2 / max_curv
#     assert np.abs(lr_max - lr_min) / np.abs(lr_max) <= 1e-9
    lr = lr_min
    clip_norm_base = lr * np.sqrt(max_curv)
    
    # estimate gradient variance
    
#     print("max-H ", np.max(H_curv), " min-H ", np.min(H_curv), " med-H ", np.median(H_curv),
#           "d range ", dr, "lr_min ", lr_min, "lr_max ", lr_max, "lr_med ", lr_med, " mu ", mu)
#    return lr, mu, clip_norm_base, dr, max_curv, min_curv, np.nanmax(H_curv[cutoff_min:cutoff_max] ), \
#        np.nanmin(H_curv[cutoff_min:cutoff_max] ), np.nanmedian(H_curv)
    return lr, mu, clip_norm_base, dr, max_curv, min_curv, np.nanmax(H_curv[start:t] ), \
        np.nanmin(H_curv[start:t] ), np.nanmedian(H_curv[start:t])
    
    
def train_rave(num_iter, lr_val, mom_val, 
               clip_norm_base, do_pred, curv_beta, param_beta, 
               single_step_func, sliding_win_width=10, display_interval=200):
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
    noisy_max_curv_list = []
    noisy_min_curv_list = []
    noisy_med_curv_list = []
    
    clip_norm_base_list = []
    rprod_mu_list = []
    
    # init variables 
    offset = 0
    max_curv = None
    min_curv = None
    med_curv = None
    dr = 1.0
    
    for iter_id in range(num_iter):
        # run a single step
        grads, g_norm, loss = single_step_func(lr_val, mom_val, clip_norm_base, iter_id)
            
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
        
        loss_list.append(loss)
        g_norm_list.append(g_norm * lr_val)
        
        if iter_id != 0:
            lr, mu, clip_base, dr, max_curv, min_curv, noisy_max_curv, noisy_min_curv, noisy_med_curv = \
                get_lr_and_mu_rave(local_curv_list, sliding_win_width, max_curv, min_curv, dr, curv_beta)
        
            lr_list.append(lr)
            dr_list.append(dr)
            mu_list.append(mu)
            max_curv_list.append(max_curv)
            min_curv_list.append(min_curv)
            noisy_max_curv_list.append(noisy_max_curv)
            noisy_min_curv_list.append(noisy_min_curv)
            noisy_med_curv_list.append(noisy_med_curv)
            clip_norm_base_list.append(clip_base)
            if mom_val == 0:
                rprod_mu_list.append(0)
            else:
                rprod_mu_list.append(np.log10(np.sqrt(mom_val) ) )
            if len(rprod_mu_list) > 1:
                rprod_mu_list[-1] = rprod_mu_list[-1] + rprod_mu_list[-2]
        
            if do_pred:
#                 print param_beta
                lr_val = lr_val * param_beta + lr * (1 - param_beta)
                mom_val = mom_val * param_beta + mu * (1 - param_beta)
                clip_norm_base = clip_norm_base * param_beta + clip_base * (1 - param_beta)
        
        if iter_id % display_interval == 0 and iter_id != 0:
            plt.figure()
            plt.plot(loss_list, label="loss")
#             plt.semilogy(np.power(10, rprod_mu_list - np.max(rprod_mu_list) ), label="running product new")
            plt.ylim( [None, 13] )
            plt.title("loss w.r.t \# iter")
            plt.legend()
            plt.grid()
            plt.show()
            
            plt.figure()
            plt.semilogy(local_curv_list, label="local curvature")
            plt.semilogy(max_curv_list, label="max curv in win")
            plt.semilogy(min_curv_list, label="min curv in win")
            plt.semilogy(clip_norm_base_list, label="clip thresh")
            plt.semilogy(g_norm_list, label="lr * grad norm")
            plt.title("On local curvature")
            plt.legend(loc="lower left")
            plt.grid()
            plt.show()
            
            plt.figure()
            plt.semilogy(g_norm_list, label="lr * grad norm")
            plt.title("On local curvature")
            plt.legend(loc="lower left")
            plt.grid()
            plt.show()
            
            
            plt.figure()
            plt.semilogy(lr_list, label="lr min")
            plt.semilogy(dr_list, label="dynamic range")
            plt.semilogy(mu_list, label="mu")
            plt.grid()
            plt.legend(loc="lower left")
            plt.show()
            
    return loss_list, g_norm_list
    
    
def train_rave_noisy(num_iter, lr_val, mom_val, 
               clip_norm_base, do_pred, curv_beta, param_beta, grad_beta, dist_beta,
               single_step_func, sliding_win_width=10, display_interval=200):
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
    noisy_max_curv_list = []
    noisy_min_curv_list = []
    noisy_med_curv_list = []
    
    clip_norm_base_list = []
    rprod_mu_list = []
    dist_to_opt_list = []
    rvar_grad_list = []
    
    # init variables 
    offset = 0
    max_curv = None
    min_curv = None
    med_curv = None
    dr = 1.0
    
    for iter_id in range(num_iter):
        # run a single step
        grads, g_norm, loss = single_step_func(lr_val, mom_val, clip_norm_base, iter_id)
            
        # get gradient norm and projected gradient norm
        val_list = []
        for item in grads:
            val_list += item.flatten().tolist()
        val_list = np.array(val_list)
        local_curv_list.append(np.sum( (val_list)**2) )
        
        loss_list.append(loss)
        g_norm_list.append(g_norm * lr_val)
        
        # for distance to opt estimation: we utilize f'=F * (x - x*)
        if iter_id == 0:
            rave_grad = val_list
            grad_var = 0
            dist_to_opt = 1 / np.linalg.norm(rave_grad)
        else:
            rave_grad = rave_grad * grad_beta + (1 - grad_beta) * val_list
            grad_var = grad_var * grad_beta + (1 - grad_beta) * np.sum( (val_list - rave_grad)**2)            
            dist_to_opt = dist_to_opt * dist_beta + (1 - dist_beta) * (1.0 / np.linalg.norm(rave_grad) )
        rvar_grad_list.append(grad_var)
        dist_to_opt_list.append(dist_to_opt)
        
        
        if iter_id != 0:
            lr, mu, clip_base, dr, max_curv, min_curv, noisy_max_curv, noisy_min_curv, noisy_med_curv = \
                get_lr_and_mu_rave_noisy(local_curv_list, sliding_win_width, max_curv, min_curv, dr, curv_beta,
                                        dist_to_opt, grad_var)
        
            # lr_list.append(lr)
            # dr_list.append(dr)
            # mu_list.append(mu)
            max_curv_list.append(max_curv)
            min_curv_list.append(min_curv)
            noisy_max_curv_list.append(noisy_max_curv)
            noisy_min_curv_list.append(noisy_min_curv)
            noisy_med_curv_list.append(noisy_med_curv)
            clip_norm_base_list.append(clip_base)
                        
            if mom_val == 0:
                rprod_mu_list.append(0)
            else:
                rprod_mu_list.append(np.log10(np.sqrt(mom_val) ) )
            if len(rprod_mu_list) > 1:
                rprod_mu_list[-1] = rprod_mu_list[-1] + rprod_mu_list[-2]
        
            if do_pred:
                #if iter_id < 500:
                if True:
                    lr_val = lr_val * 0.5 + lr * (1 - 0.5)
                    mom_val = mom_val * 0.5 + mu * (1 - 0.5)
                    clip_norm_base = clip_norm_base * 0.5 + clip_base * (1 - 0.5)
                else:
                    lr_val = lr_val * param_beta + lr * (1 - param_beta)
                    mom_val = mom_val * param_beta + mu * (1 - param_beta)
                    clip_norm_base = clip_norm_base * param_beta + clip_base * (1 - param_beta)
            lr_list.append(lr_val)
            dr_list.append(dr)
            mu_list.append(mom_val)
        
        if iter_id % display_interval == 0 and iter_id != 0:
            plt.figure()
            plt.semilogy(loss_list, label="loss")
#             plt.semilogy(np.power(10, rprod_mu_list - np.max(rprod_mu_list) ), label="running product new")
            # plt.ylim( [None, 13] )
            plt.title("loss w.r.t \# iter")
            plt.legend()
            plt.grid()
            plt.show()
            
            plt.figure()
            plt.semilogy(local_curv_list, label="local curvature")
            plt.semilogy(max_curv_list, label="max curv in win")
            plt.semilogy(min_curv_list, label="min curv in win")
            plt.semilogy(clip_norm_base_list, label="clip thresh")
#             plt.semilogy(noisy_max_curv_list, label="max curv noisy")
#             plt.semilogy(noisy_min_curv_list, label="min curv noisy")
#             plt.semilogy(noisy_med_curv_list, label="med curv noisy")
            plt.semilogy(g_norm_list, label="lr * grad norm")
            plt.semilogy(rvar_grad_list, label="grad var")
            plt.semilogy(dist_to_opt_list, label="dist to opt")
            plt.title("On local curvature")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid()
            plt.show()
            
            plt.figure()
            plt.semilogy(lr_list, label="lr min")
            plt.semilogy(dr_list, label="dynamic range")
            plt.semilogy(mu_list, label="mu")
            plt.grid()
            plt.legend(loc="lower left")
            plt.show()
            
    return loss_list, g_norm_list


def smooth_loss(loss, win_size):
    win = np.ones( (win_size,) ) / float(win_size)
    return np.convolve(loss, win)
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
from scipy.optimize import minimize
import sys
sys.path.append('../scripts')
from resnet_utils import evaluate

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def get_min_max_curvatures(all_curvatures):
    t=len(all_curvatures)
    # Just window
    W=20
    start = max([0,t-W])
    wpct99=np.percentile(all_curvatures[start:t], 100)
    wpct01=np.percentile(all_curvatures[start:t], 0)
    
    return wpct99, wpct01


def tune_everything(x0squared, C, T, gmin, gmax):
    # First tune based on dynamic range    
    if C==0:
        dr=gmax/gmin
        mustar=((np.sqrt(dr)-1)/(np.sqrt(dr)+1))**2
        alpha_star = (1+np.sqrt(mustar))**2/gmax
        
        return alpha_star,mustar

    dist_to_opt = x0squared
    grad_var = C
    max_curv = gmax
    min_curv = gmin
    
    const_fact = dist_to_opt * min_curv**2 / 2 / grad_var
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
    
    alpha_star = lr_min
    mustar = mu

    return alpha_star, mustar


def get_lr_and_mu_rave_noisy(list_curv, x0squared, grad_var, T, gmin, gmax):
    '''
    currently 1/ median_curvature is used as the learning rate rule.
    The function calls this function guarantee there are at least 2 elements in H_curv
    and the largest element is different from the smallest value.
    '''
    max_curv, min_curv = gmax, gmin
    
    dr = max_curv / min_curv
    assert max_curv >= min_curv
    lr, mu = tune_everything(x0squared, grad_var, T, min_curv, max_curv)
    clip_norm_base = lr * np.sqrt(max_curv)

    return lr, mu, clip_norm_base, dr, max_curv, min_curv, 0.0, 0.0, 0.0


def train_rave_noisy_acc(mon_sess, model_eval, num_iter, lr_val, mom_val, 
               clip_norm_base, do_pred, curv_beta, param_beta, 
               single_step_func, sliding_win_width=10, display_interval=200, test_interval=250, log_dir='./'):
    '''
    single_step_func is a single step for forward backward.
    '''
    
    # TODO remove print
    print "test 0.06"
    
    T=1
    
    precision_list = []
    best_precision = 0.0
    
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
    grad_avg_norm_list = []
    grad_var2_list = []    
    clip_norm_base_list = []
    
    grad_avg = None
    
    dist_star_estimate_list = []
    
    # init variables 
    offset = 0
    max_curv = 0.0
    min_curv = 0.0
    med_curv = None
    dr = 1.0
    gamma=1.0
    
    for iter_id in range(num_iter):
        # run a single step
        grads, loss = single_step_func(lr_val, mom_val, clip_norm_base)
            
        # get gradient norm and projected gradient norm
        val_list = []
        for item in grads:
            val_list += item.flatten().tolist()
        local_curv_list.append(np.sum( (np.array(val_list)**2) ) )
        
        i = len(local_curv_list)
        w = i**(-1.0)

        # beta_poly = w
        beta_poly = 1 - curv_beta
        # print 1 - beta_poly**gamma

        val_list = np.array(val_list)
        if grad_avg is None:
            grad_avg = val_list.copy()
            grad_mean_square = val_list**2
            grad_norm_avg = np.linalg.norm(val_list)
        else:
            grad_avg = (beta_poly**gamma)*val_list + (1-beta_poly**gamma)*grad_avg
            grad_mean_square = (beta_poly**gamma) * val_list ** 2 + (1-beta_poly**gamma) * grad_mean_square
            grad_norm_avg = (beta_poly**gamma) * np.linalg.norm(val_list) + (1-beta_poly**gamma) * grad_norm_avg

        grad_avg_norm_squared = np.sum(grad_avg**2)
        grad_avg_norm_list.append(grad_avg_norm_squared )
        
        grad_var2_list.append(sum(grad_mean_square - grad_avg**2))

        loss_list.append(loss)
        g_norm_list.append(np.linalg.norm(val_list) * lr_val)
        
        if iter_id != 0:
            max_curv_val, min_curv_val = get_min_max_curvatures(local_curv_list)
            # Smoothing the geometric mean and DR instead of min/max separately. Should help with lag
            max_curv = (beta_poly**gamma)*max_curv_val + (1-beta_poly**gamma)*max_curv
            min_curv = (beta_poly**gamma)*min_curv_val + (1-beta_poly**gamma)*min_curv

            # Estimate of distance from optimum
            if iter_id == 1:
                dist_star_estimate = grad_norm_avg / np.sum(grad_mean_square)
            else:
                dist_star_estimate = (beta_poly**gamma) * grad_norm_avg / np.sum(grad_mean_square) \
                    + (1-beta_poly**gamma)*dist_star_estimate
            dist_star_estimate_list.append(dist_star_estimate)   
        
            lr, mu, clip_base, dr, max_curv, min_curv, noisy_max_curv, noisy_min_curv, noisy_med_curv = \
                get_lr_and_mu_rave_noisy(local_curv_list, dist_star_estimate**2, grad_var2_list[-1], T, min_curv, max_curv)                
                
            max_curv_list.append(max_curv)
            min_curv_list.append(min_curv)
            
            noisy_max_curv_list.append(noisy_max_curv)
            noisy_min_curv_list.append(noisy_min_curv)
            noisy_med_curv_list.append(noisy_med_curv)
            clip_norm_base_list.append(clip_base)
                    
            if do_pred:
                lr_val = lr_val * (1 - beta_poly**gamma) + lr * (beta_poly**gamma)
                mom_val = mom_val * (1 - beta_poly**gamma) + mu * (beta_poly**gamma)
                clip_base = lr_val * np.sqrt(max_curv)
                clip_norm_base = clip_norm_base * (1 - beta_poly**gamma) + clip_base * (beta_poly**gamma)
                
            lr_list.append(lr_val)
            dr_list.append(dr)
            mu_list.append(mom_val)

        
        if iter_id % display_interval == 0 and iter_id != 0:
            print iter_id
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
            plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
            plt.close()
            
            plt.figure()
            plt.semilogy(local_curv_list, label="local curvature")
            plt.semilogy(max_curv_list, label="max curv in win")
            plt.semilogy(min_curv_list, label="min curv in win")
            plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
            plt.semilogy(g_norm_list, label="lr * grad norm")
            plt.title("On local curvature")
            plt.grid()
            ax = plt.subplot(111)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=2, fancybox=True, shadow=True)
            plt.savefig(log_dir + "/fig_curv_iter_" + str(iter_id) + ".pdf")
            plt.close()
            
            plt.figure()
            plt.semilogy(lr_list, label="lr min")
            plt.semilogy(dr_list, label="dynamic range")
            plt.semilogy(mu_list, label="mu")
            plt.semilogy(grad_avg_norm_list, label="Grad avg norm")
            plt.semilogy(dist_star_estimate_list, label="Est dist from opt")
            plt.semilogy(grad_var2_list, label="Grad variance")
            plt.title('LR='+str(lr_val)+' mu='+str(mom_val))
            plt.grid()
            plt.legend(loc="upper right")
            plt.savefig(log_dir + "/fig_hyper_iter_" + str(iter_id) + ".pdf")
            plt.close()
            
        if iter_id % test_interval == 0 and iter_id != 0:
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

    return loss_list, g_norm_list


# def train_rave_noisy(num_iter, lr_val, mom_val, 
#                clip_norm_base, do_pred, curv_beta, param_beta, 
#                single_step_func, sliding_win_width=10, display_interval=200, log_dir='./'):
#     '''
#     single_step_func is a single step for forward backward.
#     '''
    
#     # TODO remove print
#     print "test 0.05"
    
#     T=1
    
#     loss_list = []
#     g_norm_list = []
#     local_curv_list = []
    
#     lr_list = []
#     mu_list = []
#     dr_list = []
#     max_curv_list = []
#     min_curv_list = []
#     noisy_max_curv_list = []
#     noisy_min_curv_list = []
#     noisy_med_curv_list = []
#     grad_avg_norm_list = []
#     grad_var2_list = []    
#     clip_norm_base_list = []
    
#     grad_avg = None
    
#     dist_star_estimate_list = []
    
#     # init variables 
#     offset = 0
#     max_curv = 0.0
#     min_curv = 0.0
#     med_curv = None
#     dr = 1.0
#     gamma=1.0
    
#     for iter_id in range(num_iter):
#         # run a single step
#         grads, g_norm, loss = single_step_func(lr_val, mom_val, clip_norm_base)
            
#         # get gradient norm and projected gradient norm
#         val_list = []
#         for item in grads:
#             val_list += item.flatten().tolist()
#         local_curv_list.append(np.sum( (np.array(val_list)**2) ) )
        
#         i = len(local_curv_list)
#         w = i**(-1.0)

#         # beta_poly = w
#         beta_poly = 1 - curv_beta
#         # print 1 - beta_poly**gamma

#         val_list = np.array(val_list)
#         if grad_avg is None:
#             grad_avg = val_list.copy()
#             grad_mean_square = val_list**2
#             grad_norm_avg = np.linalg.norm(val_list)
#         else:
#             grad_avg = (beta_poly**gamma)*val_list + (1-beta_poly**gamma)*grad_avg
#             grad_mean_square = (beta_poly**gamma) * val_list ** 2 + (1-beta_poly**gamma) * grad_mean_square
#             grad_norm_avg = (beta_poly**gamma) * np.linalg.norm(val_list) + (1-beta_poly**gamma) * grad_norm_avg

#         grad_avg_norm_squared = np.sum(grad_avg**2)
#         grad_avg_norm_list.append(grad_avg_norm_squared )
        
#         grad_var2_list.append(sum(grad_mean_square - grad_avg**2))

#         loss_list.append(loss)
#         g_norm_list.append(g_norm * lr_val)
        
#         if iter_id != 0:
#             max_curv_val, min_curv_val = get_min_max_curvatures(local_curv_list)
#             # Smoothing the geometric mean and DR instead of min/max separately. Should help with lag
#             max_curv = (beta_poly**gamma)*max_curv_val + (1-beta_poly**gamma)*max_curv
#             min_curv = (beta_poly**gamma)*min_curv_val + (1-beta_poly**gamma)*min_curv

#             # Estimate of distance from optimum
#             if iter_id == 1:
#                 dist_star_estimate = grad_norm_avg / np.sum(grad_mean_square)
#             else:
#                 dist_star_estimate = (beta_poly**gamma) * grad_norm_avg / np.sum(grad_mean_square) \
#                     + (1-beta_poly**gamma)*dist_star_estimate
#             dist_star_estimate_list.append(dist_star_estimate)   
        
#             lr, mu, clip_base, dr, max_curv, min_curv, noisy_max_curv, noisy_min_curv, noisy_med_curv = \
#                 get_lr_and_mu_rave_noisy(local_curv_list, dist_star_estimate**2, grad_var2_list[-1], T, min_curv, max_curv)                
                
#             max_curv_list.append(max_curv)
#             min_curv_list.append(min_curv)
            
#             noisy_max_curv_list.append(noisy_max_curv)
#             noisy_min_curv_list.append(noisy_min_curv)
#             noisy_med_curv_list.append(noisy_med_curv)
#             clip_norm_base_list.append(clip_base)
                    
#             if do_pred:
#                 lr_val = lr_val * (1 - beta_poly**gamma) + lr * (beta_poly**gamma)
#                 mom_val = mom_val * (1 - beta_poly**gamma) + mu * (beta_poly**gamma)
#                 clip_base = lr_val * np.sqrt(max_curv)
#                 clip_norm_base = clip_norm_base * (1 - beta_poly**gamma) + clip_base * (beta_poly**gamma)
                
#             lr_list.append(lr_val)
#             dr_list.append(dr)
#             mu_list.append(mom_val)

        
#         if iter_id % display_interval == 0 and iter_id != 0:
#             plt.figure()
#             plt.semilogy(loss_list, '.', alpha=0.2, label="Loss")
#             plt.semilogy(running_mean(loss_list,100), label="Average Loss")
#             plt.xlabel('Iterations')
#             plt.ylabel('Loss')
#             plt.legend()
#             plt.grid()
#             ax = plt.subplot(111)
#             ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#                   ncol=3, fancybox=True, shadow=True)
#             plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
#             plt.close()
            
#             plt.figure()
#             plt.semilogy(local_curv_list, label="local curvature")
#             plt.semilogy(max_curv_list, label="max curv in win")
#             plt.semilogy(min_curv_list, label="min curv in win")
#             plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
#             plt.semilogy(g_norm_list, label="lr * grad norm")
#             plt.title("On local curvature")
#             plt.grid()
#             ax = plt.subplot(111)
#             ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#                   ncol=2, fancybox=True, shadow=True)
#             plt.savefig(log_dir + "/fig_curv_iter_" + str(iter_id) + ".pdf")
#             plt.close()
            
#             plt.figure()
#             plt.semilogy(lr_list, label="lr min")
#             plt.semilogy(dr_list, label="dynamic range")
#             plt.semilogy(mu_list, label="mu")
#             plt.semilogy(grad_avg_norm_list, label="Grad avg norm")
#             plt.semilogy(dist_star_estimate_list, label="Est dist from opt")
#             plt.semilogy(grad_var2_list, label="Grad variance")
#             plt.title('LR='+str(lr_val)+' mu='+str(mom_val))
#             plt.grid()
#             plt.legend(loc="upper right")
#             plt.savefig(log_dir + "/fig_hyper_iter_" + str(iter_id) + ".pdf")
#             plt.close()
            
#     return loss_list, g_norm_list, grad_avg_norm_list, grad_var2_list



def train_rave_noisy_ptb(num_iter, lr_val, mom_val, 
               clip_norm_base, do_pred, curv_beta, param_beta, 
               single_step_func, sliding_win_width=10, display_interval=200, log_dir='./'):
    '''
    single_step_func is a single step for forward backward.
    '''
    
    # TODO
    print "test 0.01"
    
    T=1
    
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
    grad_avg_norm_list = []
    grad_var2_list = []    
    clip_norm_base_list = []
    
    perp_list = []
    val_perp_list = []
    test_perp_list = []
    
    grad_avg = None
    
    dist_star_estimate_list = []
    
    # init variables 
    offset = 0
    max_curv = 0.0
    min_curv = 0.0
    med_curv = None
    dr = 1.0
    gamma=0.3
    
    for iter_id in range(num_iter):
        # run a single step
        grads, g_norm, loss, perp, val_perp, test_perp = single_step_func(lr_val, mom_val, clip_norm_base, iter_id)
            
        # get gradient norm and projected gradient norm
        val_list = []
        # for item in grads:
        #     if type(item) is not np.ndarray:
        #         tmp = np.zeros(item.dense_shape)
        #         tmp[item.indices, :] = item.values
        #         val_list += tmp.flatten().tolist()
        #     else:
        #         val_list += item.flatten().tolist()
        # local_curv_list.append(np.sum( (np.array(val_list)**2) ) )
        
        for i, item in emumerate(grads):
            if type(item) is not np.ndarray:
                tmp = np.zeros(item.dense_shape)
                # note here we have duplicated vectors corresponding to the same word
                np.add.at(tmp, item.indices, item.values)
                grad[i] = tmp.copy()

        val_list = np.hstack( [val.ravel() for val in grads] )

        local_curv_list.append(np.sum(val_list**2) )
        



        i = len(local_curv_list)
        w = i**(-1.0)

        beta_poly = w

        # val_list = np.array(val_list)
        if grad_avg is None:
            grad_avg = val_list.copy()
            grad_mean_square = val_list**2
            grad_norm_avg = np.linalg.norm(val_list)
        else:
            grad_avg = (beta_poly**gamma)*val_list + (1-beta_poly**gamma)*grad_avg
            grad_mean_square = (beta_poly**gamma) * val_list ** 2 + (1-beta_poly**gamma) * grad_mean_square
            grad_norm_avg = (beta_poly**gamma) * np.linalg.norm(val_list) + (1-beta_poly**gamma) * grad_norm_avg

        grad_avg_norm_squared = np.sum(grad_avg**2)
        grad_avg_norm_list.append(grad_avg_norm_squared )
        
        grad_var2_list.append(sum(grad_mean_square - grad_avg**2))

        loss_list.append(loss)
        g_norm_list.append(g_norm * lr_val)
        
        # deal with perplexity
        perp_list.append(perp)
        if val_perp != None:
            val_perp_list.append(val_perp)
        if test_perp != None:
            test_perp_list.append(test_perp)
        
        if iter_id != 0:
            max_curv_val, min_curv_val = get_min_max_curvatures(local_curv_list)
            # Smoothing the geometric mean and DR instead of min/max separately. Should help with lag
            max_curv = (beta_poly**gamma)*max_curv_val + (1-beta_poly**gamma)*max_curv
            min_curv = (beta_poly**gamma)*min_curv_val + (1-beta_poly**gamma)*min_curv

            # Estimate of distance from optimum
            if iter_id == 1:
                dist_star_estimate = grad_norm_avg / np.sum(grad_mean_square)
            else:
                dist_star_estimate = (beta_poly**gamma) * grad_norm_avg / np.sum(grad_mean_square) \
                    + (1-beta_poly**gamma)*dist_star_estimate
            dist_star_estimate_list.append(dist_star_estimate)   
        
            lr, mu, clip_base, dr, max_curv, min_curv, noisy_max_curv, noisy_min_curv, noisy_med_curv = \
                get_lr_and_mu_rave_noisy(local_curv_list, dist_star_estimate**2, grad_var2_list[-1], T, min_curv, max_curv)                
                
            max_curv_list.append(max_curv)
            min_curv_list.append(min_curv)
            
            noisy_max_curv_list.append(noisy_max_curv)
            noisy_min_curv_list.append(noisy_min_curv)
            noisy_med_curv_list.append(noisy_med_curv)
            clip_norm_base_list.append(clip_base)
                    
            if do_pred:
                lr_val = lr_val * (1 - beta_poly**gamma) + lr * (beta_poly**gamma)
                mom_val = mom_val * (1 - beta_poly**gamma) + mu * (beta_poly**gamma)
                clip_base = lr_val * np.sqrt(max_curv)
                clip_norm_base = clip_norm_base * (1 - beta_poly**gamma) + clip_base * (beta_poly**gamma)
                
            lr_list.append(lr_val)
            dr_list.append(dr)
            mu_list.append(mom_val)

        
        if iter_id % display_interval == 0 and iter_id != 0:
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
            plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
            # plt.show()
            plt.close()
            
            plt.figure()
            plt.semilogy(local_curv_list, label="local curvature")
            plt.semilogy(max_curv_list, label="max curv in win")
            plt.semilogy(min_curv_list, label="min curv in win")
            plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
            plt.semilogy(g_norm_list, label="lr * grad norm")
            plt.title("On local curvature")
            plt.grid()
            ax = plt.subplot(111)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=2, fancybox=True, shadow=True)
            plt.savefig(log_dir + "/fig_curv_iter_" + str(iter_id) + ".pdf")
            # plt.show()
            plt.close()
            
            plt.figure()
            plt.semilogy(lr_list, label="lr min")
            plt.semilogy(dr_list, label="dynamic range")
            plt.semilogy(mu_list, label="mu")
            plt.semilogy(grad_avg_norm_list, label="Grad avg norm")
            plt.semilogy(dist_star_estimate_list, label="Est dist from opt")
            plt.semilogy(grad_var2_list, label="Grad variance")
            plt.title('LR='+str(lr_val)+' mu='+str(mom_val))
            plt.grid()
            plt.legend(loc="upper right")
            plt.savefig(log_dir + "/fig_hyper_iter_" + str(iter_id) + ".pdf")
            # plt.show()
            plt.close()
            
    return loss_list, g_norm_list, perp_list, val_perp_list, test_perp_list

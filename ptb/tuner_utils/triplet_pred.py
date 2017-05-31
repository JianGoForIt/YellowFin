import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from scipy.optimize import curve_fit

def curve(x, k1, y1, dk, y2, a1, a2, a3):
    '''
    the function defining the triplet quadratics function
    '''
    k2 = k1+dk
    
    # Get parameters for middle quadratic
    b2 = (a2*(k2**2-k1**2) + y1 - y2)/(k1-k2)
    c2 = y1 - a2*k1**2 - b2*k1
    
    # Get parameters for left quadratic
    b1 = 2*(a2-a1)*k1+b2
    c1 = y1 - a1*k1**2 - b1*k1
    
    # Get parameters for left quadratic
    b3 = 2*(a2-a3)*k2+b2
    c3 = y2 - a3*k2**2 - b3*k2

    q1 = a1*np.power(x,2)+b1*x+c1    
    q2 = a2*np.power(x,2)+b2*x+c2
    q3 = a3*np.power(x,2)+b3*x+c3   
    
    result = q2
    
    result[x<=k1]=q1[x<=k1]
    result[x>=k2]=q3[x>=k2]
    
    return result


def get_losses_from_fit(all_alphas, all_losses, alpha_range, maxfev=None):
    '''
    using the given loss to fit the triplet-quadratic function
    '''
    p0 = 7*[0]
    lb = 7*[None]
    ub = 7*[None]

    alpha_lb=alpha_range[0]
    alpha_ub=alpha_range[1]

    # Basic level
    y1=y2=np.mean(all_losses)

    # Left knee
    k1=max(-0.5, alpha_lb)
    p0[0]=k1
    lb[0]=alpha_lb
    ub[0]=alpha_ub

    p0[1]=y1
    lb[1]=0
    ub[1]=y1

    # Right knee delta
    k2=min(1.0, alpha_ub)
    p0[2]=k2-k1
    lb[2]=alpha_lb
    ub[2]=alpha_ub

    p0[3]=y2
    lb[3]=0
    ub[3]=y2


    # (half)Curvatures for the three quadratics
    a = 7

    p0[4:7]=[a,1.0,a]
    lb[4:7]=[0.0, 0.0, 0.0]
    ub[4:7]=3*[100]
    
    if maxfev != None:
        popt, pcov = curve_fit(
            curve, all_alphas, all_losses,
            p0=p0, bounds=(lb, ub)
        )
    else:
        popt, pcov = curve_fit(
            curve, all_alphas, all_losses,
            p0=p0, bounds=(lb, ub), maxfev=maxfev
        )

    all_losses_fit = curve(all_alphas, *popt)    

    return np.array(all_losses_fit), popt


# def get_obj_slice_val(model, sess, train_var_init, train_var_final, n_points, alpha_range, n_batch=1):
#     '''
#     n_batch indicates how many minibatch to use obj evaluation on each point
#     '''
#     alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
#     loss_val_list = []
#     for i, alpha in enumerate(alphas):
#         # TODO remove the range requirment
#         m_pt = [m_opt + alpha * (m_opt - m_init) for m_init, m_opt in zip(train_var_init, train_var_final) ]
#         input_dict = dict(zip(model.trainable_variables, m_pt) )
#         loss_minibatch = []
#         for j in range(n_batch):
#             loss_minibatch_val = sess.run(model.cost, feed_dict=input_dict)
#             loss_minibatch.append(loss_minibatch_val)
#         loss_val_list.append(np.mean(np.array(loss_minibatch) ) )
#     return alphas, np.array(loss_val_list)


def get_curvature(all_alphas, all_losses):
    loss_grad = (all_losses[1:]-all_losses[:-1]) / (all_alphas[1:]-all_alphas[:-1])
    dalpha = (all_alphas[1:]+all_alphas[:-1])/2

    alpha_star_index = np.nanargmin(all_losses)
    if alpha_star_index == 0:
        raise Exception, 'min value at begining'
    alpha_star = all_alphas[alpha_star_index]
    
    star_curvature = loss_grad/(dalpha-alpha_star) 
    star_curvature[min(alpha_star_index, star_curvature.size - 1) ]=np.nan
    star_curvature[alpha_star_index-1]=np.nan
    return star_curvature, dalpha


def get_mu_lr_from_curvature(star_curvature):
    # drop those negative curvatures
    star_curvature = star_curvature[np.where(star_curvature > 0) ]
    
    dr = max(star_curvature)/min(star_curvature)
    print 'Dynamic range: ', dr

    mu_star = ((np.sqrt(dr)-1)/(np.sqrt(dr)+1))**2

    eta_star = (1.0 - np.sqrt(mu_star))**2/min(star_curvature)
    
    return (mu_star, eta_star)


def pred_lr_mu(all_alphas, all_losses, train_var_init, train_var_final, alpha_range, display=True):
    # The default setting for maxfev in curve_fit is 200 * (1 + N) with N the number of  
    maxfev = 200 * (1 + 7)
    fit_done = False
    # give it enough iteration to get a opt value
    while (fit_done == False):
        try:
            all_losses_fit, _ = get_losses_from_fit(all_alphas, all_losses, alpha_range, maxfev=maxfev)
            fit_done = True
        except:
            if maxfev >= 10000:
                return None, None
            print("fitting with ", maxfev, " max iters")
            pass    
        maxfev *= 2
        
    if display:
        plt.plot(all_alphas, all_losses, label='slice loss')
        plt.plot(all_alphas, all_losses_fit, label='Polynomial fit')
        plt.legend()
        plt.show()
    
#     # save for debug
#     np.savetxt("all_alphas_debug.txt", np.array(all_alphas) )
#     np.savetxt("all_losses_debug.txt", np.array(all_losses_fit) )
    
    # jump over the cases when the first point is the lowest
    try:
        curvature_smooth_s, dalpha = get_curvature(all_alphas, all_losses_fit)
    except Exception as e:
        if e.message == 'min value at begining':
            # case where the lowest point is at the begining
            print("reuse last predicted lr and mu")
            return None, None
    
    # plt.plot(dalpha, curvature_smooth_s)
    if all_losses_fit[0] == np.nanmax(all_losses_fit):
        alpha_cutoff=all_alphas[-1]
    else:
        alpha_cutoff=all_alphas[all_losses_fit>all_losses_fit[0]][0]
    curvature_smooth_s = curvature_smooth_s[dalpha<alpha_cutoff]
    if np.all(np.isnan(curvature_smooth_s) ):
        print "cut-offed curvature are all nan"
        return None, None
    
    dalpha = dalpha[dalpha<alpha_cutoff]

    if display:
        plt.figure()
        plt.plot(dalpha, curvature_smooth_s)
        plt.title('*-curvature (contraction)')
        plt.ylabel('Contraction')
        plt.xlabel('Position on the slice')
        plt.show()
    
    # get lr mu
    print "max curvature ", "  min curvature ", "   min positive curvature"
    print np.nanmax(curvature_smooth_s), np.nanmin(curvature_smooth_s), np.nanmin(curvature_smooth_s[np.where(curvature_smooth_s > 0) ] )

    mu_star_smooth_s, lr_star_smooth_s = get_mu_lr_from_curvature(curvature_smooth_s)
    
    orig_dist = 0
    # TODO recover the range issue
    for m_init, m_final in zip(train_var_init, train_var_final):
        orig_dist += np.linalg.norm(m_init - m_final)**2
    orig_dist = np.sqrt(orig_dist)
    
    alpha_star_index = np.nanargmin(all_losses_fit)
    alpha_star = all_alphas[alpha_star_index]
    
    # lr = lr_star_smooth_s*orig_dist/(1+alpha_star)
    # lr = lr*orig_dist
    lr = lr_star_smooth_s*(orig_dist**2)

    mu = mu_star_smooth_s
    
    return lr, mu


# def pred_lr_mu(model_pred, sess, get_obj_slice_val_func, train_var_init, train_var_final, n_slice_point, n_batch, alpha_range, display=True):
#     all_alphas, all_losses = get_obj_slice_val_func(model_pred, sess, train_var_init, train_var_final, n_slice_point, alpha_range, n_batch=n_batch)    
    
# #     np.savetxt("all_alphas_raw_debug.txt", np.array(all_alphas) )
# #     np.savetxt("all_losses_raw_debug.txt", np.array(all_losses) )
    
#     if display:
#         plt.figure()
#         plt.plot(all_alphas, all_losses, alpha=0.3, label='Noisy measurement')
    
#     # The default setting for maxfev in curve_fit is 200 * (1 + N) with N the number of  
#     maxfev = 200 * (1 + 7)
#     fit_done = False
#     # give it enough iteration to get a opt value
#     while (fit_done == False):
#         print("fitting with ", maxfev, " max iters")
#         try:
#             all_losses_fit, _ = get_losses_from_fit(all_alphas, all_losses, alpha_range, maxfev=maxfev)
#             fit_done = True
#         except:
#             if maxfev >= 10000:
#                 return None, None
#             pass    
#         maxfev *= 2
        
#     if display:
#         plt.plot(all_alphas, all_losses_fit, label='Polynomial fit')
#         plt.legend()
#         plt.show()
    
# #     # save for debug
# #     np.savetxt("all_alphas_debug.txt", np.array(all_alphas) )
# #     np.savetxt("all_losses_debug.txt", np.array(all_losses_fit) )
    
#     # jump over the cases when the first point is the lowest
#     try:
#         curvature_smooth_s, dalpha = get_curvature(all_alphas, all_losses_fit)
#     except Exception as e:
#         if e.message == 'min value at begining':
#             # case where the lowest point is at the begining
#             print("reuse last predicted lr and mu")
#             return None, None
    
#     plt.plot(dalpha, curvature_smooth_s)
#     if all_losses_fit[0] == np.nanmax(all_losses_fit):
#         alpha_cutoff=all_alphas[-1]
#     else:
#         alpha_cutoff=all_alphas[all_losses_fit>all_losses_fit[0]][0]
#     curvature_smooth_s = curvature_smooth_s[dalpha<alpha_cutoff]
#     if np.all(np.isnan(curvature_smooth_s) ):
#         print "cut-offed curvature are all nan"
#         return None, None
    
#     dalpha = dalpha[dalpha<alpha_cutoff]

#     if display:
#         plt.figure()
#         plt.plot(dalpha, curvature_smooth_s)
#         plt.title('*-curvature (contraction)')
#         plt.ylabel('Contraction')
#         plt.xlabel('Position on the slice')
#         plt.show()
    
#     # get lr mu
#     print "max curvature ", "  min curvature ", "   min positive curvature"
#     print np.nanmax(curvature_smooth_s), np.nanmin(curvature_smooth_s), np.nanmin(curvature_smooth_s[np.where(curvature_smooth_s > 0) ] )

#     mu_star_smooth_s, lr_star_smooth_s = get_mu_lr_from_curvature(curvature_smooth_s)
    
#     orig_dist = 0
#     # TODO recover the range issue
#     for m_init, m_final in zip(train_var_init, train_var_final):
#         orig_dist += np.linalg.norm(m_init - m_final)**2
#     orig_dist = np.sqrt(orig_dist)
    
#     alpha_star_index = np.nanargmin(all_losses_fit)
#     alpha_star = all_alphas[alpha_star_index]
    
#     lr = lr_star_smooth_s*orig_dist/(1+alpha_star)
#     lr = lr_star_smooth_s*orig_dist

#     mu = mu_star_smooth_s
    
#     print("mu*=", mu, " lr*=", lr)
    
#     return lr, mu
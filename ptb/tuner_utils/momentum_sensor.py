import numpy as np
import matplotlib.pyplot as plt


class MomentumSensor():
    def __init__(self):
        self._prev_prev_x = None
        self._prev_x = None
        self._current_x = None
        self._avg_step = None
        self._avg_step_diff = None
        self._avg_two_step_diff = None
        self._grad_avg = None
        
    def process_sample(self,x,grad,beta):
        # Shift xs
        self._prev_prev_prev_x = self._prev_prev_x
        self._prev_prev_x = self._prev_x
        self._prev_x = self._current_x
        self._current_x = np.array(x)
        
        if self._grad_avg is None:
            self._grad_avg = np.array(grad)
        else:
            self._grad_avg = beta*np.array(grad) + (1-beta)*self._grad_avg
        
        if not self._prev_x is None:
            if self._avg_step is None:
                self._avg_step = self._current_x-self._prev_x
            else:
                self._avg_step = beta*(self._current_x-self._prev_x) + (1-beta)*self._avg_step

        if not self._prev_prev_x is None:
            if self._avg_step_diff is None:
                self._avg_step_diff = self._current_x - 2*self._prev_x + self._prev_prev_x
            else:
                self._avg_step_diff = beta*(self._current_x - 2*self._prev_x + self._prev_prev_x) + (1-beta)*self._avg_step_diff            
                
        if not self._prev_prev_prev_x is None:
            if self._avg_two_step_diff is None:
                self._avg_two_step_diff = self._current_x - self._prev_x - self._prev_prev_x + self._prev_prev_prev_x
            else:
                self._avg_two_step_diff = beta*(self._current_x - self._prev_x - self._prev_prev_x + self._prev_prev_prev_x) + (1-beta)*self._avg_two_step_diff   
        
    def estimate_total_momentum(self, lr, mom=None, debug=False):
        num = (lr*self._grad_avg + self._avg_step)
        den = (self._avg_step - self._avg_step_diff)
        mom_estimates = num/(den + 1e-10)
        mom_hat = np.median(mom_estimates)
        
        # if debug:
        #     print "Momentum estimate=", mom_hat
        #     sortd = np.sort(mom_estimates)
        #     yvals = np.arange(len(sortd))/float(len(sortd))
        #     plt.plot(sortd, yvals, label='Total momentum')
        #     if not mom is None:
        #         plt.plot([mom, mom], [0.0,1.0], label='Algorithmic (known)')
        #     plt.axis([np.percentile(mom_estimates, 10) ,np.percentile(mom_estimates, 90), 0.0,1.0 ])

        #     plt.axis([-1.0,2.0,None,None])
        #     plt.grid(True)
        #     plt.title('CDF of old momentum estimates')
        #     plt.show(block=False)

        return mom_hat
    
    def estimate_implicit_momentum(self, lr, mom, debug=False):
        num = ( (1-mom)*self._avg_step + mom*self._avg_step_diff + lr*self._grad_avg)
        den = ((1-mom)*self._avg_step - self._avg_step_diff + mom*self._avg_two_step_diff + lr*self._grad_avg)
        mom_estimates = num/den
        mom_hat = np.median(mom_estimates)        
        
        if debug:
            print "Implicit momentum estimate=", mom_hat
            sortd = np.sort(mom_estimates)
            yvals = np.arange(len(sortd))/float(len(sortd))
            plt.plot(sortd, yvals, label='Asynchrony-induced')
            plt.axis([-1.0,2.0,None,None])
            plt.grid(True)
            plt.xlim( [] )
            plt.title('CDF of momentum estimates')
            plt.legend(loc='lower right')

            
        return mom_hat
    
    def estimate_stale_momentum(self, lr, mom, debug=False):
        # we first simulate one step staleness then multiple
        num = (self._avg_step - self._avg_step_diff + lr * self._grad_avg)
        den = (self._avg_step - self._avg_two_step_diff)
        mom_estimates = num/den
        mom_hat = np.median(mom_estimates)
        if debug:
            print "1 step staleness momentum estimate=", mom_hat
            sortd = np.sort(mom_estimates)
            yvals = np.arange(len(sortd))/float(len(sortd))
            plt.plot(sortd, yvals, label='1 step staleness')
            plt.axis([-1.0,2.0,None,None])
            plt.grid(True)
            plt.title('CDF of momentum estimates')
            plt.legend(loc='lower right')
                

class MomentumSensorStale():
    def __init__(self, max_staleness=16):
        self._prev_prev_x = None
        self._prev_x = None
        self._current_x = None
        self._avg_step = None
        self._avg_step_diff = None
        self._avg_two_step_diff = None
        self._grad_avg = None
        
        self._all_x = [None]*(max_staleness+4)

        self._median_est_list = []
        self._cut_mean_est_list = []

        self._median_est = None
        self._cut_mean_est = None

        self._iter_id = 0

        
    # def process_sample(self,x,grad,beta,staleness):        
    #     self._all_x.append(np.array(x))
        
    #     self._prev_prev_prev_x = self._all_x[-staleness-4]
    #     self._prev_prev_x = self._all_x[-staleness-3]
    #     self._prev_x = self._all_x[-staleness-2]
    #     self._current_x = self._all_x[-staleness-1]
        
    #     # print "beta ", beta

    #     if len(self._all_x) >= staleness + 6:
    #         self._all_x.pop(0)
          
    #     if self._grad_avg is None:
    #         self._grad_avg = np.array(grad)
    #     else:
    #         self._grad_avg = beta*np.array(grad) + (1-beta)*self._grad_avg
        
    #     if not self._prev_x is None:
    #         if self._avg_step is None:
    #             self._avg_step = self._current_x-self._prev_x
    #         else:
    #             self._avg_step = beta*(self._current_x-self._prev_x) + (1-beta)*self._avg_step

    #     if not self._prev_prev_x is None:
    #         if self._avg_step_diff is None:
    #             self._avg_step_diff = self._current_x - 2*self._prev_x + self._prev_prev_x
    #         else:
    #             self._avg_step_diff = beta*(self._current_x - 2*self._prev_x + self._prev_prev_x) + (1-beta)*self._avg_step_diff            

    #     if not self._prev_prev_prev_x is None:
    #         if self._avg_two_step_diff is None:
    #             self._avg_two_step_diff = self._current_x - self._prev_x - self._prev_prev_x + self._prev_prev_prev_x
    #         else:
    #             self._avg_two_step_diff = beta*(self._current_x - self._prev_x - self._prev_prev_x + self._prev_prev_prev_x) + (1-beta)*self._avg_two_step_diff   
         
    #     self._iter_id += 1

    def process_sample(self,x,grad,beta,staleness,bias_correction=True):        
        self._all_x.append(np.array(x))
        
        self._prev_prev_prev_x = self._all_x[-staleness-4]
        self._prev_prev_x = self._all_x[-staleness-3]
        self._prev_x = self._all_x[-staleness-2]
        self._current_x = self._all_x[-staleness-1]
        
        # print "beta ", beta

        if len(self._all_x) >= staleness + 6:
            self._all_x.pop(0)
          
        if self._grad_avg is None:
            self._grad_avg = np.array(grad)
        else:
            self._grad_avg = beta*np.array(grad) + (1-beta)*self._grad_avg
            # if bias_correction:
	           #  self._grad_avg /= (1.0 - beta**self._iter_id)
        
        if not self._prev_x is None:
            if self._avg_step is None:
                self._avg_step = self._current_x-self._prev_x
            else:
                self._avg_step = beta*(self._current_x-self._prev_x) + (1-beta)*self._avg_step
                # if bias_correction:
	               #  self._avg_step /= (1.0 - beta**self._iter_id)

        if not self._prev_prev_x is None:
            if self._avg_step_diff is None:
                self._avg_step_diff = self._current_x - 2*self._prev_x + self._prev_prev_x
            else:
                self._avg_step_diff = beta*(self._current_x - 2*self._prev_x + self._prev_prev_x) + (1-beta)*self._avg_step_diff            
                # if bias_correction:
	               #  self._avg_step_diff /= (1.0 - beta**self._iter_id)

        if not self._prev_prev_prev_x is None:
            if self._avg_two_step_diff is None:
                self._avg_two_step_diff = self._current_x - self._prev_x - self._prev_prev_x + self._prev_prev_prev_x
            else:
                self._avg_two_step_diff = beta*(self._current_x - self._prev_x - self._prev_prev_x + self._prev_prev_prev_x) + (1-beta)*self._avg_two_step_diff   
                # if bias_correction:
	               #  self._avg_two_step_diff /= (1.0 - beta**self._iter_id)

        self._iter_id += 1


    def estimate_momentum(self, lr, mom, beta_measure, debug=False, bias_correction=False):
			num = (lr*self._grad_avg + self._avg_step)
			den = (self._avg_step - self._avg_step_diff)
			mom_estimates = num/(den + 1e-10)
			percentile_lower = np.percentile(mom_estimates, 5)
			percentile_upper = np.percentile(mom_estimates, 95) 
			median_estimator = np.median(mom_estimates)
			cut_mean_estimator = np.mean(mom_estimates[np.logical_and(mom_estimates > percentile_lower, mom_estimates < percentile_upper) ] )
      

			if self._median_est is None:
				self._median_est = median_estimator
			else:
				self._median_est = beta_measure * median_estimator \
					+ (1 - beta_measure) * self._median_est

			if self._cut_mean_est is None:
				self._cut_mean_est = cut_mean_estimator
			else:
				self._cut_mean_est = beta_measure * cut_mean_estimator \
					+ (1 - beta_measure) * self._cut_mean_est

			if debug:
				self._median_est_list.append(self._median_est)
				self._cut_mean_est_list.append(self._cut_mean_est)
				# print "cut mean estimate ", cut_mean_estimator
				# print "median estimate=", median_estimator
				# if not mom is None:
				#   plt.plot([mom, mom], [0.0,1.0], label='Algorithmic (known)')


    #     # sortd = np.sort(num)
    #     # yvals = np.arange(len(sortd))/float(len(sortd))
    #     # plt.plot(sortd, yvals)
    #     # plt.axis([np.percentile(sortd, 5),np.percentile(sortd, 95), None, None] )
    #     # plt.grid()
    #     # plt.title('CDF of numerator')
    #     # plt.show(block=False)
        
    #     # sortd = np.sort(den)
    #     # yvals = np.arange(len(sortd))/float(len(sortd))
    #     # plt.plot(sortd, yvals)
    #     # #plt.axis([-1.0,2.0,None,None])
    #     # plt.axis([np.percentile(sortd, 5),np.percentile(sortd, 95), None, None] )
    #     # plt.grid()
    #     # plt.title('CDF of denominator')
    #     # plt.show(block=False)
   
				# sortd = np.sort(mom_estimates)
				# yvals = np.arange(len(sortd))/float(len(sortd))
				# plt.plot(sortd, yvals)
				# plt.axis([-0.5 ,1.0, 0.0,1.0 ])
				# plt.grid()
				# plt.title('CDF of momentum estimates')
				# plt.show(block=False)

			return median_estimator, cut_mean_estimator


    def plot_est_history(self, mom=-1.0):
			print "median ", self._median_est_list[-1]
			print "cut mean ", self._cut_mean_est_list[-1]
			plt.plot(self._cut_mean_est_list, label="cut mean")
			plt.plot(self._median_est_list, label="median")
			plt.plot([0, len(self._median_est_list) ], [mom, mom], label="real")
			plt.legend()
			plt.ylim( [-0.5, 1.5] )
			plt.grid()
			plt.show(block=False)
    
    def estimate_momentum_fancy(self, lr, mom, debug=False):
        num = ( (1-mom)*self._avg_step + mom*self._avg_step_diff + lr*self._grad_avg)
        den = ((1-mom)*self._avg_step - self._avg_step_diff + mom*self._avg_two_step_diff + lr*self._grad_avg)
        mom_estimates = num/den
        
        mom_hat = np.median(mom_estimates)
        if debug:
            print "Fancy momentum estimate=", mom_hat
            sortd = np.sort(num)
            yvals = np.arange(len(sortd))/float(len(sortd))
            plt.plot(sortd, yvals)
            plt.axis([-1.0,2.0,None,None])
            plt.grid()
            plt.title('CDF of numerator')
            plt.show()
            
            print "Fancy momentum estimate=", mom_hat
            sortd = np.sort(mom_estimates)
            yvals = np.arange(len(sortd))/float(len(sortd))
            plt.plot(sortd, yvals)
            plt.axis([-1.0,2.0,None,None])
            plt.grid()
            plt.title('CDF of momentum estimates')
        return mom_hat            
import tensorflow as tf
import numpy as np

from yellow_fin import *


def test_sync():
	x = tf.Variable( [1.0,], tf.float32)
	y = tf.Variable( [1.0,], tf.float32)
	loss = tf.reduce_sum(x + 100 * y)

	grad = tf.gradients(loss, [x, y] )

	optimizer = MetaOptimizer( [0.1, ], [0.9, ], [10000.0, ], use_async=False)

	apply_op = optimizer.apply_gradients([zip(grad, [x, y] ), ])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer() )

		res = sess.run( [grad, optimizer._optimizers[0]._grad_squared, 
			optimizer._optimizers[0]._moving_grad_squared,
			optimizer._optimizers[0].grad_squared_sum] )
		print "grad / moving grad squared ", res[0], res[1], res[2], res[3]

		for i in range(2):
			print "iter ", i
			res = sess.run( [optimizer._optimizers[0].lr_var, optimizer._optimizers[0].mu_var, optimizer._optimizers[0].grad_squared_sum] )
			print "lr ", res[-3], "mu ", res[-2], " accum ", res[-1]
			res = sess.run( [apply_op, grad, optimizer._optimizers[0]._grad_squared, optimizer._optimizers[0]._moving_grad_squared] )
			print "grad / moving grad squared ", res[1], res[2], res[3]
			res = sess.run( [optimizer._optimizers[0].lr_var, optimizer._optimizers[0].mu_var] )
			print "lr ", res[-2], "mu ", res[-1]

		# assert np.abs(res[-2] - 0.0707071) < 1e-5
		# assert np.abs(res[-1] - 0.669421) < 1e-5
	return


def test_async():
	x = tf.Variable( [10.0,], tf.float32)
	y = tf.Variable( [10.0,], tf.float32)
	z = tf.Variable( [10.0,], tf.float32)
	loss = tf.reduce_sum(x**2 + y**2 + z**2)

	grad = tf.gradients(loss, [x, y, z] )

	optimizer = MetaOptimizer( [0.002, ], [0.3, ], [10000.0, ], gamma=0.9, use_async=True)

	apply_op = optimizer.apply_gradients([zip(grad, [x, y, z] ), ], staleness=8)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer() )

		# res = sess.run( [grad, optimizer._optimizers[0]._grad_squared, 
		# 	optimizer._optimizers[0]._moving_grad_squared,
		# 	optimizer._optimizers[0].grad_squared_sum] )
		# print "grad / moving grad squared ", res[0], res[1], res[2], res[3]
		# verification: 1. delay works[Done] 2. diffrence works[Done] 3. moving works[Done] 

		# for i in range(100):
		# 	print
		# 	print 
		# 	print "iter ", i
		# 	res = sess.run( [optimizer._optimizers[0].test_momentum, 
		# 		optimizer._optimizers[0].test_momentum_delay,
		# 		optimizer._optimizers[0].test_momentum_diff] )
		# 	print "mom ", res[-3], "mom_delay ", res[-2], " mom_diff ", res[-1]
		# 	res = sess.run( [apply_op, grad, optimizer._optimizers[0]._grad_squared] )
		# 	# print "mom ", res[-3], "mom_delay ", res[-2], " mom_diff ", res[-1]
		# 	res = sess.run( [optimizer._optimizers[0].lr_var, optimizer._optimizers[0].mu_var, optimizer._optimizers[0].test_momentum, 
		# 		optimizer._optimizers[0].test_momentum_delay,
		# 		optimizer._optimizers[0].test_momentum_diff] )	
		# 	print "lr ", res[0], "mu ", res[1]		
		# 	print "mom ", res[-3], "mom_delay ", res[-2], " mom_diff ", res[-1]
		

		for i in range(300):
			print
			print 
			print "iter ", i

			res = sess.run([x,])
			print "before apply var ", res

			agent = optimizer._optimizers[0]._optimizer
			res = sess.run([agent.get_slot(x, "momentum"), 
											agent.get_slot(x, "momentum_delay"),
											agent.get_slot(x, "momentum_diff") ]) 
			print "before apply var triplet ", res


			res = sess.run( [apply_op, grad, optimizer._optimizers[0].mu_array] )
			print "estimation ", res[2]
			print "grad ", res[1]
			# print "mom ", res[-3], "mom_delay ", res[-2], " mom_diff ", res[-1]
			
			agent = optimizer._optimizers[0]._optimizer
			res = sess.run([agent.get_slot(x, "momentum"), 
											agent.get_slot(x, "momentum_delay"),
											agent.get_slot(x, "momentum_diff") ]) 
			print "after apply var triplet ", res







if __name__ == "__main__":
	# test_sync()
	test_async()
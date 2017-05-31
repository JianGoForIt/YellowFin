import tensorflow as tf 
import numpy as np

if __name__ == "__main__":
	step0 = tf.Variable(1.0, trainable=False)
	step1 = tf.Variable(0.0, trainable=False)
	step2 = tf.Variable(0)

	step1 = step0
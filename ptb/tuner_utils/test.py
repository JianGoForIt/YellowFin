import tensorflow as tf

data_path="../resnet/cifar10/data_batch*"
data_files = tf.gfile.Glob(data_path)
print data_files, type(data_files[0])

import tensorflow as tf

from resnet import resnet_model
from resnet import cifar_input

from resnet.resnet_utils import *

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


# PROBLEMS TO SOLVE
# data feeding

# Build model...
NUM_CLASSES = 10
NUM_TRAIN_FILE = 5
TRAIN_DATA_PATH = '../resnet/cifar10/data_batch'
TEST_DATA_PATH = '../resnet/cifar10/test_batch.bin'
MODE = 'train'
LOG_ROOT='../resnet/ckpt/resnet_model'
DATASET='cifar10'
DEV = '/cpu:0'
log_dir = "../resnet/resnet_res/resnet-async-test-sgd-mom"
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
batch_size_train = 128
num_steps = 40000
hps_train = resnet_model.HParams(batch_size=batch_size_train,
                   num_classes=NUM_CLASSES,
                   min_lrn_rate=0.0001,
                   lrn_rate=0.001,
                   mom=0.9,
                   clip_norm_base=10.0,
                   num_residual_units=5,
                   use_bottleneck=False,
                   weight_decay_rate=0.0002,
                   relu_leakiness=0.1,
                   optimizer='meta-bundle',
                   model_scope='train')

lr_vals = 0.001
mu_vals = 0.9
clip_thresh_vals = 1000.0
display_interval = 50
test_interval = 50


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      with tf.variable_scope("train"):
        # data file is index in 1-based way
        file_id_start = NUM_TRAIN_FILE / len(worker_hosts) * FLAGS.task_index + 1
        if FLAGS.task_index == len(worker_hosts) - 1:
          file_id_end = NUM_TRAIN_FILE
        else:
          file_id_end = NUM_TRAIN_FILE / len(worker_hosts) * (FLAGS.task_index + 1) + 1
        datafiles = [TRAIN_DATA_PATH + "_" + str(datafile_id) for datafile_id in range(file_id_start, file_id_end) ]
        model_train = get_model(hps_train, DATASET, datafiles, mode='train')
     
      # # tmp = tf.Variable(0.0, dtype=tf.float32)
      # loss = model_train.cost
      global_step = tf.Variable(0)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      print "start execution"
      # # Loop until the supervisor shuts down or 1000000 steps have completed.
      # while not sv.should_stop() and step < 1000000:
      #   # Run a training step asynchronously.
      #   # See `tf.train.SyncReplicasOptimizer` for additional details on how to
      #   # perform *synchronous* training.
      #   _, step, loss = sess.run([model_train.train_op, global_step, model_train.cost])
      training(model_train, None, sess, num_steps, lr_vals, mu_vals, 
         clip_thresh_vals, init_op, display_interval, 
         log_dir, test_interval, use_meta=True)

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()


# python async_train.py \
#      --ps_hosts=raiders1.stanford.edu:2222 \
#      --worker_hosts=raiders1.stanford.edu:2224 \
#      --job_name=ps --task_index=0

# python async_training.py \
#      --ps_hosts=raiders1.stanford.edu:2222 \
#      --worker_hosts=raiders1.stanford.edu:2224 \
#      --job_name=ps --task_index=0

# python async_training.py \
#      --ps_hosts=raiders1.stanford.edu:2222 \
#      --worker_hosts=raiders1.stanford.edu:2224 \
#      --job_name=worker --task_index=0

# python async_training.py \
#      --ps_hosts=raiders1.stanford.edu:2222 \
#      --worker_hosts=raiders1.stanford.edu:2224,raiders1.stanford.edu:2225 \
#      --job_name=worker --task_index=0

#      python async_training.py \
#      --ps_hosts=raiders1.stanford.edu:2222 \
#      --worker_hosts=raiders1.stanford.edu:2224,raiders1.stanford.edu:2225 \
#      --job_name=worker --task_index=1

# python async_run.py \
#      --ps_hosts=raiders1.stanford.edu:2222 \
#      --worker_hosts=raiders1.stanford.edu:2224 \
#      --job_name=worker --task_index=0

# # python async_run.py \
# #      --ps_hosts=raiders1.stanford.edu:2222,raiders1.stanford.edu:2223 \
# #      --worker_hosts=raiders1.stanford.edu:2224,raiders1.stanford.edu:2225 \
# #      --job_name=worker --task_index=1
# 
# 

# python async_run.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2223 --job_name=ps --task_index=0
# python async_run.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2223 --job_name=ps --task_index=0



# python async_training.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2223,127.0.0.1:2224 --job_name=ps --task_index=0

# python async_training.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2223,127.0.0.1:2224 --job_name=worker --task_index=0

# python async_training.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2223,127.0.0.1:2224 --job_name=worker --task_index=1


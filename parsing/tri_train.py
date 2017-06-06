from __future__ import absolute_import, division, print_function
from random import shuffle
from utils import MediumConfig, PTBModel, chop, run_epoch, run_epoch2

from utils import ptb_iterator, nbest_iterator

import itertools, sys, time

import cPickle as pickle
import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_float('init_scale', None, 'init_scale')
flags.DEFINE_float('learning_rate', None, 'learning_rate')
flags.DEFINE_float('max_grad_norm', None, 'max_grad_norm')
flags.DEFINE_integer('num_layers', None, 'num_layers')
flags.DEFINE_integer('num_steps', None, 'num_steps')
flags.DEFINE_integer('hidden_size', None, 'hidden_size')
flags.DEFINE_integer('max_epoch', None, 'max_epoch')
flags.DEFINE_integer('max_max_epoch', None, 'max_max_epoch')
flags.DEFINE_float('keep_prob', 0.55, 'keep_prob')
flags.DEFINE_float('lr_decay', 0.85, 'lr_decay')
flags.DEFINE_integer('batch_size', None, 'batch_size')
flags.DEFINE_string('model_path', None, 'model_path')
flags.DEFINE_integer('silver', 10, 'silver')

FLAGS = flags.FLAGS


def train():
  print('data_path: %s' % FLAGS.data_path)
  raw_data = reader.ptb_raw_data3(FLAGS.data_path)
  train_data, silver_path, valid_data, valid_nbest_data, vocab = raw_data
  train_data = chop(train_data, vocab['<eos>'])
  
  config = MediumConfig()
  if FLAGS.init_scale: config.init_scale = FLAGS.init_scale
  if FLAGS.learning_rate: config.learning_rate = FLAGS.learning_rate
  if FLAGS.max_grad_norm: config.max_grad_norm = FLAGS.max_grad_norm
  if FLAGS.num_layers: config.num_layers = FLAGS.num_layers
  if FLAGS.num_steps: config.num_steps = FLAGS.num_steps
  if FLAGS.hidden_size: config.hidden_size = FLAGS.hidden_size
  if FLAGS.max_epoch: config.max_epoch = FLAGS.max_epoch
  if FLAGS.max_max_epoch: config.max_max_epoch = FLAGS.max_max_epoch
  if FLAGS.keep_prob: config.keep_prob = FLAGS.keep_prob
  if FLAGS.lr_decay: config.lr_decay = FLAGS.lr_decay
  if FLAGS.batch_size: config.batch_size = FLAGS.batch_size
  config.vocab_size = len(vocab)
  if FLAGS.silver: config.silver = FLAGS.silver
  print('init_scale: %.2f' % config.init_scale)
  print('learning_rate: %.2f' % config.learning_rate)
  print('max_grad_norm: %.2f' % config.max_grad_norm)
  print('num_layers: %d' % config.num_layers)
  print('num_steps: %d' % config.num_steps)
  print('hidden_size: %d' % config.hidden_size)
  print('max_epoch: %d' % config.max_epoch)
  print('max_max_epoch: %d' % config.max_max_epoch)
  print('keep_prob: %.2f' % config.keep_prob)
  print('lr_decay: %.2f' % config.lr_decay)
  print('batch_size: %d' % config.batch_size)
  print('vocab_size: %d' % config.vocab_size)
  print('silver: %d' % config.silver)
  sys.stdout.flush()
  
  eval_config = MediumConfig()
  eval_config.init_scale = config.init_scale
  eval_config.learning_rate = config.learning_rate
  eval_config.max_grad_norm = config.max_grad_norm
  eval_config.num_layers = config.num_layers
  eval_config.num_steps = config.num_steps
  eval_config.hidden_size = config.hidden_size
  eval_config.max_epoch = config.max_epoch
  eval_config.max_max_epoch = config.max_max_epoch
  eval_config.keep_prob = config.keep_prob
  eval_config.lr_decay = config.lr_decay
  eval_config.batch_size = 200
  eval_config.vocab_size = len(vocab)

  prev = 0 # record F1 scores
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()
    if FLAGS.model_path:
      saver = tf.train.Saver()

    silver_generator = reader.file_to_word_ids3(silver_path)
    j = 0
    for i in range(config.max_max_epoch):
      shuffle(train_data)
      shuffled_data = list(itertools.chain(*train_data))
      
      start_time = time.time()
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)
      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, shuffled_data, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
      valid_f1, num = run_epoch2(session, mvalid, valid_nbest_data,
                                 tf.no_op(), vocab['<eos>'])
      print("Epoch: %d Valid F1: %.2f (%d trees)" % (i + 1, valid_f1, num))
      if valid_f1 > prev:
        prev = valid_f1
        if FLAGS.model_path:
          print('Save a model to %s' % FLAGS.model_path)
          saver.save(session, FLAGS.model_path)
          pickle.dump(eval_config, open(FLAGS.model_path + '.config', 'wb'))
      print('It took %.2f seconds' % (time.time() - start_time))
      sys.stdout.flush()

      start_time = time.time()
      for k in xrange(config.silver):
        try:
          silver_data = silver_generator.next()
        except:
          silver_generator = reader.file_to_word_ids3(silver_path)
          silver_data = silver_generator.next()
        j += 1
        silver_data = chop(silver_data, vocab['<eos>'])
        shuffle(silver_data)
        silver_data = list(itertools.chain(*silver_data))
        silver_perplexity = run_epoch(session, m, silver_data, m.train_op,
                                      verbose=False)
        print("Epoch: %d Silver(%d) Perplexity: %.3f" %
              (i + 1, j, silver_perplexity))
        valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
        print("Epoch: %d Silver(V) Perplexity: %.3f" % (i+1, valid_perplexity))
        
      valid_f1, num = run_epoch2(session, mvalid, valid_nbest_data,
                                 tf.no_op(), vocab['<eos>'])
      print("Epoch: %d Silver(V) F1: %.2f (%d trees)" % (i+1, valid_f1, num))
      if valid_f1 > prev:
        prev = valid_f1
        if FLAGS.model_path:
          print('Save a model to %s' % FLAGS.model_path)
          saver.save(session, FLAGS.model_path)
          pickle.dump(eval_config, open(FLAGS.model_path + '.config', 'wb'))
      print('It took %.2f seconds' % (time.time() - start_time))
      sys.stdout.flush()
      


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  print(' '.join(sys.argv))
  train()
    

if __name__ == "__main__":
  tf.app.run()

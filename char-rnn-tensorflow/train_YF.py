from __future__ import print_function
import tensorflow as tf
import numpy as np

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--opt_method', type=str, default="YF", help="the optimizer to use")
    parser.add_argument('--seed', type=int, default=1, help="random seed for numpy and pytorch")
    parser.add_argument('--h_max_log_smooth', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    print("rand seed", args.seed)

    #with tf.device("gpu:0"):
    train(args)

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, partition='train')
    eval_data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, partition='eval')
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args, opt_method="YF")
    loss_list = []
    eval_loss_list = []
    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge(model.train_summary)
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # do evaluation
        e = -1
        eval_loss = 0.0
        start = time.time()
        eval_data_loader.reset_batch_pointer()
        state = sess.run(model.initial_state)
        for b in range(eval_data_loader.num_batches):
            x, y = eval_data_loader.next_batch()
            feed = {model.input_data: x, model.targets: y}
            for i, (c, h) in enumerate(model.initial_state):
                feed[c] = state[i].c
                feed[h] = state[i].h
            eval_loss_batch, state = sess.run([model.eval_cost, model.final_state], feed)
            eval_loss += eval_loss_batch
        eval_loss /= eval_data_loader.num_batches
        # instrument for tensorboard
        summ = tf.Summary(value=[tf.Summary.Value(tag="eval_loss", simple_value=eval_loss), ])
        writer.add_summary(summ, e * data_loader.num_batches)

        eval_loss_list.append( [(e + 1) * data_loader.num_batches, eval_loss] )
        end = time.time()
        print("{}/{} (epoch {}), eval_loss = {:.3f}, time/batch = {:.3f}"
              .format( (e + 1) * data_loader.num_batches,
                      args.num_epochs * data_loader.num_batches,
                      0, eval_loss, end - start))

        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            sess.run(tf.assign(model.optimizer.lr_factor, args.decay_rate ** e))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                # train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)

                # instrument for tensorboard
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * data_loader.num_batches + b)

                loss_list.append(train_loss)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                        or (e == args.num_epochs-1 and
                            b == data_loader.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

            # do evaluation
            eval_loss = 0.0
            start = time.time()
            print("start evaluation")
            eval_data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(eval_data_loader.num_batches):
                x, y = eval_data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                eval_loss_batch, state = sess.run([model.eval_cost, model.final_state], feed)
                eval_loss += eval_loss_batch
            eval_loss /= eval_data_loader.num_batches
            # instrument for tensorboard
            summ = tf.Summary(value=[tf.Summary.Value(tag="eval_loss", simple_value=eval_loss), ])
            writer.add_summary(summ, e * data_loader.num_batches)

            eval_loss_list.append( [(e + 1) * data_loader.num_batches, eval_loss] )
            end = time.time()
            print("{}/{} (epoch {}), eval_loss = {:.3f}, time/batch = {:.3f}"
                  .format( (e + 1) * data_loader.num_batches,
                          args.num_epochs * data_loader.num_batches,
                          e, eval_loss, end - start))
            

            with open(args.log_dir + "/loss.txt", "w") as f:
                np.savetxt(f, np.array(loss_list) )
            with open(args.log_dir + "/eval_loss.txt", "w") as f:
                np.savetxt(f, np.array(eval_loss_list) )

if __name__ == '__main__':
    main()

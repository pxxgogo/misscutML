# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2016-12-15 09:51:05
# @Last Modified by:   yuchen
# @Last Modified time: 2016-12-15 10:45:53

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import numpy.random as nr
import tensorflow as tf
import os
from tensorflow.python.ops import rnn as rnn_op

# from pyspin.spin import make_spin, Spin1

from provider import ptb_data_provider

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
provider = None
config = None
batch_size = 1024
num_steps = 35
size = 200
vocab_size = 20000
initial_state = None
final_state = None

X = None
y = None

cell = None

cost = None
train_op = None
new_lr = None
lr_update = None


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config['batch_size']
        self.num_steps = num_steps = config['num_steps']
        self.hidden_size = size = config['hidden_size']
        vocab_size = config['vocab_size']
        # print ("Batch size = {batch_size}. Num steps = {num_steps}.".format(
        # 	batch_size=batch_size, num_steps=num_steps))
        # raw_input()
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.


        with tf.device("/gpu:0"):
            lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            if is_training and config['keep_prob'] < 1:
                lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell1, output_keep_prob=config['keep_prob'])

            lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            if is_training and config['keep_prob'] < 1:
                lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2, output_keep_prob=config['keep_prob'])

            lstm_cell3 = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            if is_training and config['keep_prob'] < 1:
                lstm_cell3 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell3, output_keep_prob=config['keep_prob'])

            lstm_cell4 = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            if is_training and config['keep_prob'] < 1:
                lstm_cell4 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell4, output_keep_prob=config['keep_prob'])

        self._initial_state1 = lstm_cell1.zero_state(batch_size, data_type())
        self._initial_state2 = lstm_cell2.zero_state(batch_size, data_type())
        self._initial_state3 = lstm_cell3.zero_state(batch_size, data_type())
        self._initial_state4 = lstm_cell4.zero_state(batch_size, data_type())

        with tf.device("/gpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config['keep_prob'] < 1:
            inputs = tf.nn.dropout(inputs, config['keep_prob'])

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        state1 = self._initial_state1
        state2 = self._initial_state2
        state3 = self._initial_state3
        state4 = self._initial_state4
        inputs_list = [tf.squeeze(input_, [1])
                       for input_ in tf.split(1, num_steps, inputs)]

        outputs, state1, state2 = rnn_op.bidirectional_rnn(lstm_cell1, lstm_cell2, inputs_list, state1, state2,
                                                           dtype=tf.float32)
        inputs_list2 = [tf.squeeze(input_, [0])
                        for input_ in tf.split(0, num_steps, outputs)]

        outputs, state3, state4 = rnn_op.bidirectional_rnn(lstm_cell3, lstm_cell4, inputs_list2, state3, state4,
                                                           dtype=tf.float32)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps * 2], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        self._final_state1 = state1
        self._final_state2 = state2
        self._final_state3 = state3
        self._final_state4 = state4

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config['max_grad_norm'])
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state1(self):
        return self._initial_state1

    @property
    def initial_state2(self):
        return self._initial_state2

    @property
    def initial_state3(self):
        return self._initial_state3

    @property
    def initial_state4(self):
        return self._initial_state4

    @property
    def cost(self):
        return self._cost

    @property
    def final_state1(self):
        return self._final_state1

    @property
    def final_state2(self):
        return self._final_state2

    @property
    def final_state3(self):
        return self._final_state3

    @property
    def final_state4(self):
        return self._final_state4

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


# @make_spin(Spin1, "Running epoch...")
def run_epoch(session, model, provider, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state1 = session.run(model.initial_state1)
    state2 = session.run(model.initial_state2)
    state3 = session.run(model.initial_state3)
    state4 = session.run(model.initial_state4)
    provider.status = data
    for step, (x, y) in enumerate(provider()):
        fetches = [model.cost, model.final_state1, model.final_state2, model.final_state3, model.final_state4,
                   eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.initial_state1] = state1
        feed_dict[model.initial_state2] = state2
        feed_dict[model.initial_state3] = state3
        feed_dict[model.initial_state4] = state4
        cost, state1, state2, state3, state4, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.num_steps

        epoch_size = provider.get_epoch_size()
        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)


def main():
    provider = ptb_data_provider()
    provider.status = 'train'
    config = provider.get_config()
    eval_config = config.copy()
    eval_config['batch_size'] = 1
    eval_config['num_steps'] = 1

    # print (config)
    # print (eval_config)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i in range(config['max_max_epoch']):
            lr_decay = config['lr_decay'] ** max(i - config['max_epoch'], 0.0)
            m.assign_lr(session, config['learning_rate'] * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

            train_perplexity = run_epoch(session, m, provider, 'train', m.train_op, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, provider, 'valid', tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            save_path = saver.save(session, './model/misscut_rnn_1_model', global_step=i)
            print("Model saved in file: %s" % save_path)
            if (i % 13 == 0 and not i == 0):
                test_perplexity = run_epoch(session, mtest, provider, 'test', tf.no_op())
                print("Test Perplexity: %.3f" % test_perplexity)

        test_perplexity = run_epoch(session, mtest, provider, 'test', tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()

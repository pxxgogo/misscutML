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
        size = config['hidden_size']
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
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            if is_training and config['keep_prob'] < 1:
            	lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config['keep_prob'])
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config['num_layers'], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())
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
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

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
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

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
    state = session.run(model.initial_state)
    provider.status = data
    for step, (x, y) in enumerate(provider()):
        fetches = [model.cost, model.final_state, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
	# print (feed_dict)
    	cost, state, _ = session.run(fetches, feed_dict)
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
            save_path = saver.save(session, '/tmp/misscut_rnn_1_model', global_step=i)
            print("Model saved in file: %s" % save_path)
            if(i % 5 == 0 and not i == 0):
                test_perplexity = run_epoch(session, mtest, provider, 'test', tf.no_op())
                print("Test Perplexity: %.3f" % test_perplexity)

    	test_perplexity = run_epoch(session, mtest, provider, 'test', tf.no_op())
    	print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
    main()

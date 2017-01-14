# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2016-12-15 09:51:05
# @Last Modified by:   yuchen
# @Last Modified time: 2016-12-26 16:25:10

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
size = 750
vocab_size = 8000
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


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, keep_prob_):
    # print (x.get_shape())
    # print (W.shape)
    conv_2d = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME', True, data_format="NHWC")
    return tf.nn.dropout(conv_2d, keep_prob_)


def conv1d(x, W, keep_prob_):
    # print (x.get_shape())
    # print (W.shape)
    conv_1d = tf.nn.conv1d(value=x, filters=W, stride=1, padding='SAME')
    return tf.nn.dropout(conv_1d, keep_prob_)


class CNN_Cell(object):
    def __init__(self, w1, b1, w2, b2, keep_prob):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.keep_prob = keep_prob

    def __call__(self, x):
        conv2 = conv2d(x, self.w1, self.keep_prob)
        h_conv = tf.nn.relu(conv2 + self.b1)
        mp = tf.reduce_max(h_conv, axis=[2])
        print(mp.get_shape())
        # print (mp.get_shape())
        # print (self.w2.get_shape())
        logits = tf.sigmoid(tf.matmul(mp, self.w2) + self.b2)
        print(logits.get_shape())
        return logits


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config['batch_size']
        self.num_steps = num_steps = config['num_steps']
        self.vocab_size = vocab_size = config['vocab_size']
        self.hidden_size = size = config['hidden_size']
        # size = config['hidden_size']
        # vocab_size = config['vocab_size']
        # print ("Batch size = {batch_size}. Num steps = {num_steps}.".format(
        # 	batch_size=batch_size, num_steps=num_steps))
        # raw_input()
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        filter_size = 3
        input_channels = self.hidden_size
        output_channels = self.hidden_size
        stddev = 0.1
        keep_prob = 0.5

        with tf.device("/cpu:0"):
            w1 = tf.Variable(tf.random_normal([1, filter_size, input_channels, output_channels]))
            # w1 = weight_variable([filter_size, input_channels, output_channels], stddev)
            # w1 = tf.unpack(w1, axis=0)
            b1 = bias_variable([output_channels])
            w2 = tf.Variable(tf.random_normal([self.batch_size, output_channels, output_channels]))
            # w2 = weight_variable([output_channels, output_channels], stddev)
            b2 = bias_variable([output_channels])
            # embedding = tf.get_variable("embedding", [vocab_size, input_channels], dtype=data_type())
            cell = CNN_Cell(w1, b1, w2, b2, keep_prob)

        outputs = []
        # print (self._input_data)
        # inputs = np.array(self._input_data)
        # print (inputs.shape)
        # state = self._initial_state
        inputs = np.array([[] for i in range(batch_size)])
        # print(inputs.shape)
        model_inputs = tf.zeros([self.batch_size, 0, self.num_steps, self.hidden_size])
        # print(model_inputs.shape)
        with tf.variable_scope("CNN"):
            for time_step in range(num_steps):
                inputs = tf.concat(1, [inputs, self._input_data[:, time_step: time_step + 1]])
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
                x = tf.nn.embedding_lookup(embedding, inputs)
                zeros = tf.zeros([self.batch_size, self.num_steps - time_step - 1, self.hidden_size])
                x = tf.concat(1, [x, zeros])
                x = tf.reshape(x, [self.batch_size, 1, self.num_steps, self.hidden_size])
                # print(x.get_shape())
                model_inputs = tf.concat(1, [model_inputs, x])
                # cell_output = cell(x)
                # outputs.append(cell_output)
            # print(model_inputs.get_shape())
            cell_output = cell(model_inputs)
        # print ("Outputs")
        # print (np.array(outputs).shape)
        # print (np.array(outputs))
        # print (list(map(lambda t: np.array(t).shape, outputs)))
        output = tf.reshape(tf.concat(0, cell_output), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
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
    def cost(self):
        return self._cost


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
    provider.status = data
    for step, (x, y) in enumerate(provider()):
        fetches = [model.cost, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        # print (feed_dict)
        cost, _ = session.run(fetches, feed_dict)
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
            save_path = saver.save(session, './model/misscut_cnn_reduce_max_model', global_step=i)
            print("Model saved in file: %s" % save_path)
            if (i % 13 == 0 and not i == 0):
                test_perplexity = run_epoch(session, mtest, provider, 'test', tf.no_op())
                print("Test Perplexity: %.3f" % test_perplexity)

        test_perplexity = run_epoch(session, mtest, provider, 'test', tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()

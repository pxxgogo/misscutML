# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os
import json

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")


class RNN_model(object):
    """The LSTM Language Model."""

    class __RNN_model(object):
        def __init__(self, session_dir, words_map_dir, config_dir):
            self.config = json.load(open(config_dir, 'r'))

            # restore session

            self.session = tf.Session()
            with tf.variable_scope("model"):
                self.embedding = tf.get_variable("embedding", [self.config["vocab_size"], self.config["hidden_size"]],
                                                 dtype=tf.float32)

                self.softmax_w = tf.get_variable("softmax_w", [self.config["hidden_size"], self.config["vocab_size"]],
                                                 )
                self.softmax_b = tf.get_variable("softmax_b", [self.config["vocab_size"]], dtype=tf.float32)

                print(session_dir)

                self.init_LSTM()

            self.session.run(tf.global_variables_initializer())
            # for v in tf.global_variables():
            #     print(v.name)
            new_saver = tf.train.Saver()
            new_saver.restore(self.session, tf.train.latest_checkpoint(
                session_dir))

            print("------Model loaded------")

            for v in tf.global_variables():
                print(v.name)

            # init wordsMap
            self.words_map = json.load(open(words_map_dir, 'r'))
            self.words_map['\n'] = len(self.words_map)

        def init_LSTM(self):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config["hidden_size"], forget_bias=0.0,
                                                     state_is_tuple=True)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config["num_layers"], state_is_tuple=True)
            _input_data = tf.placeholder(tf.int32, [1, self.config["num_steps"]])
            inputs = tf.nn.embedding_lookup(self.embedding, _input_data)
            state = self.cell.zero_state(1, tf.float32)
            outputs = []
            with tf.variable_scope("RNN"):
                for time_step in range(self.config["num_steps"]):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = self.cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)

        def feed_data(self, sentence):
            with tf.variable_scope("model", reuse=True):
                tf.get_variable_scope().reuse_variables()
                np_array = np.array(sentence)
                sentence.pop(0)
                sentence.append(len(self.words_map) - 1)
                target_np_array = np.array(sentence)
                target = tf.reshape(tf.convert_to_tensor(target_np_array), [1, len(sentence)])
                tensor = tf.reshape(tf.convert_to_tensor(np_array), [1, len(sentence)])
                inputs = tf.nn.embedding_lookup(self.embedding, tensor)
                state = self.cell.zero_state(1, tf.float32)
                outputs = []
                with tf.variable_scope("RNN"):
                    for time_step in range(len(sentence)):
                        if time_step > 0:
                            tf.get_variable_scope().reuse_variables()
                        (cell_output, state) = self.cell(inputs[:, time_step, :], state)
                        # print(cell_output)
                        outputs.append(cell_output)
                output = tf.reshape(tf.concat(1, outputs), [-1, self.config["hidden_size"]])
                logits = tf.matmul(output, self.softmax_w) + self.softmax_b
                # logits = tf.nn.softmax(logits)
                loss = tf.nn.seq2seq.sequence_loss_by_example(
                    [logits],
                    [tf.reshape(target, [-1])],
                    [tf.ones([1 * len(sentence)], dtype=tf.float32)])
                loss = tf.reduce_sum(loss)
                # print(logits.eval(session=self.session))
                # score = tf.reduce_sum(tf.log(logits))
                return loss.eval(session=self.session)

        def get_data(self, sentence):
            sentence = sentence.decode("utf-8")
            sentence_new = []
            for word in sentence.split():
                try:
                    sentence_new.append(self.words_map[word])
                except:
                    sentence_new.append(self.words_map["{{R}}"])
            return sentence_new

        def calculate_score(self, sentence):
            return float(- self.feed_data(self.get_data(sentence)))


        def __str__(self):
            return repr(self)

    instance = None

    @staticmethod
    def initModel(session_dir, words_map_dir, config_dir):
        if not RNN_model.instance:
            RNN_model.instance = RNN_model.__RNN_model(session_dir, words_map_dir, config_dir)
        return RNN_model.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

def test():
    model = RNN_model.initModel(session_dir='/Users/pxxgogo/Desktop/misscut/Miss-Cut-V2/rnn_model/',
                    words_map_dir='/Users/pxxgogo/Desktop/misscut/Miss-Cut-V2/rnn_model/words_map.json',
                    config_dir='/Users/pxxgogo/Desktop/misscut/Miss-Cut-V2/rnn_model/config.json')
    print(model.calculate_score("中 美 执 法 部 门 连 手 成 功 破 获 特 大 跨 国 走 私 武 器 弹 药 案 ， "))





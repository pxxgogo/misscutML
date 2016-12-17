# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2016-12-15 09:54:22
# @Last Modified by:   yuchen
# @Last Modified time: 2016-12-15 11:29:40

from __future__ import print_function
import numpy as np
import sys
import os
import json
import collections
from os import path as op
# from pyspin.spin import make_spin, Spin1
# import showme


class ptb_data_provider(object):

    model_sample = {
        's': {
            'init_scale': 0.1,
            'learning_rate': 1.0,
            'max_grad_norm': 5,
            'num_layers': 2,
            'num_steps': 20,
            'hidden_size': 200,
            'max_epoch': 4,
            'max_max_epoch': 13,
            'keep_prob': 1.0,
            'lr_decay': 0.5,
            'batch_size': 128,
            'vocab_size': 17000
        },
        'm': {
            'init_scale': 0.05,
            'learning_rate': 1.0,
            'max_grad_norm': 5,
            'num_layers': 2,
            'num_steps': 35,
            'hidden_size': 550,
            'max_epoch': 6,
            'max_max_epoch': 39,
            'keep_prob': 0.5,
            'lr_decay': 0.8,
            'batch_size': 128,
            'vocab_size': 17000
        },
        'l': {
            'init_scale': 0.04,
            'learning_rate': 1.0,
            'max_grad_norm': 10,
            'num_layers': 2,
            'num_steps': 35,
            'hidden_size': 1500,
            'max_epoch': 14,
            'max_max_epoch': 55,
            'keep_prob': 0.35,
            'lr_decay': 1. / 1.15,
            'batch_size': 128,
            'vocab_size': 17000
        },
        't': {
            'init_scale': 0.1,
            'learning_rate': 1.0,
            'max_grad_norm': 1,
            'num_layers': 1,
            'num_steps': 2,
            'hidden_size': 2,
            'max_epoch': 1,
            'max_max_epoch': 1,
            'keep_prob': 1.0,
            'lr_decay': 0.5,
            'batch_size': 128,
            'vocab_size': 17000
        },
    }

    def __init__(self):
        self.data_dir = ''

        self.input_compat = raw_input if sys.version_info[0] < 3 else input
        self.filenames = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt', 'words_map.json']
        self.data_dir = ''
        self.model = ''
        self.status = 'IDLE'

        self.num_steps = 0
        self.batch_size = 1
        self.yield_pos = [0, 0, 0]

        self._parse_config()
        self._read_data()

    def _parse_config(self):
        self.legal_model_size = ['s', 'm', 'l', 't']
        try:
            assert op.isfile('config.json') and os.access('config.json', os.R_OK)
            config = json.load(open('config.json', 'r'))
            assert 'data_dir' in config and 'model' in config and 'threshold' in config
            self.data_dir = config['data_dir']
            self.model = config['model']
            self.threshold = config['threshold']
            assert len(self.model) > 0
            self.model = self.model[0].lower()
            assert self.model in self.legal_model_size and op.isdir(self.data_dir) and os.access(self.data_dir, os.R_OK)
            assert isinstance(self.threshold, int)
            for filename in self.filenames:
                fullname = op.join(self.data_dir, filename)
                assert op.isfile(fullname) and os.access(fullname, os.R_OK)
        except AssertionError:
            self.data_dir = ''
            self.model = ''
            self.threshold = -1
            print ("Configure file load failed.")
            cond = False
            while not cond:
                self.data_dir = self.input_compat("Please input data path: ")
                cond, info = self._test_path()
                print (info)
            cond = False
            while not cond:
                self.model = self.input_compat("Please input model size(s/m/l/t): ")
                cond, info = self._test_model()
                print (info)
            cond = False
            while not cond:
                self.threshold = self.input_compat("Please input word threshold: ")
                cond, info = self._test_threshold()
                print (info)
        finally:
            if os.access('config.json', os.R_OK) or (not op.exists('config.json') and os.access('.', os.R_OK)):
                json.dump({
                    'data_dir': self.data_dir,
                    'model': self.model,
                    'threshold': self.threshold,
                }, open('config.json', 'w'))
            self.model_config = ptb_data_provider.model_sample[self.model]
            self.batch_size = self.model_config['batch_size']
            self.num_steps = self.model_config['num_steps']
            print ("OK")

    def _test_path(self):
        if len(self.data_dir) == 0:
            return False, "Data path length should be greater than 0."
        elif not op.isdir(self.data_dir):
            return False, "This path is not a directory."
        elif not os.access(self.data_dir, os.R_OK):
            return False, "This path is not accessible."
        else:
            for filename in self.filenames:
                fullname = op.join(self.data_dir, filename)
                if not op.isfile(fullname):
                    return False, "File {} not found.".format(filename)
                elif not os.access(fullname, os.R_OK):
                    return False, "File {} not accessible"
        return True, "Accepted"

    def _test_model(self):
        if len(self.model) == 0:
            return False, "Model size length should be greater than 0."
        self.model = self.model[0].lower()
        if self.model not in self.legal_model_size:
            return False, "Model size character not acceptable."
        return True, "Accepted"

    def _test_threshold(self):
        if len(self.threshold) == 0:
            return False, "Threshold should be an integer"
        try:
            if isinstance(self.threshold, str):
                self.threshold = int(self.threshold)
            if self.threshold < 0:
                self.threshold = 0
        except:
            return False, "Threshold should be an integer"
        return True, "Accepted"

    def _read_data(self):
        train_path = op.join(self.data_dir, self.filenames[0])
        valid_path = op.join(self.data_dir, self.filenames[1])
        test_path = op.join(self.data_dir, self.filenames[2])
        vocab_path = op.join(self.data_dir, self.filenames[3])
        decoder = lambda x: x.decode('utf-8').replace("\n", "<eos>").split()
        self.vocab = json.load(open(vocab_path, 'r'))
        self.vocab["<eos>"] = len(self.vocab)+1
        self.vocab_nums = self.vocab.values()
        with open(train_path, 'r') as training_source:
            self.training_data = np.array(decoder(training_source.read()))
            self.training_data = self.training_data[: (len(self.training_data) // self.batch_size) * self.batch_size].reshape((self.batch_size, len(self.training_data) // self.batch_size, ))

        with open(valid_path, 'r') as valid_source:
            self.valid_data = np.array(decoder(valid_source.read()))
            self.valid_data = self.valid_data[: (len(self.valid_data) // self.batch_size) * self.batch_size].reshape((self.batch_size, len(self.valid_data) // self.batch_size, ))

        with open(test_path, 'r') as test_source:
            self.test_data = np.array(decoder(test_source.read()))
            self.test_data = self.test_data[: (len(self.test_data) // self.batch_size) * self.batch_size].reshape((self.batch_size, len(self.test_data) // self.batch_size, ))

    def get_config(self):
        return self.model_config

    def get_epoch_size(self):
        if self.status == 'train':
            return (self.training_data.shape[1]) // self.num_steps - 1
        elif self.status == 'valid':
            return (self.valid_data.shape[1]) // self.num_steps - 1
        elif self.status == 'test':
            return (self.test_data.shape[1]) // self.num_steps - 1
        else:
            return None

    def __call__(self):
        self.status = self.status.strip().lower()
        if self.status == 'train':
            self.yield_pos[0] = (self.yield_pos[0] + 1) % self.training_data.shape[1]
            epoch_size = (self.training_data.shape[2]) // self.num_steps - 1
            for i in range(epoch_size):
                x = self.training_data[:, self.yield_pos[0], i * self.num_steps: (i + 1) * self.num_steps]
                y = self.training_data[:, self.yield_pos[0], i * self.num_steps + 1: (i + 1) * self.num_steps + 1]
                yield (x, y)
        elif self.status == 'valid':
            self.yield_pos[1] = (self.yield_pos[1] + 1) % self.valid_data.shape[1]
            epoch_size = (self.valid_data.shape[2]) // self.num_steps - 1
            for i in range(epoch_size):
                x = self.valid_data[:, self.yield_pos[1], i * self.num_steps: (i + 1) * self.num_steps]
                y = self.valid_data[:, self.yield_pos[1], i * self.num_steps + 1: (i + 1) * self.num_steps + 1]
                yield (x, y)
        else:
            self.yield_pos[2] = (self.yield_pos[2] + 1) % self.test_data.shape[0]
            epoch_size = (self.test_data.shape[2]) // self.num_steps - 1
            for i in range(epoch_size):
                x = self.test_data[:, self.yield_pos[2], i * self.num_steps: (i + 1) * self.num_steps]
                y = self.test_data[:, self.yield_pos[2], i * self.num_steps + 1: (i + 1) * self.num_steps + 1]
                yield (x, y)

if __name__ == "__main__":
    '''
    Debug
    '''
    provide = ptb_data_provider()
    provide.status = 'train'
    for x, y in provide():
        print (x, y)
	raw_input("Next")

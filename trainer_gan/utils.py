# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import tensorflow as tf
from data_extractor.utils import read_vocab, dump_vocab

CONFIG_GENERAL = ['batch_size', 'num_epochs', 'learning_rate',
                  'print_step', 'checkpoint_step','trunct_num_batch',
                  'max_ast_depth', 'max_fp_depth', 'max_keywords']
CONFIG_GENERATOR = ['units', 'num_layers']
CONFIG_DISCRIMINATOR = ['units', 'num_layers']


def get_var_list(input):
    generator_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    discriminator_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    var_dict = {'generator_vars': generator_vars,
                'discriminator_vars': discriminator_vars
                }
    return var_dict[input]


# convert JSON to config
def read_config(js, infer=False):
    config = argparse.Namespace()

    for attr in CONFIG_GENERAL:
        config.__setattr__(attr, js[attr])

    config.generator = argparse.Namespace()
    for attr in CONFIG_GENERATOR:
        config.generator.__setattr__(attr, js['generator'][attr])

    config.discriminator = argparse.Namespace()
    for attr in CONFIG_DISCRIMINATOR:
        config.discriminator.__setattr__(attr, js['discriminator'][attr])

    if infer:
        config.vocab = read_vocab(js['vocab'])

    return config


# convert config to JSON
def dump_config(config):
    js = {}
    for attr in CONFIG_GENERAL:
        js[attr] = config.__getattribute__(attr)

    js['generator'] = {attr: config.generator.__getattribute__(attr) for attr in
                       CONFIG_GENERATOR}

    js['discriminator'] = {attr: config.discriminator.__getattribute__(attr) for attr in
                           CONFIG_DISCRIMINATOR}

    js['vocab'] = dump_vocab(config.vocab)

    return js

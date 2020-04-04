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

CONFIG_GENERAL = ['batch_size', 'num_epochs', 'latent_size',
                  'alpha1', 'alpha2', 'beta', 'drop_rate',
                  'learning_rate', 'max_ast_depth', 'max_fp_depth', 'max_keywords',
                  'trunct_num_batch', 'print_step', 'checkpoint_step', ]
CONFIG_ENCODER = ['units', 'num_layers']
CONFIG_DECODER = ['units', 'num_layers']



def get_var_list(input):

    all_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    encoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    var_dict = {'encoder_vars': encoder_vars,
                'decoder_vars': decoder_vars,
                'all_vars': all_vars
                }
    return var_dict[input]


# convert JSON to config
def read_config(js, infer=False):
    config = argparse.Namespace()

    for attr in CONFIG_GENERAL:
        config.__setattr__(attr, js[attr])

    config.encoder = argparse.Namespace()
    for attr in CONFIG_ENCODER:
        config.encoder.__setattr__(attr, js['encoder'][attr])

    config.decoder = argparse.Namespace()
    for attr in CONFIG_DECODER:
        config.decoder.__setattr__(attr, js['decoder'][attr])

    if infer:
        config.vocab = read_vocab(js['vocab'])

    return config


# convert config to JSON
def dump_config(config):
    js = {}

    for attr in CONFIG_GENERAL:
        js[attr] = config.__getattribute__(attr)

    js['encoder'] = {attr: config.encoder.__getattribute__(attr) for attr in
                     CONFIG_ENCODER}

    js['decoder'] = {attr: config.decoder.__getattribute__(attr) for attr in
                     CONFIG_DECODER}

    js['vocab'] = dump_vocab(config.vocab)

    return js


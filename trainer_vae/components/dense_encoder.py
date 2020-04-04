# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unlconfig.vocab.fp_dict_size,ess required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


class DenseEncoder(object):
    def __init__(self, config, inputs, vocab_size, drop_prob=None):

        if drop_prob is None:
            drop_prob = tf.constant(1.0, dtype=tf.float32)

        emb = tf.get_variable('emb_ret', [vocab_size, config.encoder.units])
        emb_inp = tf.nn.embedding_lookup(emb, inputs)

        encoding = tf.layers.dense(emb_inp, config.encoder.units, activation=tf.nn.tanh)
        encoding = tf.layers.dropout(encoding, rate=drop_prob)
        for i in range(config.encoder.num_layers - 1):
            encoding = tf.layers.dense(encoding, config.encoder.units, activation=tf.nn.tanh)
            encoding = tf.layers.dropout(encoding, rate=drop_prob)

        w = tf.get_variable('w', [config.encoder.units, config.latent_size])
        b = tf.get_variable('b', [config.latent_size])
        latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

        zeros = tf.zeros([config.batch_size, config.latent_size])
        condition = tf.not_equal(inputs, 0)

        self.latent_encoding = tf.where(condition, latent_encoding, zeros)

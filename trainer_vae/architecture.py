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
import tensorflow as tf
from trainer_vae.components import TreeEncoder, TreeDecoder, SequenceEncoder, \
    SequenceDecoder, DenseEncoder, DenseDecoder


class Encoder(object):

    def __init__(self, config, nodes, edges, ret, fps):

        self.drop_prob = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), ())

        with tf.variable_scope("Mean", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("ast_tree"):
                self.ast_mean_tree = TreeEncoder(config, nodes, edges, config.vocab.api_dict_size,
                                                 drop_prob=self.drop_prob)
                ast_mean = self.ast_mean_tree.last_output

            with tf.variable_scope("formal_param"):
                self.fp_mean_enc = SequenceEncoder(config, fps, config.vocab.fp_dict_size,
                                                   drop_prob=self.drop_prob)
                fp_mean = self.fp_mean_enc.output

            with tf.variable_scope("ret_type"):
                self.ret_mean_enc = DenseEncoder(config, ret, config.vocab.ret_dict_size,
                                                 drop_prob=self.drop_prob)
                ret_mean = self.ret_mean_enc.latent_encoding

            merged_mean = tf.concat([ast_mean, fp_mean, ret_mean], axis=1)

            layer1 = tf.layers.dense(merged_mean, config.latent_size, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, config.latent_size, activation=tf.nn.tanh)
            layer3 = tf.layers.dense(layer2, config.latent_size)
            self.output_mean = layer3

        with tf.variable_scope("Covariance", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("ast_tree"):
                self.ast_covar_tree = TreeEncoder(config, nodes, edges, config.vocab.api_dict_size,
                                                  drop_prob=self.drop_prob)
                ast_covar = self.ast_covar_tree.last_output

            with tf.variable_scope("formal_param"):
                self.fp_covar_enc = SequenceEncoder(config, fps, config.vocab.fp_dict_size,
                                                    drop_prob=self.drop_prob)
                fp_covar = self.fp_covar_enc.output

            with tf.variable_scope("ret_type"):
                self.ret_covar_enc = DenseEncoder(config, ret, config.vocab.ret_dict_size,
                                                  drop_prob=self.drop_prob)
                ret_covar = self.ret_covar_enc.latent_encoding

            merged_covar = tf.concat([ast_covar, fp_covar, ret_covar], axis=1)

            layer1 = tf.layers.dense(merged_covar, config.latent_size, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, config.latent_size, activation=tf.nn.tanh)
            layer3 = tf.layers.dense(layer2, 1)
            self.output_covar = 1. / (tf.tile(tf.square(layer3), [1, config.latent_size]) + 1)


class Decoder(object):
    def __init__(self, config, nodes, edges, fps, initial_state):

        with tf.variable_scope("ast_tree", reuse=tf.AUTO_REUSE):
            self.ast_tree = TreeDecoder(config, nodes, edges, initial_state, config.vocab.api_dict_size)
            self.ast_logits = self.ast_tree.output_logits

        with tf.variable_scope("formal_param"):
            self.fp_decoder = SequenceDecoder(config, fps, initial_state, config.vocab.fp_dict_size)
            self.fp_logits = self.fp_decoder.output_logits

        # For return type we only need layer 0
        with tf.variable_scope("ret_type"):
            self.ret_decoder = DenseDecoder(config, initial_state[0], config.vocab.ret_dict_size)
            self.ret_logits = self.ret_decoder.logits

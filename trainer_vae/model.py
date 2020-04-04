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

from trainer_vae.architecture import Encoder, Decoder
from trainer_vae.utils import get_var_list
from tensorflow.contrib import seq2seq


class Model:
    def __init__(self, config):
        self.config = config

        self.nodes = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])
        self.edges = tf.placeholder(tf.bool, [self.config.batch_size, self.config.max_ast_depth])
        self.targets = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])

        self.return_type = tf.placeholder(tf.int32, [self.config.batch_size])

        self.formal_params = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_fp_depth])
        self.formal_param_targets = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_fp_depth])

        nodes = tf.unstack(tf.transpose(self.nodes), axis=0)
        edges = tf.unstack(tf.transpose(self.edges), axis=0)
        formal_params = tf.unstack(tf.transpose(self.formal_params), axis=0)

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            self.encoder = Encoder(config, nodes, edges, self.return_type, formal_params)

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            samples = tf.random.normal([config.batch_size, config.latent_size], mean=0., stddev=1., dtype=tf.float32)
            self.latent_state = self.encoder.output_mean + tf.sqrt(self.encoder.output_covar) * samples
            latent_state_lifted = tf.layers.dense(self.latent_state, self.config.decoder.units)
            self.initial_state = [latent_state_lifted] * config.decoder.num_layers
            self.decoder = Decoder(config, nodes, edges, formal_params, self.initial_state)

        # 1. Generator loss for AST API calls
        weights = tf.ones_like(self.targets, dtype=tf.float32) \
                    * tf.cast(tf.greater(self.targets, 0), tf.float32)
        self.ast_gen_loss = tf.reduce_mean(tf.reduce_sum(
            seq2seq.sequence_loss(self.decoder.ast_logits, self.targets,
                                  weights,
                                  average_across_batch=False,
                                  average_across_timesteps=False), axis=1), axis=0)

        # 3. Generator loss for Ret Type node

        ret_logits = tf.expand_dims(self.decoder.ret_logits, axis=1)
        return_targets = tf.expand_dims(self.return_type, axis=1)
        weights = tf.ones_like(return_targets, dtype=tf.float32)
        self.ret_gen_loss = tf.reduce_mean(tf.reduce_sum(
            seq2seq.sequence_loss(ret_logits, return_targets,
                                  weights,
                                  average_across_batch=False,
                                  average_across_timesteps=False), axis=1), axis=0)

        # 3. Generator loss for FP nodes
        weights = tf.ones_like(self.formal_param_targets, dtype=tf.float32) \
                    * tf.cast(tf.greater(self.formal_param_targets, 0), tf.float32)
        self.fp_gen_loss = tf.reduce_mean(tf.reduce_sum(
            seq2seq.sequence_loss(self.decoder.fp_logits,
                                  self.formal_param_targets, weights,
                                  average_across_batch=False,
                                  average_across_timesteps=False), axis=1), axis=0)

        # 2. KL loss: negative of the KL-divergence between P(\Psi | f(\Theta)) and P(\Psi)
        self.KL_loss = tf.reduce_mean(0.5 * tf.reduce_sum(- tf.math.log(self.encoder.output_covar)
                                                          - 1 + self.encoder.output_covar
                                                          + tf.square(-self.encoder.output_mean)
                                                          , axis=1) , axis=0)

        with tf.variable_scope("optimization", reuse=tf.AUTO_REUSE): # TODO: NOTE: this was changed to variable scope from name scope
            opt = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
            self.gen_loss = self.ast_gen_loss + \
                            config.alpha1 * self.ret_gen_loss + \
                            config.alpha2 * self.fp_gen_loss
            self.loss = self.gen_loss + config.beta * self.KL_loss
            gen_train_ops = get_var_list('all_vars')
            self.train_op = opt.minimize(self.loss, var_list=gen_train_ops)


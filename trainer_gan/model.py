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

from trainer_gan.architecture import Generator, Discriminator
from trainer_gan.utils import get_var_list


class Model:
    def __init__(self, config):
        self.config = config

        self.nodes = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])
        self.edges = tf.placeholder(tf.bool, [self.config.batch_size, self.config.max_ast_depth])
        self.keywords = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_keywords])

        nodes = tf.transpose(self.nodes)
        edges = tf.transpose(self.edges)

        nodes = [nodes[i] for i in range(config.max_ast_depth)]
        edges = [edges[i] for i in range(config.max_ast_depth)]

        with tf.variable_scope("generator"):
            self.generator = Generator(config, nodes, edges, self.keywords)

        with tf.variable_scope("discriminator"):
            input_node_logits = tf.one_hot(self.nodes, config.vocab.api_dict_size) * 20 + \
                              tf.random.normal([config.batch_size, config.max_ast_depth, config.vocab.api_dict_size],
                                               mean=0.0, stddev=1)
            input_node_logits = tf.unstack(input_node_logits, axis=1)
            self.real = Discriminator(config, input_node_logits, edges)

        with tf.variable_scope("discriminator", reuse=True):
            self.faker = Discriminator(config, self.generator.output_node_embs, edges)

        opt = tf.compat.v1.train.RMSPropOptimizer(config.learning_rate)

        with tf.name_scope("discriminator_loss"):
            opt1 = tf.compat.v1.train.RMSPropOptimizer(0.0001)
            self.disc_loss = tf.reduce_mean(self.faker.logits - self.real.logits)
            disc_train_ops = get_var_list('discriminator_vars')
            self.disc_train_op = opt1.minimize(self.disc_loss, var_list=disc_train_ops)

        with tf.name_scope("generator_loss"):
            opt2 = tf.compat.v1.train.RMSPropOptimizer(0.0005)
            self.gen_loss = - tf.reduce_mean(self.faker.logits)
            gen_train_ops = get_var_list('generator_vars')
            self.gen_train_op = opt2.minimize(self.gen_loss, var_list=gen_train_ops)

        with tf.name_scope('inference'):
            logits = self.generator.output_node_embs[-1]
            self.ln_probs = tf.nn.log_softmax(logits)
            self.idx = tf.multinomial(logits, 1)
            self.top_k_values, self.top_k_indices = tf.nn.top_k(self.ln_probs, k=config.batch_size)


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


def encode_keywords(config, keywords):
    denom = tf.reduce_sum(tf.cast(tf.greater(keywords, 0), dtype=tf.float32), axis=1)
    keywords = tf.reshape(keywords, [config.batch_size * config.max_keywords])

    emb = tf.get_variable('emb_kw', [config.vocab.keyword_dict_size, config.generator.units])
    keywords_emb = tf.nn.embedding_lookup(emb, keywords)

    layer1 = tf.layers.dense(keywords_emb, config.generator.units, activation=tf.nn.tanh)
    # TODO : make layer2 normmal
    layer2 = tf.layers.dense(layer1, config.generator.units)  #[bs*max_kw, units]

    non_zeros = tf.where(tf.not_equal(keywords, 0), layer2, tf.zeros_like(layer2))
    reshaper = tf.reshape(non_zeros, [config.batch_size, config.max_keywords, config.generator.units])
    summer = tf.reduce_sum(reshaper, axis=1)/denom[:, None]

    return summer


class Generator(object):
    def __init__(self, config, nodes, edges, keywords):

        cells1, cells2 = [], []
        for _ in range(config.generator.num_layers):
            cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.generator.units))
            cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.generator.units))

        with tf.variable_scope('keyword_based_latent_space'):
            self.latent_state = encode_keywords(config, keywords)

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        emb = tf.get_variable('emb', [config.vocab.api_dict_size, config.generator.units])
        curr_out = tf.zeros([config.batch_size, config.generator.units])

        with tf.variable_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [config.generator.units,
                                                                 config.vocab.api_dict_size])
            self.projection_b = tf.get_variable('projection_b', [config.vocab.api_dict_size])

        # placeholders
        self.initial_state = [self.latent_state] * config.generator.num_layers

        with tf.variable_scope('generator_network'):
            self.state = self.initial_state
            curr_state = self.initial_state
            self.output_node_embs = []
            for i, node in enumerate(nodes):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
                edge = edges[i]
                node = nodes[i]
                emb_inp = tf.nn.embedding_lookup(emb, node)
                with tf.variable_scope('cell1'):  # handles node
                    output1, state1 = self.cell1(emb_inp, curr_state)
                with tf.variable_scope('cell2'):  # handles edge
                    output2, state2 = self.cell2(emb_inp, curr_state)

                output = tf.where(edge, output1, output2)
                curr_out = tf.where(tf.not_equal(node, 0), output, curr_out)

                self.state = [tf.where(edge, state1[j], state2[j]) for j in range(config.generator.num_layers)]
                curr_state = [tf.where(tf.not_equal(node, 0), self.state[j], curr_state[j])
                              for j in range(config.generator.num_layers)]

                node_embs = tf.matmul(curr_out, self.projection_w) + self.projection_b
                self.output_node_embs.append(node_embs)


class Discriminator(object):

    def __init__(self, config, node_embs, edges):
        with tf.variable_scope("APITree"):
            tree_enc = TreeEncoder(config, node_embs, edges)
            output_emb = tree_enc.last_output

        layer1 = tf.layers.dense(output_emb, config.discriminator.units)
        self.logits = tf.layers.dense(layer1, 1)
        # self.prob = tf.layers.dense(layer1, 1, activation=tf.nn.sigmoid)


class TreeEncoder(object):
    def __init__(self, config, node_embs, edges):

        batch_size = config.batch_size
        num_layers = config.discriminator.num_layers
        units = config.discriminator.units

        cells1 = []
        cells2 = []
        for _ in range(num_layers):
            cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))
            cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # initial_state has get_shape (batch_size, latent_size), same as psi_mean in the prev code
        curr_state = [tf.random.truncated_normal([batch_size, units], stddev=0.01)] * num_layers
        curr_out = tf.zeros([config.batch_size, config.discriminator.units])

        with tf.variable_scope('recursive_nn'):
            self.state = curr_state
            for i, (node_emb, edge) in enumerate(zip(node_embs, edges)):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
                # node_emb = tf.nn.embedding_lookup(emb, node)

                with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                    output1, state1 = self.cell1(node_emb, curr_state)
                with tf.variable_scope('cell2'):  # handles SIBLING EDGE
                    output2, state2 = self.cell2(node_emb, curr_state)

                # not_zero_condition = tf.not_equal(tf.reduce_sum(node_emb, axis=1), 0)
                output = tf.where(edge, output1, output2)
                # curr_out = tf.where(not_zero_condition, output, curr_out)

                self.state = [tf.where(edge, state1[j], state2[j]) for j in range(num_layers)]
                curr_state = self.state #[tf.where(not_zero_condition, self.state[j], curr_state[j])
                              #for j in range(config.discriminator.num_layers)]

        with tf.name_scope("Output"):
            self.last_output = tf.layers.dense(output, units)

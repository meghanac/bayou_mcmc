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


class TreeEncoder(object):
    def __init__(self, config, nodes, edges, vocab_size, drop_prob=None):

        if drop_prob is None:
            drop_prob = tf.constant(1.0, dtype=tf.float32)

        emb = tf.get_variable('emb_api', [vocab_size, config.encoder.units])
        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in nodes)

        cells1 = []
        cells2 = []
        for _ in range(config.encoder.num_layers):
            cell1 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.encoder.units)
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1,
                                                  state_keep_prob=drop_prob)
            cells1.append(cell1)

            cell2 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.encoder.units)
            cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2,
                                                  state_keep_prob=drop_prob)
            cells2.append(cell2)

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # initial_state has get_shape (batch_size, latent_size), same as psi_mean in the prev code
        curr_state = [tf.random.truncated_normal([config.batch_size, config.encoder.units],
                                                 stddev=0.01)] * config.encoder.num_layers
        curr_out = tf.zeros([config.batch_size, config.encoder.units])

        # projection matrices for output
        with tf.name_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [config.encoder.units, config.latent_size])
            self.projection_b = tf.get_variable('projection_b', [config.latent_size])

        # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
        with tf.variable_scope('recursive_nn'):
            self.state = curr_state
            for i, inp in enumerate(emb_inp):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
                with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                    output1, state1 = self.cell1(inp, curr_state)
                with tf.variable_scope('cell2'):  # handles SIBLING EDGE
                    output2, state2 = self.cell2(inp, curr_state)

                edge = edges[i]
                node = nodes[i]
                output = tf.where(edge, output1, output2)
                curr_out = tf.where(tf.not_equal(node, 0), output, curr_out)

                self.state = [tf.where(edge, state1[j], state2[j]) for j in range(config.encoder.num_layers)]
                curr_state = [tf.where(tf.not_equal(node, 0), self.state[j], curr_state[j])
                              for j in range(config.encoder.num_layers)]

        with tf.name_scope("Output"):
            self.last_output = tf.nn.xw_plus_b(output, self.projection_w, self.projection_b)

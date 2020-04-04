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


class TreeDecoder(object):
    def __init__(self, config, nodes, edges, initial_state, vocab_size):

        cells1, cells2 = [], []
        for _ in range(config.decoder.num_layers):
            cell1 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units)
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1)
            cells1.append(cell1)

            cell2 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units)
            cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2)
            cells2.append(cell2)

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        self.emb = tf.get_variable('emb', [vocab_size, config.decoder.units])
        emb_inp = (tf.nn.embedding_lookup(self.emb, i) for i in nodes)

        with tf.variable_scope("projections"):
            projection_w = tf.get_variable('projection_w', [config.decoder.units,
                                                            vocab_size])
            projection_b = tf.get_variable('projection_b', [vocab_size])

        with tf.variable_scope('decoder_network'):
            self.state = initial_state
            output_logits = []
            for i, inp in enumerate(emb_inp):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
                with tf.variable_scope('cell1'):  # handles node
                    output1, state1 = self.cell1(inp, self.state)
                with tf.variable_scope('cell2'):  # handles edge
                    output2, state2 = self.cell2(inp, self.state)

                edge = edges[i]
                output = tf.where(edge, output1, output2)
                self.state = [tf.where(edge, state1[j], state2[j])
                              for j in range(config.decoder.num_layers)]

                output_projection = tf.matmul(output, projection_w) + projection_b
                output_logits.append(output_projection)

        self.output_logits = tf.stack(output_logits, 1)

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


class SequenceDecoder(object):
    def __init__(self, config, nodes, initial_state, vocab_size):

        cells = []
        for _ in range(config.decoder.num_layers):
            cells.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units))
        self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # setup embedding
        emb = tf.get_variable('emb', [vocab_size, config.decoder.units])
        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in nodes)

        # projection matrices for output
        with tf.variable_scope("projections"):
            projection_w = tf.get_variable('projection_w', [self.cell.output_size,
                                                            vocab_size])
            projection_b = tf.get_variable('projection_b', [vocab_size])
            # tf.summary.histogram("projection_w", self.projection_w)
            # tf.summary.histogram("projection_b", self.projection_b)



        with tf.variable_scope('decoder_network'):
            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            self.state = initial_state
            self.outputs = []
            for i, inp in enumerate(emb_inp):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = self.cell(inp, self.state)
                self.state = [state[j] for j in range(config.decoder.num_layers)]

                output_logits = tf.nn.xw_plus_b(output, projection_w, projection_b)
                self.outputs.append(output_logits)

        self.output_logits = tf.stack(self.outputs, 1)

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


class SequenceEncoder(object):
    def __init__(self, config, inputs, vocab_size, drop_prob=None):

        if drop_prob is None:
            drop_prob = tf.constant(1.0, dtype=tf.float32)

        cell_list = []
        for cell in range(config.encoder.num_layers):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.encoder.units)  # both default behaviors
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 state_keep_prob=drop_prob)
            cell_list.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)

        # inputs is BS * depth, after unstack it is depth * BS
        # inputs = tf.unstack(inputs, axis=1)

        with tf.variable_scope("projections"):
            projection_w = tf.get_variable('projection_w', [config.encoder.units,
                                                            vocab_size])
            projection_b = tf.get_variable('projection_b', [vocab_size])

        curr_state = [tf.truncated_normal([config.batch_size, config.encoder.units],
                                          stddev=0.01)] * config.encoder.num_layers
        curr_out = tf.zeros([config.batch_size, config.encoder.units])

        emb = tf.get_variable('emb_fs', [vocab_size, config.encoder.units])

        with tf.variable_scope('encoder_network'):
            for i, inp in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                emb_inp = tf.nn.embedding_lookup(emb, inp)

                output, out_state = cell(emb_inp, curr_state)
                curr_state = [tf.where(tf.not_equal(inp, 0), out_state[j], curr_state[j])
                              for j in range(config.encoder.num_layers)]
                curr_out = tf.where(tf.not_equal(inp, 0), output, curr_out)

        self.output = tf.nn.xw_plus_b(curr_out, projection_w, projection_b)

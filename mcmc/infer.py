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
import tensorflow as tf
import os
import json
import numpy as np

from trainer_vae.model import Model
from trainer_vae.utils import get_var_list, read_config


class BayesianPredictor(object):

    def __init__(self, save_dir, depth=None, batch_size=None):

        config_file = os.path.join(save_dir, 'config.json')
        with open(config_file) as f:
            self.config = read_config(json.load(f), infer=True)

        if depth is not None:
            self.config.max_ast_depth = 1
            self.config.max_fp_depth = 1

        if batch_size is not None:
            self.config.batch_size = batch_size

        self.model = Model(self.config)
        self.sess = tf.Session()
        self.restore(save_dir)

        with tf.name_scope("ast_inference"):
            ast_logits = self.model.decoder.ast_logits[:, 0, :]
            self.ast_ln_probs = tf.nn.log_softmax(ast_logits)
            self.ast_idx = tf.multinomial(ast_logits, 1)
            self.ast_top_k_values, self.ast_top_k_indices = tf.nn.top_k(self.ast_ln_probs,
                                                                        k=3)

        with tf.name_scope("fp_inference"):
            fp_logits = self.model.decoder.fp_logits[:, 0, :]
            self.fp_ln_probs = tf.nn.log_softmax(fp_logits)
            self.fp_idx = tf.multinomial(fp_logits, 1)
            self.fp_top_k_values, self.fp_top_k_indices = tf.nn.top_k(self.fp_ln_probs,
                                                                      k=self.config.batch_size)

        with tf.name_scope("ret_inference"):
            ret_logits = self.model.decoder.ret_logits
            self.ret_ln_probs = tf.nn.log_softmax(ret_logits)
            self.ret_idx = tf.multinomial(ret_logits, 1)
            self.ret_top_k_values, self.ret_top_k_indices = tf.nn.top_k(self.ret_ln_probs,
                                                                      k=self.config.batch_size)

    def restore(self, save):
        # restore the saved model
        vars_ = get_var_list('all_vars')
        old_saver = tf.compat.v1.train.Saver(vars_)
        ckpt = tf.train.get_checkpoint_state(save)
        old_saver.restore(self.sess, ckpt.model_checkpoint_path)
        return

    def close(self):
        self.sess.close()
        tf.reset_default_graph()
        return

    def get_latent_state(self, nodes, edges, return_type, formal_params):
        feed = {self.model.edges.name: edges, self.model.nodes.name: nodes,
                self.model.return_type: return_type,
                self.model.formal_params: formal_params}
        state = self.sess.run(self.model.latent_state, feed)
        return state

    def get_initial_state(self, nodes, edges, return_type, formal_params):
        feed = {self.model.edges.name: edges, self.model.nodes.name: nodes,
                self.model.return_type: return_type,
                self.model.formal_params: formal_params}
        state = self.sess.run(self.model.initial_state, feed)
        return state

    def get_random_initial_state(self):
        latent_state = np.random.normal(loc=0., scale=1.,
                                        size=(self.config.batch_size, self.config.latent_size))
        initial_state = self.sess.run(self.model.initial_state,
                                      feed_dict={self.model.latent_state: latent_state})
        initial_state = np.transpose(np.array(initial_state), [1, 0, 2])  # batch-first
        return initial_state

    def get_next_ast_state(self, ast_node, ast_edge, ast_state):
        feed = {self.model.nodes.name: np.array(ast_node, dtype=np.int32),
                self.model.edges.name: np.array(ast_edge, dtype=np.bool)}
        for i in range(self.config.decoder.num_layers):
            feed[self.model.initial_state[i].name] = np.array(ast_state[i])

        [ast_state, beam_ids, beam_ln_probs] = self.sess.run(
            [self.model.decoder.ast_tree.state, self.ast_top_k_indices, self.ast_top_k_values], feed)

        return ast_state, beam_ids, beam_ln_probs

    def get_ast_logits(self, ast_node, ast_edge, ast_state):
        feed = {self.model.nodes.name: np.array(ast_node, dtype=np.int32),
                self.model.edges.name: np.array(ast_edge, dtype=np.bool)}
        for i in range(self.config.decoder.num_layers):
            feed[self.model.initial_state[i].name] = np.array(ast_state[i])

        [ast_state, ast_ln_probs] = self.sess.run(
            [self.model.decoder.ast_tree.state, self.ast_ln_probs], feed)

        return ast_state, ast_ln_probs

    def get_next_seq_state(self, seq_node, seq_state):
        feed = {self.model.formal_params.name: np.array(seq_node, dtype=np.int32)}
        for i in range(self.config.decoder.num_layers):
            feed[self.model.initial_state[i].name] = np.array(seq_state[i])

        [fp_state, beam_ids, beam_ln_probs] = self.sess.run(
            [self.model.decoder.fp_decoder.state, self.fp_top_k_indices, self.fp_top_k_values], feed)

        return fp_state, beam_ids, beam_ln_probs


    def get_return_type(self, ret_state):
        feed = {}
        for i in range(self.config.decoder.num_layers):
            feed[self.model.initial_state[i].name] = np.array(ret_state[i])

        [beam_ids, beam_ln_probs] = self.sess.run(
            [self.ret_top_k_indices, self.ret_top_k_values], feed)
        return beam_ids, beam_ln_probs

    # def get_prediction_node(self, node, edge, state):
    #     feed = {self.model.nodes.name: np.array([[self.config.decoder.vocab.api_dict[node]]], dtype=np.int32),
    #             self.model.edges.name: np.array([[edge]], dtype=np.bool), self.model.initial_state.name: state}
    #
    #     [state, idx] = self.sess.run([self.model.decoder.state, self.model.idx], feed)
    #     idx = idx[0][0]
    #     state = state[0]
    #     prediction = self.config.decoder.chars[idx]
    #
    #     return Node(prediction), state

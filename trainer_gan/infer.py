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
import numpy as np
import os
import json
import tensorflow as tf


from trainer_gan.model import Model
from trainer_gan.utils import get_var_list, read_config


HELP = """"""


class BayesianPredictor:

    def __init__(self, save, depth_mod=False, beam_width=None):
        config_file = os.path.join(save, 'config.json')
        with open(config_file) as f:
            self.config = read_config(json.load(f), infer=True)

        if depth_mod:
            self.config.max_ast_depth = 1

        if beam_width is not None:
            self.config.batch_size = beam_width

        self.model = Model(self.config)

        # restore the saved model
        self.sess = tf.Session()
        vars = get_var_list('generator_vars')
        old_saver = tf.compat.v1.train.Saver(vars)
        ckpt = tf.train.get_checkpoint_state(save)
        old_saver.restore(self.sess, ckpt.model_checkpoint_path)

    def close(self):
        self.sess.close()
        tf.reset_default_graph()
        return

    def get_initial_state(self, keywords):
        feed = {self.model.keywords.name: keywords}
        state = self.sess.run(self.model.generator.initial_state, feed)
        return state

    def get_latent_state(self, keywords):
        feed = {self.model.keywords.name: keywords}
        state = self.sess.run(self.model.generator.latent_state, feed)
        return state

    def get_random_initial_state(self):
        latent_state = np.random.normal(loc=0., scale=1.,
                                        size=(self.config.batch_size, self.config.generator.units))
        initial_state = self.sess.run(self.model.generator.initial_state,
                                 feed_dict={self.model.generator.latent_state: latent_state})
        initial_state = np.transpose(np.array(initial_state), [1, 0, 2])  # batch-first
        return initial_state

    def get_next_ast_state(self, last_item, last_edge, states):
        feed = {self.model.nodes.name: np.array(last_item, dtype=np.int32),
                self.model.edges.name: np.array(last_edge, dtype=np.bool)}
        for l in range(self.config.generator.num_layers):
            feed[self.model.generator.initial_state[l].name] = np.array(states[l])

        [states, beam_ids, beam_ln_probs] = self.sess.run(
            [self.model.generator.state, self.model.top_k_indices, self.model.top_k_values], feed)
        return states, beam_ids, beam_ln_probs



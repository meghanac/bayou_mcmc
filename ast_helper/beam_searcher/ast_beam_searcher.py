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

import math

import numpy as np
from copy import deepcopy

from ast_helper.candidate import Candidate

from ast_helper.node import CHILD_EDGE, SIBLING_EDGE, Node

MAX_GEN_UNTIL_STOP = 20

class TreeBeamSearcher:

    def __init__(self, infer_model):
        self.infer_model = infer_model
        self.beam_width = infer_model.config.batch_size
        return

    def beam_search(self, initial_state=None, transpose_state=True):
        # print("init state:", initial_state.shape)
        if initial_state is None:
            initial_state = self.infer_model.get_random_initial_state()
        # print(self.beam_width)
        # print(len(initial_state))
        candies = [Candidate(initial_state[k]) for k in range(self.beam_width)]
        candies[0].log_probabilty = -0.0  # However only 0-th seed is used
        # print(len(candies))
        i = 0
        while True:
            # states was batch_size * LSTM_Decoder_state_size
            candies = self.get_next_output_with_fan_out(candies, transpose_state=transpose_state)

            if self.check_for_all_STOP(candies):  # branch_stack and last_item
                break

            i += 1

            if i == MAX_GEN_UNTIL_STOP:
                break

        candies.sort(key=lambda x: x.log_probabilty, reverse=True)
        return candies

    def check_for_all_STOP(self, candies):
        for candy in candies:
            if candy.rolling:
                return False

        return True

    def get_next_output_with_fan_out(self, candies, transpose_state=True):

        topK = len(candies)
        # print("topk:", topK)
        # print(len(candies))
        # print(len(candies[0]))

        last_item = [[self.infer_model.config.vocab.api_dict[candy.last_item]] for candy in candies]
        # print("last item:", [candy.last_item for candy in candies])
        last_edge = [[candy.last_edge] for candy in candies]
        states = [candy.state for candy in candies]
        if transpose_state:
            states = np.transpose(np.array(states), [1, 0, 2])
        # states = np.array(states)

        states, beam_ids, beam_ln_probs = self.infer_model.get_next_ast_state(last_item, last_edge,
                                                                              states)
        # states = states[0]
        next_nodes = [[self.infer_model.config.vocab.chars_api[idx] for idx in beam] for beam in beam_ids]

        # states is still topK * LSTM_Decoder_state_size
        # next_node is topK * topK
        # node_probs in  topK * topK
        # log_probabilty is topK

        log_probabilty = np.array([candy.log_probabilty for candy in candies])
        length = np.array([candy.length for candy in candies])

        # print("beam ln probs:", beam_ln_probs)
        # print("log probabilities:", log_probabilty)
        # print("probs:", [math.exp(i) for i in log_probabilty])

        for i in range(topK):
            if candies[i].rolling == False:
                length[i] = candies[i].length + 1
            else:
                length[i] = candies[i].length

        for i in range(topK):  # denotes the candidate
            for j in range(topK):  # denotes the items
                if candies[i].rolling == False and j > 0:
                    beam_ln_probs[i][j] = -np.inf
                elif candies[i].rolling == False and j == 0:
                    beam_ln_probs[i][j] = 0.0

        new_probs = log_probabilty[:, None] + beam_ln_probs

        # print("new probs:", new_probs)

        len_norm_probs = new_probs / np.power(length[:, None], 1.0)

        # print("len norm probs:", len_norm_probs)

        rows, cols = np.unravel_index(np.argsort(len_norm_probs, axis=None)[::-1], new_probs.shape)
        rows, cols = rows[:topK], cols[:topK]
        # print(rows)
        # print(cols)

        # rows mean which of the original candidate was finally selected
        new_candies = []
        for row, col in zip(rows, cols):
            new_candy = deepcopy(candies[row])  # candies[row].copy()
            if new_candy.rolling:
                new_candy.state = [states[l][row] for l in range(len(states))]
                new_candy.log_probabilty = new_probs[row][col]
                new_candy.length += 1
                # print("row:", row)
                # print("col:", col)
                # print(len(next_nodes))
                # print(len(next_nodes[0]))
                value2add = next_nodes[row][col]
                # value2add = next_nodes[0][0]
                node2add = Node({"node": "DAPICall", "_call": value2add})
                if new_candy.last_edge == SIBLING_EDGE:
                    new_candy.tree_currNode = new_candy.tree_currNode.add_and_progress_sibling_node(node2add)
                else:
                    new_candy.tree_currNode = new_candy.tree_currNode.add_and_progress_child_node(node2add)

                # before updating the last item lets check for penultimate value
                if new_candy.last_edge == CHILD_EDGE and new_candy.last_item in ['DBranch', 'DExcept', 'DLoop']:
                    new_candy.branch_stack.append(new_candy.tree_currNode)
                    new_candy.last_edge = CHILD_EDGE
                    new_candy.last_item = value2add

                elif value2add in ['DBranch', 'DExcept', 'DLoop']:
                    new_candy.branch_stack.append(new_candy.tree_currNode)
                    new_candy.last_edge = CHILD_EDGE
                    new_candy.last_item = value2add

                elif value2add == 'DStop':
                    if len(new_candy.branch_stack) == 0:
                        new_candy.rolling = False
                    else:
                        new_candy.tree_currNode = new_candy.branch_stack.pop()
                        new_candy.last_item = new_candy.tree_currNode.val
                        new_candy.last_edge = SIBLING_EDGE
                else:
                    new_candy.last_edge = SIBLING_EDGE
                    new_candy.last_item = value2add

            new_candies.append(new_candy)

        return new_candies

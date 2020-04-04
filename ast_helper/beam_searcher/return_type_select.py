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
from copy import deepcopy

from ast_helper.candidate import Candidate
from ast_helper.node import SIBLING_EDGE, Node


class DenseSelector:

    def __init__(self, infer_model):
        self.infer_model = infer_model
        return

    def dense_search(self, initial_state=None):
        if initial_state is None:
            initial_state = self.infer_model.get_random_initial_state()

        initial_state = np.transpose(np.array(initial_state), [1,0,2])
        beam_ids, beam_ln_probs = self.infer_model.get_return_type(initial_state)
        beam = beam_ids[0] # values over batch size are repeated
        predictions = [self.infer_model.config.vocab.chars_ret[idx] for idx in beam]
        return predictions
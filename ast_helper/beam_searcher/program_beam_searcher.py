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

from ast_helper.beam_searcher.ast_beam_searcher import TreeBeamSearcher
from ast_helper.beam_searcher.seq_beam_searcher import SeqBeamSearcher
from ast_helper.beam_searcher.return_type_select import DenseSelector

from ast_helper.ast_traverser import AstTraverser

class ProgramBeamSearcher:

    def __init__(self, infer_model):
        self.infer_model = infer_model
        self.beam_width = infer_model.config.batch_size

        self.tree_beam_searcher = TreeBeamSearcher(infer_model)
        self.seq_beam_searcher = SeqBeamSearcher(infer_model)
        self.ret_type_selector = DenseSelector(infer_model)
        self.ast_traverser = AstTraverser()
        return

    def beam_search(self, initial_state=None):
        if initial_state is None:
            initial_state = self.infer_model.get_random_initial_state()
        ast_candies = self.tree_beam_searcher.beam_search(initial_state=initial_state)
        ast_paths = [self.ast_traverser.depth_first_search(candy.head) for candy in ast_candies]

        fp_candies = self.seq_beam_searcher.beam_search(initial_state=initial_state)
        fp_paths = [self.ast_traverser.depth_first_search(candy.head) for candy in fp_candies]

        ret_types = self.ret_type_selector.dense_search(initial_state=initial_state)

        return ast_paths, fp_paths, ret_types

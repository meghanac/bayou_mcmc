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


from ast_helper.node import SIBLING_EDGE, Node
import numpy as np


class Candidate:
    def __init__(self, initial_state):
        self.tree_currNode = Node({"node": "DSubTree"})
        self.head = self.tree_currNode

        self.last_item = self.tree_currNode.val
        self.last_edge = SIBLING_EDGE
        self.branch_stack = []

        self.length = 1
        self.log_probabilty = -np.inf
        self.state = initial_state

        self.rolling = True


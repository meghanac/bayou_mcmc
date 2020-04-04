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

CHILD_EDGE = True
SIBLING_EDGE = False

class Node():
    def __init__(self, node_js, child=None, sibling=None):
        self.type = node_js['node']
        self.child = child
        self.sibling = sibling
        if self.type in ['DBranch', 'DLoop', 'DExcept', 'DSubTree']:
            self.val = self.type
        elif self.type == 'DAPICall':
            self.val = node_js['_call']
        else:
            self.val = 'Ignore_for_now'

    def add_and_progress_sibling_node(self, node):
        self.sibling = node
        return self.sibling

    def add_and_progress_child_node(self, node):
        self.child = node
        return self.child



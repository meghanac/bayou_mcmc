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

from ast_helper.node import Node


class AstReader:

    def __init__(self):
        return

    def get_ast_from_json(self, js):
        ast = self.get_ast(js, idx=0)
        real_head = Node({"node": "DSubTree"})
        real_head.sibling = ast
        return real_head

    def get_ast(self, js, idx=0):
        i = idx
        curr_Node = Node({"node": "Dummy_Fist_Sibling"})
        head = curr_Node

        while i < len(js):
            if js[i]['node'] == 'DAPICall':
                curr_Node.sibling = Node(js[i])
                curr_Node = curr_Node.sibling
            else:
                break
            i += 1
        if i == len(js):
            curr_Node.sibling = Node({"node": "DAPICall", "_call": "DStop"})
            curr_Node = curr_Node.sibling
            return head.sibling

        node_type = js[i]['node']
        if node_type == 'DBranch':
            nodeC = self.read_DBranch(js[i])

            future = self.get_ast(js, i + 1)
            branching = Node(js[i], child=nodeC, sibling=future)

            curr_Node.sibling = branching
            curr_Node = curr_Node.sibling
            return head.sibling

        if node_type == 'DExcept':
            nodeT = self.read_DExcept(js[i])

            future = self.get_ast(js, i + 1)

            exception = Node(js[i], child=nodeT, sibling=future)
            curr_Node.sibling = exception
            curr_Node = curr_Node.sibling
            return head.sibling

        if node_type == 'DLoop':
            nodeC = self.read_DLoop(js[i])
            future = self.get_ast(js, i + 1)

            loop = Node(js[i], child=nodeC, sibling=future)
            curr_Node.sibling = loop
            curr_Node = curr_Node.sibling

            return head.sibling

    def read_DLoop(self, js_branch):
        # assert len(pC) <= 1
        nodeC = self.get_ast(js_branch['_cond'])  # will have at most 1 "path"
        nodeB = self.get_ast(js_branch['_body'])
        nodeC.child = nodeB

        return nodeC

    def read_DExcept(self, js_branch):
        nodeT = self.get_ast(js_branch['_try'])
        nodeC = self.get_ast(js_branch['_catch'])
        nodeC.child = nodeT

        return nodeC

    def read_DBranch(self, js_branch):
        nodeC = self.get_ast(js_branch['_cond'])  # will have at most 1 "path"
        # assert len(pC) <= 1
        nodeT = self.get_ast(js_branch['_then'])
        # nodeC.child = nodeT
        nodeE = self.get_ast(js_branch['_else'])

        nodeC.sibling = nodeE
        nodeC.child = nodeT

        return nodeC

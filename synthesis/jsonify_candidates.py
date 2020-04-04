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

from ast_helper.node import CHILD_EDGE, SIBLING_EDGE, Node

MAX_GEN_UNTIL_STOP = 20
MAX_AST_DEPTH = 5

class TooLongPathError(Exception):
    pass


class IncompletePathError(Exception):
    pass


class InvalidSketchError(Exception):
    pass

# TODO : This requires work before being production ready
def get_jsons_from_beam_search(self, topK):
    candidates = self.beam_search(topK)

    candidates = [candidate for candidate in candidates if candidate.rolling is False]
    # candidates = candidates[0:1]
    # print(candidates[0].head.breadth_first_search())
    candidate_jsons = [self.paths_to_ast(candidate.head) for candidate in candidates]
    return candidate_jsons


class JSON_Synthesis:

    def __init__(self):
        return

    def paths_to_ast(self, head_node):
        """
        Converts a AST
        :param paths: the set of paths
        :return: the AST
        """
        json_nodes = []
        ast = {'node': 'DSubTree', '_nodes': json_nodes}
        self.expand_all_siblings_till_STOP(json_nodes, head_node.sibling)

        return ast

    def expand_all_siblings_till_STOP(self, json_nodes, head_node):
        """
        Updates the given list of AST nodes with those along the path starting from pathidx until STOP is reached.
        If a DBranch, DExcept or DLoop is seen midway when going through the path, recursively updates the respective
        node type.
        :param nodes: the list of AST nodes to update
        :param path: the path
        :param pathidx: index of path at which update should start
        :return: the index at which STOP was encountered if there were no recursive updates, otherwise -1
        """

        while head_node.val != 'STOP':
            node_value = head_node.val
            astnode = {}
            if node_value == 'DBranch':
                astnode['node'] = node_value
                astnode['_cond'] = []
                astnode['_then'] = []
                astnode['_else'] = []
                self.update_DBranch(astnode, head_node.child)
                json_nodes.append(astnode)
            elif node_value == 'DExcept':
                astnode['node'] = node_value
                astnode['_try'] = []
                astnode['_catch'] = []
                self.update_DExcept(astnode, head_node.child)
                json_nodes.append(astnode)
            elif node_value == 'DLoop':
                astnode['node'] = node_value
                astnode['_cond'] = []
                astnode['_body'] = []
                self.update_DLoop(astnode, head_node.child)
                json_nodes.append(astnode)
            else:
                json_nodes.append({'node': 'DAPICall', '_call': node_value})

            head_node = head_node.sibling

        return

    def update_DBranch(self, astnode, loop_node):
        """
        Updates a DBranch AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        # self.expand_all_siblings_till_STOP(astnode['_cond'], loop_node, pathidx+1)

        astnode['_cond'] = json_nodes = [{'node': 'DAPICall', '_call': loop_node.val}]
        self.expand_all_siblings_till_STOP(astnode['_then'], loop_node.sibling)
        self.expand_all_siblings_till_STOP(astnode['_else'], loop_node.child)
        return

    def update_DExcept(self, astnode, loop_node):
        """
        Updates a DExcept AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_try'], loop_node)
        self.expand_all_siblings_till_STOP(astnode['_catch'], loop_node.child)
        return

    def update_DLoop(self, astnode, loop_node):
        """
        Updates a DLoop AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_cond'], loop_node)
        self.expand_all_siblings_till_STOP(astnode['_body'], loop_node.child)
        return

    # def random_search(self, evidences):
    #
    #     # got the state, to be used subsequently
    #     state = self.get_initial_state(evidences)
    #     start_node = Node("DSubTree")
    #     head, final_state = self.consume_siblings_until_STOP(state, start_node)
    #
    #     return head.sibling

    def consume_siblings_until_STOP(self, state, init_node):
        # all the candidate solutions starting with a DSubTree node
        head = candidate = init_node
        if init_node.val == 'STOP':
            return head

        while True:

            predictionNode, state = self.infer_model.get_prediction(candidate.val, SIBLING_EDGE, state)
            candidate = candidate.addAndProgressSiblingNode(predictionNode)

            prediction = predictionNode.val
            if prediction == 'DBranch':
                candidate.child, state = self.consume_DBranch(state)
            elif prediction == 'DExcept':
                candidate.child, state = self.consume_DExcept(state)
            elif prediction == 'DLoop':
                candidate.child, state = self.consume_DLoop(state)
            # end of inner while

            elif prediction == 'STOP':
                break

        # END OF WHILE
        return head, state

    def consume_DExcept(self, state):
        catchStartNode, state = self.get_prediction('DExcept', CHILD_EDGE, state)

        tryStartNode, state = self.get_prediction(catchStartNode.val, CHILD_EDGE, state)
        tryBranch, state = self.consume_siblings_until_STOP(state, tryStartNode)

        catchBranch, state = self.consume_siblings_until_STOP(state, catchStartNode)

        catchStartNode.child = tryStartNode

        return tryBranch, state

    def consume_DLoop(self, state):
        loopConditionNode, state = self.get_prediction('DLoop', CHILD_EDGE, state)
        loopStartNode, state = self.get_prediction(loopConditionNode.val, CHILD_EDGE, state)
        loopBranch, state = self.consume_siblings_until_STOP(state, loopStartNode)

        loopConditionNode.sibling = Node('STOP')
        loopConditionNode.child = loopBranch

        return loopConditionNode, state

    def consume_DBranch(self, state):
        ifStatementNode, state = self.get_prediction('DBranch', CHILD_EDGE, state)
        thenBranchStartNode, state = self.get_prediction(ifStatementNode.val, CHILD_EDGE, state)

        thenBranch, state = self.consume_siblings_until_STOP(state, thenBranchStartNode)
        ifElseBranch, state = self.consume_siblings_until_STOP(state, ifStatementNode)

        #
        ifElseBranch.child = thenBranch

        return ifElseBranch, state

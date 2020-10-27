import json

from node import Node, CHILD_EDGE, SIBLING_EDGE
from tree_modifier import TreeModifier
from configuration import Configuration

from utils import print_verbose_tree_info
from trainer_vae.utils import read_config


STOP ='DStop'
START = 'DSubTree'
BRANCH = 'DBranch'
LOOP = 'DLoop'
EXCEPT = 'DExcept'


class JSONSynthesis:
    def __init__(self, config_path):
        self.config = Configuration(config_path)
        self.tree_mod = TreeModifier(self.config)

    def convert_list_representation_to_tree(self, data):
        nodes, edges, targets = data
        head = Node(nodes[0], self.config.vocab2node[nodes[0]])
        curr_node = head
        i = 0
        while i < len(nodes) - 1:
            print("\n\n\n i:", i)
            print_verbose_tree_info(head)
            if curr_node.api_name == BRANCH:
                i, curr_node = self.__add_branch_to_tree(nodes, edges, targets, curr_node, i)

            elif curr_node.api_name == EXCEPT or curr_node.api_name == LOOP:
                i, curr_node = self.__add_loop_or_except_to_tree(i, curr_node, nodes, edges, targets)

            else:
                i, curr_node = self.__add_node(i, targets[i], curr_node, edges[i])

        print("final tree:")
        print_verbose_tree_info(head)
        return head

    def __add_node(self, i, api_name, parent, edge):
        new_node = self.tree_mod.create_and_add_node(api_name, parent, edge)
        i += 1
        return i, new_node

    def __add_branch_to_tree(self, nodes, edges, targets, curr_node, i):
        i, cond_node = self.__add_node(i, targets[i], curr_node, edges[i])
        print("cond node:")
        print_verbose_tree_info(curr_node)
        if edges[i] == CHILD_EDGE and nodes[i] == cond_node.api_name:
            i, then_node = self.__add_node(i, targets[i], cond_node, edges[i])
            print("then node")
            print_verbose_tree_info(curr_node)
            i = self.__add_struct_to_tree(i, cond_node, then_node, nodes, edges, targets)
            print_verbose_tree_info(curr_node)
        if edges[i] == SIBLING_EDGE and nodes[i] == cond_node.api_name:
            i, else_node = self.__add_node(i, targets[i], cond_node, edges[i])
            print("else node")
            print_verbose_tree_info(curr_node)
            i = self.__add_struct_to_tree(i, cond_node, else_node, nodes, edges, targets)
            print_verbose_tree_info(curr_node)
        if nodes[i] == BRANCH:
            i, curr_node = self.__add_node(i, targets[i], curr_node, edges[i])
            print('end node')
            print_verbose_tree_info(curr_node)
        print("\n\n\n")
        return i, curr_node

    def __add_loop_or_except_to_tree(self, i, curr_node, nodes, edges, targets):
        struct_name = curr_node.api_name
        i, cond_node = self.__add_node(i, targets[i], curr_node, edges[i])
        print("cond_node")
        print_verbose_tree_info(curr_node)
        if edges[i] == CHILD_EDGE and nodes[i] == cond_node.api_name:
            i, body_node = self.__add_node(i, targets[i], cond_node, edges[i])
            print("body_node")
            print_verbose_tree_info(body_node)
            i = self.__add_struct_to_tree(i, cond_node, body_node, nodes, edges, targets)
            print_verbose_tree_info(curr_node)
        if nodes[i] == struct_name:
            i, curr_node = self.__add_node(i, targets[i], curr_node, edges[i])
            print_verbose_tree_info(curr_node)
        print("\n\n\n")
        return i, curr_node

    def __add_struct_to_tree(self, i, cond_node, last_node, nodes, edges, targets):
        while targets[i] != STOP:
            if last_node.api_name == BRANCH:
                i, last_node = self.__add_branch_to_tree(nodes, edges, targets, last_node, i)
            elif last_node.api_name == LOOP or last_node.api_name == EXCEPT:
                i, last_node = self.__add_loop_or_except_to_tree(i, last_node, nodes, edges, targets)
            else:
                print(nodes[i])
                print(last_node.api_name)
                print(edges[i])
                if nodes[i] == last_node.api_name and edges[i] == SIBLING_EDGE:
                    i, last_node = self.__add_node(i, targets[i], last_node, edges[i])
                # if nodes[i] == cond_node.api_name:
                #     break
        print(targets[i])
        if targets[i] == STOP:
            i, _ = self.__add_node(i, targets[i], last_node, edges[i])
            print_verbose_tree_info(cond_node)
        return i

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

        while head_node is not None and head_node.api_name != STOP:
            # print(head_node.api_name)
            node_value = head_node.api_name
            astnode = {}
            if node_value == BRANCH:
                astnode['node'] = node_value
                astnode['_cond'] = []
                astnode['_then'] = []
                astnode['_else'] = []
                self.update_DBranch(astnode, head_node.child)
                json_nodes.append(astnode)
            elif node_value == EXCEPT:
                astnode['node'] = node_value
                astnode['_try'] = []
                astnode['_catch'] = []
                self.update_DExcept(astnode, head_node.child)
                json_nodes.append(astnode)
            elif node_value == LOOP:
                # print("HERE")
                astnode['node'] = node_value
                astnode['_cond'] = []
                astnode['_body'] = []
                self.update_DLoop(astnode, head_node.child)
                # print(astnode)
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

        astnode['_cond'] = json_nodes = [{'node': 'DAPICall', '_call': loop_node.api_name}]
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
        # print(loop_node.api_name)
        # print(loop_node.child.api_name)
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

    # def consume_siblings_until_STOP(self, state, init_node):
    #     # all the candidate solutions starting with a DSubTree node
    #     head = candidate = init_node
    #     if init_node.val == 'STOP':
    #         return head
    #
    #     while True:
    #
    #         predictionNode, state = self.infer_model.get_prediction(candidate.val, SIBLING_EDGE, state)
    #         candidate = candidate.addAndProgressSiblingNode(predictionNode)
    #
    #         prediction = predictionNode.val
    #         if prediction == 'DBranch':
    #             candidate.child, state = self.consume_DBranch(state)
    #         elif prediction == 'DExcept':
    #             candidate.child, state = self.consume_DExcept(state)
    #         elif prediction == 'DLoop':
    #             candidate.child, state = self.consume_DLoop(state)
    #         # end of inner while
    #
    #         elif prediction == 'STOP':
    #             break
    #
    #     # END OF WHILE
    #     return head, state
    #
    # def consume_DExcept(self, state):
    #     catchStartNode, state = self.get_prediction('DExcept', CHILD_EDGE, state)
    #
    #     tryStartNode, state = self.get_prediction(catchStartNode.val, CHILD_EDGE, state)
    #     tryBranch, state = self.consume_siblings_until_STOP(state, tryStartNode)
    #
    #     catchBranch, state = self.consume_siblings_until_STOP(state, catchStartNode)
    #
    #     catchStartNode.child = tryStartNode
    #
    #     return tryBranch, state
    #
    # def consume_DLoop(self, state):
    #     loopConditionNode, state = self.get_prediction('DLoop', CHILD_EDGE, state)
    #     loopStartNode, state = self.get_prediction(loopConditionNode.val, CHILD_EDGE, state)
    #     loopBranch, state = self.consume_siblings_until_STOP(state, loopStartNode)
    #
    #     loopConditionNode.sibling = Node('STOP')
    #     loopConditionNode.child = loopBranch
    #
    #     return loopConditionNode, state
    #
    # def consume_DBranch(self, state):
    #     ifStatementNode, state = self.get_prediction('DBranch', CHILD_EDGE, state)
    #     thenBranchStartNode, state = self.get_prediction(ifStatementNode.val, CHILD_EDGE, state)
    #
    #     thenBranch, state = self.consume_siblings_until_STOP(state, thenBranchStartNode)
    #     ifElseBranch, state = self.consume_siblings_until_STOP(state, ifStatementNode)
    #
    #     #
    #     ifElseBranch.child = thenBranch
    #
    #     return ifElseBranch, state
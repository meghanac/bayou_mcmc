import argparse
import math
import os
import sys
import textwrap
from copy import deepcopy
import numpy as np
import random
import json
import tensorflow as tf

from mcmc.node import Node, SIBLING_EDGE, CHILD_EDGE, DNODES, DBRANCH, DLOOP, DEXCEPT, START, STOP, EMPTY


class SwapProposal:

    def __init__(self, tree_modifier):
        self.config = tree_modifier.config
        self.tree_mod = tree_modifier
        self.max_num_api = self.config.max_num_api
        self.max_length = self.config.max_length
        self.proposal_dist = 1

        self.curr_prog = None

        # Logging
        self.attempted = 0
        self.accepted = 0

    def random_swap(self, curr_prog):
        """
        Randomly swaps 2 nodes in the current program. Only the node will be swapped, the subtree will be detached and
        added to the node its being swapped with.
        :return: (Node) node that was swapped, (Node) other node that was swapped
        """
        # Temporarily save curr_prog
        self.curr_prog = curr_prog

        # get 2 distinct node positions
        node1, rand_node1_pos = self.__get_random_node_to_swap()
        other_nodes = list(range(1, self.curr_prog.length))
        other_nodes.remove(rand_node1_pos)
        node2, node2_pos = self.__get_random_node_to_swap(given_list=other_nodes)

        if node1 is None or node2 is None:
            return None, None

        # swap nodes
        self.__swap_nodes(node1, node2)

        # Reset self.curr_prog
        self.curr_prog = None

        return curr_prog, node1, node2, self.proposal_dist

    def undo_swap_nodes(self, node1, node2):
        self.__swap_nodes(node1, node2)

    def __get_random_node_to_swap(self, given_list=None):
        """
        Returns a valid node in the program or in the given list of positions of nodes that can be chosen.
        Valid node is one that is not a 'DSubtree' or 'DStop' nodes. Additionally, it must be one that can have a
        sibling node. Nodes that cannot have a sibling node are the condition/catch nodes that occur right after a DLoop
        or DExcept node.
        :param given_list: (list of ints) list of positions (ints) that represent nodes in the program that can
        be selected
        :return: (Node) the randomly selected node, (int) position of the randomly selected node
        """
        # Boolean flag that checks whether a valid node exists in the program or given list. Is computed at the end of
        # the first iteration
        selectable_node_exists_in_program = None

        # Unselectable nodes
        unselectable_nodes = {self.config.vocab2node[START], self.config.vocab2node[STOP],
                              self.config.vocab2node[EMPTY],
                              self.config.vocab2node[DLOOP], self.config.vocab2node[DEXCEPT],
                              self.config.vocab2node[DBRANCH]}

        while True:  # while a valid node exists in the program
            # If no list of nodes is specified, choose a random one from the program
            if given_list is None:
                if self.curr_prog.length > 1:
                    # exclude DSubTree node, randint is [a,b] inclusive
                    rand_node_pos = random.randint(1, self.curr_prog.length - 1)
                else:
                    return None, None
            elif len(given_list) == 0:
                return None, None
            else:
                rand_node_pos = random.choice(given_list)

            node = self.tree_mod.get_node_in_position(self.curr_prog, rand_node_pos)

            # Check validity of selected node
            if node.api_num not in unselectable_nodes:
                return node, rand_node_pos

            # If node is invalid, check if valid node exists in program or given list
            if selectable_node_exists_in_program is None:
                nodes, _ = self.tree_mod.get_vector_representation(self.curr_prog)
                if given_list is not None:
                    nodes = [nodes[i] for i in given_list]
                for i in range(len(nodes)):
                    if i == 0:
                        nodes[i] = 0
                    else:
                        if nodes[i] in unselectable_nodes:
                            nodes[i] = 0
                if sum(nodes) == 0:
                    return None, None
                else:
                    selectable_node_exists_in_program = True

    def __swap_nodes(self, node1, node2):
        """
        Swap given nodes. Only swaps individual nodes and not their subtrees as well.
        :param node1: (Node) node to be swapped
        :param node2: (Node) node to be swapped
        :return:
        """

        # Save parents and parent edges for nodes
        node1_parent = node1.parent
        node2_parent = node2.parent
        node1_edge = node1.parent_edge
        node2_edge = node2.parent_edge

        # If one node is the parent of another
        if node1_parent == node2 or node2_parent == node1:
            if node1_parent == node2:
                parent = node2
                node = node1
            else:
                parent = node1
                node = node2

            # get pointers to parent child and sibling nodes
            parent_edge = node.parent_edge
            if parent_edge == SIBLING_EDGE:
                parent_other_node = parent.child
                parent_other_edge = CHILD_EDGE
            else:
                parent_other_node = parent.sibling
                parent_other_edge = SIBLING_EDGE

            # get grandparent node and edge
            grandparent_node = parent.parent
            grandparent_edge = parent.parent_edge

            # remove nodes from parent
            parent.remove_node(SIBLING_EDGE)
            parent.remove_node(CHILD_EDGE)

            # get pointers to node's child and siblings and remove them
            node_child = node.child
            node_sibling = node.sibling
            node.remove_node(SIBLING_EDGE)
            node.remove_node(CHILD_EDGE)

            # add node to grandparent
            grandparent_node.add_node(node, grandparent_edge)

            # add old parent and other other to new parent node
            node.add_node(parent, parent_edge)
            node.add_node(parent_other_node, parent_other_edge)

            # add node's child and sibling to parent
            parent.add_node(node_child, CHILD_EDGE)
            parent.add_node(node_sibling, SIBLING_EDGE)

        else:
            # remove from parents
            node1_parent.remove_node(node1_edge)
            node2_parent.remove_node(node2_edge)

            # save all siblings and children
            node1_sibling = node1.sibling
            node1_child = node1.child
            node2_sibling = node2.sibling
            node2_child = node2.child

            # remove all siblings and children
            node1.remove_node(SIBLING_EDGE)
            node1.remove_node(CHILD_EDGE)
            node2.remove_node(SIBLING_EDGE)
            node2.remove_node(CHILD_EDGE)

            # add siblings and children to swapped nodes
            node1.add_node(node2_sibling, SIBLING_EDGE)
            node1.add_node(node2_child, CHILD_EDGE)
            node2.add_node(node1_sibling, SIBLING_EDGE)
            node2.add_node(node1_child, CHILD_EDGE)

            # and nodes back to swapped parents
            node1_parent.add_node(node2, node1_edge)
            node2_parent.add_node(node1, node2_edge)



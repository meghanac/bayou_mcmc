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

from node import Node, SIBLING_EDGE, CHILD_EDGE, DNODES, DBRANCH, DLOOP, DEXCEPT, START, STOP, EMPTY


class SwapProposal:

    def __init__(self, tree_modifier, verbose=False, debug=False):
        self.config = tree_modifier.config
        self.tree_mod = tree_modifier
        self.max_num_api = self.config.max_num_api
        self.max_length = self.config.max_length
        self.ln_proposal_dist = 0

        self.curr_prog = None

        # Logging
        self.attempted = 0
        self.accepted = 0
        self.verbose = verbose or debug
        self.debug = debug

    def random_swap(self, curr_prog):
        """
        Randomly swaps 2 nodes in the current program. Only the node will be swapped, the subtree will be detached and
        added to the node its being swapped with.
        :return: (Node) node that was swapped, (Node) other node that was swapped
        """
        # Temporarily save curr_prog
        self.curr_prog = curr_prog

        # get 2 distinct node positions
        node1, node1_pos = self.__get_random_node_to_swap()
        node2, node2_pos = self.__get_valid_node2(node1, node1_pos)

        if node1 is None or node2 is None:
            return None, None, None, None

        # swap nodes
        self.__swap_nodes(node1, node2)

        # Reset self.curr_prog
        self.curr_prog = None

        return curr_prog, node1, node2, self.ln_proposal_dist

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

        return node, rand_node_pos

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

            if node.api_name in {DLOOP, DBRANCH, DEXCEPT} or parent.api_name in {DLOOP, DBRANCH, DEXCEPT}:
                if node.api_name in {DLOOP, DBRANCH, DEXCEPT}:
                    assert parent.child is None

                grandparent = parent.parent
                grandparent_edge = parent.parent_edge

                grandparent.remove_node(grandparent_edge)
                parent.remove_node(node.parent_edge)
                sibling = node.remove_node(SIBLING_EDGE)

                grandparent.add_node(node, grandparent_edge)
                node.add_node(parent, SIBLING_EDGE)
                parent.add_node(sibling, SIBLING_EDGE)
                return

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

            # save all siblings
            node1_sibling = node1.sibling
            node2_sibling = node2.sibling
            node1_child = node1.child
            node2_child = node2.child

            # save children only if swapping APIs
            if node1.api_name in {DBRANCH, DLOOP, DEXCEPT}:
                node1_child = None
            if node2.api_name in {DBRANCH, DLOOP, DEXCEPT}:
                node2_child = None

            # remove all siblings and children
            node1.remove_node(SIBLING_EDGE)
            node2.remove_node(SIBLING_EDGE)

            # remove child if needed
            if node1.api_name not in {DBRANCH, DLOOP, DEXCEPT}:
                node1.remove_node(CHILD_EDGE)
            if node2.api_name not in {DBRANCH, DLOOP, DEXCEPT}:
                node2.remove_node(CHILD_EDGE)

            # add siblings and children to swapped nodes
            node1.add_node(node2_sibling, SIBLING_EDGE)
            node1.add_node(node2_child, CHILD_EDGE)
            node2.add_node(node1_sibling, SIBLING_EDGE)
            node2.add_node(node1_child, CHILD_EDGE)

            # and nodes back to swapped parents
            node1_parent.add_node(node2, node1_edge)
            node2_parent.add_node(node1, node2_edge)

    def calculate_ln_prob_of_move(self):
        return self.ln_proposal_dist

    def __get_valid_node2(self, node1, node1_pos):
        if node1.api_name in {DBRANCH, DLOOP, DEXCEPT}:
            parent_node = node1.parent
            parent_edge = node1.parent_edge
            parent_node.remove_node_save_siblings()

            node2, _ = self.__get_random_node_to_swap()
            if node2 is None:
                return None, None

            parent_node.insert_in_between_after_self(node1, parent_edge)

            node2_pos = self.tree_mod.get_nodes_position(self.curr_prog, node2)

            return node2, node2_pos

        else:
            parent_node = node1.parent
            cfs_nodes = {}
            while parent_node.api_name != START:
                if parent_node.api_name in {DBRANCH, DLOOP, DEXCEPT}:
                    cfs_nodes[parent_node] = self.tree_mod.get_nodes_position(self.curr_prog, parent_node)
                parent_node = parent_node.parent

            other_nodes = list(range(1, self.curr_prog.length))
            other_nodes.remove(node1_pos)
            other_nodes = list(filter(lambda x: x not in set(cfs_nodes.values()), other_nodes))
            return self.__get_random_node_to_swap(given_list=other_nodes)

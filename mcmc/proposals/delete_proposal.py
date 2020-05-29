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


class DeleteProposal:

    def __init__(self, tree_modifier):
        self.config = tree_modifier.config
        self.tree_mod = tree_modifier
        self.max_num_api = self.config.max_num_api
        self.max_length = self.config.max_length
        self.ln_proposal_dist = 0

        # Temporary attributes
        self.curr_prog = None

        # Logging
        self.attempted = 0
        self.accepted = 0

    def delete_random_node(self, curr_prog):
        """
        Deletes a random node in the current program.
        :return: (Node) deleted node, (Node) deleted node's parent node,
        (bool- SIBLING_EDGE or CHILD_EDGE) edge between deleted node and its parent
        """
        # Temporarily save curr_prog
        self.curr_prog = curr_prog

        node, _ = self.__get_deletable_node()
        assert node.api_name != STOP
        parent_node = node.parent
        parent_edge = node.parent_edge

        parent_node.remove_node(parent_edge)

        # If a sibling edge was removed and removed node has sibling node, add that sibling node to parent
        if parent_edge == SIBLING_EDGE and node.sibling is not None:
            sibling = node.sibling
            node.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling, SIBLING_EDGE)

        # Reset self.curr_prog
        self.curr_prog = None

        return curr_prog, node, parent_node, parent_edge, self.ln_proposal_dist

    def undo_delete_random_node(self, node, parent_node, edge):
        """
        Adds back the node deleted from the program in delete_random_node(). Restores program state to what it was
        before delete_random_node() was called.
        :param node: (Node) node that was deleted
        :param parent_node: (Node) parent of the node that was deleted
        :param edge: (bool- SIBLING_EDGE or CHILD_EDGE) edge between deleted node and its parent
        :return:
        """
        sibling = None
        if edge == SIBLING_EDGE:
            if parent_node.sibling is not None:
                sibling = parent_node.sibling
        parent_node.add_node(node, edge)
        if sibling is not None:
            node.add_node(sibling, SIBLING_EDGE)

    def __get_deletable_node(self):
        """
        Returns a random node and its position in the current program that can be deleted without causing any
        fragmentation in the program.
        :return: (Node) selected node, (int) selected node's position
        """
        # exclude DSubTree node, randint is [a,b] inclusive
        rand_node_pos = random.randint(1, self.curr_prog.length - 1)
        node = self.tree_mod.get_node_in_position(self.curr_prog, rand_node_pos)

        # Checks parent edge to prevent deleting half a branch or leave dangling D-nodes
        return node, rand_node_pos

    def calculate_ln_prob_of_move(self):
        return self.ln_proposal_dist

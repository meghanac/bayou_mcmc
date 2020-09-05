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
from numba import jit, cuda

from node import Node, SIBLING_EDGE, CHILD_EDGE, DNODES, DBRANCH, DLOOP, DEXCEPT, START, STOP, EMPTY
from proposals.insertion_proposals import ProposalWithInsertion


class InsertProposal(ProposalWithInsertion):

    def __init__(self, tree_modifier, decoder, tf_session, top_k_prob=0.95, verbose=False, debug=False):
        super().__init__(tree_modifier, decoder, tf_session, top_k_prob, verbose=verbose, debug=debug)

    def add_random_node(self, curr_prog, initial_state):
        """
        Adds a node to a random position in the current program.
        Node is chosen probabilistically based on all the nodes that come before it (DFS).
        :return:
        """
        # Temporarily save curr_prog and initial_state
        self.curr_prog = curr_prog
        self.initial_state = initial_state

        # if tree not at max AST depth, can add a node
        if curr_prog.non_dnode_length >= self.max_num_api or curr_prog.length >= self.max_length:
            return None

        # Get a random position in the tree to be the parent of the new node to be added
        rand_node_pos = random.randint(1, curr_prog.length - 1)  # exclude DSubTree node, randint is [a,b] inclusive
        new_node_parent = self.tree_mod.get_node_in_position(curr_prog, rand_node_pos)

        # Probabilistically choose the node that should appear after selected random parent
        new_node, _, prob = self._get_new_node(new_node_parent, SIBLING_EDGE, verbose=self.debug)

        if new_node is None:
            return None

        added_stop_node = False
        # If a dnode is chosen, grow it out
        if new_node.api_name == DBRANCH:
            ln_prob, added_stop_node = self._grow_dbranch(new_node)
            if ln_prob is not None:
                prob += ln_prob
            else:
                # remove dbranch
                self.undo_add_random_node(new_node, added_stop_node)
                return None
        elif new_node.api_name == DLOOP:
            ln_prob, added_stop_node = self._grow_dloop_or_dexcept(new_node)
            if ln_prob is not None:
                prob += ln_prob
            else:
                # remove dloop
                self.undo_add_random_node(new_node, added_stop_node)
                return None
        elif new_node.api_name == DEXCEPT:
            ln_prob, added_stop_node = self._grow_dloop_or_dexcept(new_node)
            if ln_prob is not None:
                prob += ln_prob
            else:
                # remove dexcept
                self.undo_add_random_node(new_node, added_stop_node)
                return None

        # Reset self.curr_prog and self.initial_state
        self.curr_prog = None
        self.initial_state = None

        return curr_prog, new_node, prob, added_stop_node

    def undo_add_random_node(self, added_node, added_stop_node):
        """
        Undoes add_random_node() and returns current program to state it was in before the given node was added.
        :param added_node: (Node) the node that was added in add_random_node() that is to be removed.
        :return:
        """

        if added_node.api_name in {DBRANCH, DLOOP, DEXCEPT} and added_stop_node:
            assert added_node.sibling.api_name == STOP, "added_stop_node is True but sibling is not stop node"
            added_node.remove_node(SIBLING_EDGE)

        if added_node.sibling is None:
            added_node.parent.remove_node(SIBLING_EDGE)
        else:
            sibling_node = added_node.sibling
            parent_node = added_node.parent
            parent_node.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling_node, SIBLING_EDGE)

        # parent_edge = added_node.parent_edge
        # parent = added_node.parent
        # parent.remove_node(parent_edge, save_neighbors=True)
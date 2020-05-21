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
from mcmc.proposals.insertion_proposals import ProposalWithInsertion


class InsertProposal(ProposalWithInsertion):

    def __init__(self, tree_modifier, decoder, top_k_prob=0.95):
        super().__init__(tree_modifier, decoder, top_k_prob)

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
        new_node_parent = self.tree_mod.get_node_in_position(curr_prog, self.max_length, rand_node_pos)

        # Probabilistically choose the node that should appear after selected random parent
        new_node_idx, prob = self._get_ast_idx(rand_node_pos, SIBLING_EDGE, non_dnode=False)
        new_node_api = self.config.node2vocab[new_node_idx]

        # If a dnode is chosen, grow it out
        if new_node_api == DBRANCH:
            new_node, ln_prob = self._grow_dbranch(new_node_parent)
            prob += ln_prob
        elif new_node_api == DLOOP:
            new_node, ln_prob = self._grow_dloop_or_dexcept(new_node_parent, True)
            prob += ln_prob
        elif new_node_api == DEXCEPT:
            new_node, ln_prob = self._grow_dloop_or_dexcept(new_node_parent, False)
            prob += ln_prob
        else:
            # Add node to parent
            if new_node_parent.sibling is None:
                new_node = self.tree_mod.create_and_add_node(new_node_api, new_node_parent, SIBLING_EDGE)
            else:
                old_sibling_node = new_node_parent.sibling
                new_node_parent.remove_node(SIBLING_EDGE)
                new_node = self.tree_mod.create_and_add_node(new_node_api, new_node_parent, SIBLING_EDGE)
                new_node.add_node(old_sibling_node, SIBLING_EDGE)

        # Reset self.curr_prog and self.initial_state
        self.curr_prog = None
        self.initial_state = None

        return curr_prog, new_node, prob

    def undo_add_random_node(self, added_node):
        """
        Undoes add_random_node() and returns current program to state it was in before the given node was added.
        :param added_node: (Node) the node that was added in add_random_node() that is to be removed.
        :return:
        """
        if added_node.api_name in {DBRANCH, DLOOP, DEXCEPT}:
            dnode_sibling = added_node.sibling
            dnode_parent = added_node.parent
            dnode_parent.remove_node(SIBLING_EDGE)
            dnode_parent.add_node(dnode_sibling, SIBLING_EDGE)

        if added_node.sibling is None:
            added_node.parent.remove_node(SIBLING_EDGE)
        else:
            sibling_node = added_node.sibling
            parent_node = added_node.parent
            added_node.parent.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling_node, SIBLING_EDGE)

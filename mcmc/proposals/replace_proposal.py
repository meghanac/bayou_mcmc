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

class ReplaceProposal(ProposalWithInsertion):

    def __init__(self, tree_modifier, decoder):
        super().__init__(tree_modifier, decoder)
        self.top_k_prob = 1
        # self.ln_proposal_dist = 0

    def replace_random_node(self, curr_prog, initial_state):
        # Temporarily save curr_prog and initial_state
        self.curr_prog = curr_prog
        self.initial_state = initial_state

        # if tree not at max AST depth, can add a node
        if curr_prog.non_dnode_length >= self.max_num_api or curr_prog.length >= self.max_length:
            return None

        # Get a random position in the tree to be the parent of the new node to be added
        rand_node_pos = random.randint(1, curr_prog.length - 1)  # exclude DSubTree node, randint is [a,b] inclusive
        replaced_node = self.tree_mod.get_node_in_position(curr_prog, self.max_length, rand_node_pos)

        # remove node
        parent = replaced_node.parent
        parent_edge = replaced_node.parent_edge
        parent.remove_node(parent_edge)
        sibling = replaced_node.remove_node(SIBLING_EDGE)
        child = replaced_node.remove_node(CHILD_EDGE)

        # Probabilistically choose the node that should appear after selected random parent
        new_node_idx, prob = self._get_ast_idx(rand_node_pos, SIBLING_EDGE, non_dnode=False)
        new_node_api = self.config.node2vocab[new_node_idx]

        # If a dnode is chosen, grow it out
        if new_node_api == DBRANCH:
            new_node, ln_prob = self._grow_dbranch(parent)
            prob += ln_prob
        elif new_node_api == DLOOP:
            new_node, ln_prob = self._grow_dloop_or_dexcept(parent, True)
            prob += ln_prob
        elif new_node_api == DEXCEPT:
            new_node, ln_prob = self._grow_dloop_or_dexcept(parent, False)
            prob += ln_prob
        else:
            new_node = self.tree_mod.create_and_add_node(new_node_api, parent, parent_edge)

        new_node.add_node(sibling, SIBLING_EDGE)
        new_node.add_node()

        # Reset self.curr_prog and self.initial_state
        self.curr_prog = None
        self.initial_state = None

        return curr_prog, new_node, replaced_node, prob

    def undo_replace_random_node(self, curr_prog, new_node, replaced_node):
        parent = new_node.parent
        parent_edge = new_node.parent_edge

        # Remove nodes
        parent.remove_node(parent_edge)
        sibling = new_node.remove_node(SIBLING_EDGE)

        # Add replaced node bac
        parent.add_node(replaced_node, parent_edge)
        replaced_node.add_node(sibling, SIBLING_EDGE)

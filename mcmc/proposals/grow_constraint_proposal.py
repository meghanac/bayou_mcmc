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
from proposals.insertion_proposals import ProposalWithInsertion
from utils import print_verbose_tree_info

MAX_INSERTIONS = 5

class GrowConstraintProposal(ProposalWithInsertion):

    def __init__(self, tree_modifier, decoder, tf_session, verbose=False, debug=False):
        super().__init__(tree_modifier, decoder, tf_session, verbose=verbose, debug=debug)

    def grow_constraint(self, curr_prog, initial_state, constraint_node, num_constraints):
        # Temporarily save curr_prog and initial_state
        self.curr_prog = curr_prog
        self.initial_state = initial_state

        # if tree not at max AST depth, can add a node
        if curr_prog.non_dnode_length >= self.max_num_api or curr_prog.length >= self.max_length:
            return None
        num_insertions = random.randint(1, MAX_INSERTIONS)

        prob = -num_constraints
        first_node = None
        last_node = constraint_node
        num_sibling_nodes_added = 0
        for i in range(num_insertions):
            # Probabilistically choose the node that should appear after selected random parent
            new_node, _, ln_prob = self._get_new_node(last_node, SIBLING_EDGE, verbose=self.debug, grow_new_subtree=True)

            if new_node is None:
                if last_node == constraint_node:
                    return None
                else:
                    return curr_prog, first_node, last_node, prob, num_sibling_nodes_added

            # if Stop node was added, end adding nodes
            if new_node.api_name == STOP:
                # remove stop node
                if new_node.sibling is None:
                    # add probability
                    prob += ln_prob
                    last_node = new_node
                    num_sibling_nodes_added += 1
                else:
                    new_node.parent.remove_node_save_siblings()
                break

            # If a dnode is chosen, grow it out
            if new_node.api_name in {DBRANCH, DLOOP, DEXCEPT}:
                if new_node.api_name == DBRANCH:
                    ln_prob, added_stop_node = self._grow_dbranch(new_node)
                else:
                    ln_prob, added_stop_node = self._grow_dloop_or_dexcept(new_node)

                if ln_prob is not None:
                    # add probability
                    prob += ln_prob
                    last_node = new_node
                    num_sibling_nodes_added += 1
                    if i == 0:
                        first_node = new_node
                    if added_stop_node:
                        return curr_prog, first_node, new_node.sibling, prob, num_sibling_nodes_added + 1
                    else:
                        return curr_prog, first_node, last_node, prob, num_sibling_nodes_added
                else:
                    # remove dnode
                    new_node.parent.remove_node_save_siblings()
                    if last_node == constraint_node:
                        return None
                    else:
                        if first_node is None:
                            return curr_prog, first_node, first_node, prob, 0
                        else:
                            return curr_prog, first_node, last_node, prob, num_sibling_nodes_added

            # add probability
            prob += ln_prob
            last_node = new_node
            num_sibling_nodes_added += 1
            if i == 0:
                first_node = new_node

        # Reset self.curr_prog and self.initial_state
        self.curr_prog = None
        self.initial_state = None

        if first_node is None:
            last_node = None

        return curr_prog, first_node, last_node, prob, num_sibling_nodes_added

    def undo_grown_constraint(self, first_added_node, last_added_node):
        if first_added_node is None and last_added_node is None:
            return

        print("first:", first_added_node.api_name)
        print("last:", last_added_node.api_name)

        parent_node = first_added_node.parent

        if first_added_node == last_added_node:
            print("last == first")
            parent_node.remove_node_save_siblings()
            return
        else:
            for i in range(MAX_INSERTIONS):
                removed_node = parent_node.remove_node_save_siblings()
                if removed_node == last_added_node:
                    return
            raise ValueError("Error: could not undo grown constraint")


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


class GrowConstraintProposal(ProposalWithInsertion):

    def __init__(self, tree_modifier, decoder, tf_session, verbose=False, debug=False):
        super().__init__(tree_modifier, decoder, tf_session, verbose=verbose, debug=debug)

    def grow_constraint(self, curr_prog, initial_state, constraint_node):
        # Temporarily save curr_prog and initial_state
        self.curr_prog = curr_prog
        self.initial_state = initial_state

        # if tree not at max AST depth, can add a node
        if curr_prog.non_dnode_length >= self.max_num_api or curr_prog.length >= self.max_length:
            return None
        max_insertions = 3
        num_insertions = random.randint(1, max_insertions)

        parent_node = constraint_node
        counter = 0
        prob = 0
        for i in range(num_insertions):
            # Probabilistically choose the node that should appear after selected random parent
            new_node, _, ln_prob = self._get_new_node(parent_node, SIBLING_EDGE, verbose=self.debug)

            if new_node is None:
                if parent_node == constraint_node:
                    return None
                else:
                    return curr_prog, parent_node, prob

            # add probability
            prob += ln_prob

            # If a dnode is chosen, grow it out
            if new_node.api_name == DBRANCH:
                ln_prob = self._grow_dbranch(new_node)
                if ln_prob is not None:
                    prob += ln_prob
                    break
                else:
                    # remove dbranch
                    self.undo_grow_constraint(new_node)
                    if parent_node == constraint_node:
                        return None
                    else:
                        return curr_prog, parent_node, prob
            elif new_node.api_name == DLOOP:
                ln_prob = self._grow_dloop_or_dexcept(new_node)
                if ln_prob is not None:
                    prob += ln_prob
                    break
                else:
                    # remove dloop
                    self.undo_grow_constraint(new_node)
                    if parent_node == constraint_node:
                        return None
                    else:
                        return curr_prog, parent_node, prob
            elif new_node.api_name == DEXCEPT:
                ln_prob = self._grow_dloop_or_dexcept(new_node)
                if ln_prob is not None:
                    prob += ln_prob
                    break
                else:
                    # remove dexcept
                    self.undo_grow_constraint(new_node)
                    if parent_node == constraint_node:
                        return None
                    else:
                        return curr_prog, parent_node, prob

            counter += 1
            parent_node = new_node

        # Reset self.curr_prog and self.initial_state
        self.curr_prog = None
        self.initial_state = None

        return curr_prog, constraint_node.sibling, prob

    def undo_grown_constraint(self, added_node):
        if added_node.api_name in {DBRANCH, DLOOP, DEXCEPT} and added_node.sibling.api_name == STOP:
            added_node.remove_node(SIBLING_EDGE)

        if added_node.sibling is None:
            added_node.parent.remove_node(SIBLING_EDGE)
        else:
            sibling_node = added_node.sibling
            parent_node = added_node.parent
            parent_node.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling_node, SIBLING_EDGE)
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
from proposals.grow_constraint_proposal import GrowConstraintProposal
from utils import print_verbose_tree_info

MAX_INSERTIONS = 3


class GrowConstraintUpwardsProposal(GrowConstraintProposal):

    def grow_constraint(self, curr_prog, initial_state, constraint_node, num_constraints):
        # NOTE: last_node will appear before first_node since the tree is growing upwards

        # Temporarily save curr_prog and initial_state
        self.curr_prog = curr_prog
        self.initial_state = initial_state

        # if tree not at max AST depth, can add a node
        if curr_prog.non_dnode_length >= self.max_num_api or curr_prog.length >= self.max_length:
            return None
        # num_insertions = random.randint(1, min(MAX_INSERTIONS, self.max_length - curr_prog.length))
        num_insertions = 1

        prob = -num_constraints
        first_node = None
        last_node = constraint_node
        num_sibling_nodes_added = 0
        for i in range(num_insertions):
            new_node, _, ln_prob = self._get_new_node(last_node.parent, SIBLING_EDGE, verbose=self.debug,
                                                      grow_new_subtree=self.grow_new_subtree)

            if new_node is None:
                if last_node == constraint_node:
                    return None
                else:
                    return curr_prog, last_node, first_node, prob, num_sibling_nodes_added

            # if Stop node was added, end adding nodes
            if new_node.api_name == STOP:
                # remove stop node
                if new_node.sibling is None:
                    # add probability
                    prob += ln_prob
                    last_node = new_node
                    num_sibling_nodes_added += 1
                    if i == 0:
                        first_node = new_node
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
                        return curr_prog, new_node.sibling, first_node, prob, num_sibling_nodes_added + 1
                    else:
                        return curr_prog, last_node, first_node, prob, num_sibling_nodes_added
                else:
                    # remove dnode
                    new_node.parent.remove_node_save_siblings()
                    if last_node == constraint_node:
                        return None
                    else:
                        if first_node is None:
                            return curr_prog, first_node, first_node, prob, 0
                        else:
                            return curr_prog, last_node, first_node, prob, num_sibling_nodes_added

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

        return curr_prog, last_node, first_node, prob, num_sibling_nodes_added

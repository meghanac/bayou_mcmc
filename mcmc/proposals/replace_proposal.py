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


class ReplaceProposal(ProposalWithInsertion):

    def __init__(self, tree_modifier, decoder, tf_session, verbose=False, debug=False):
        super().__init__(tree_modifier, decoder, tf_session, verbose=verbose, debug=debug)
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
        rand_node_pos = random.randint(1, curr_prog.length - 2)  # exclude DSubTree node, randint is [a,b] inclusive
        replaced_node = self.tree_mod.get_node_in_position(curr_prog, rand_node_pos)
        replaced_node_api = replaced_node.api_name

        old_child = None
        added_stop_node = False
        # if a dbranch, dloop or dexcept is getting replaced, remove child nodes
        if replaced_node_api in {DBRANCH, DLOOP, DEXCEPT}:
            print("replaced node api:", replaced_node_api)
            old_child = replaced_node.remove_node(CHILD_EDGE)

        # Probabilistically choose the node that should appear after selected random parent
        new_node, new_node_pos, prob = self._replace_node_api(replaced_node, rand_node_pos, replaced_node.parent_edge)

        # Calculate prob of move
        prob -= math.log(self.curr_prog.length)

        # If a dnode is chosen, grow it out
        if new_node.api_name == DBRANCH:
            ln_prob, added_stop_node = self._grow_dbranch(new_node)
            if ln_prob is not None:
                prob += ln_prob
            else:
                self.undo_replace_random_node(new_node, replaced_node_api, old_child, added_stop_node)
        elif new_node.api_name == DLOOP:
            ln_prob, added_stop_node = self._grow_dloop_or_dexcept(new_node)
            if ln_prob is not None:
                prob += ln_prob
            else:
                self.undo_replace_random_node(new_node, replaced_node_api, old_child, added_stop_node)
        elif new_node.api_name == DEXCEPT:
            ln_prob, added_stop_node = self._grow_dloop_or_dexcept(new_node)
            if ln_prob is not None:
                prob += ln_prob
            else:
                self.undo_replace_random_node(new_node, replaced_node_api, old_child, added_stop_node)

        # print("replace node is pos:", rand_node_pos)
        # print_verbose_tree_info(self.curr_prog)
        # print_verbose_tree_info(curr_prog)

        # Reset self.curr_prog and self.initial_state
        self.curr_prog = None
        self.initial_state = None

        return curr_prog, new_node, replaced_node_api, prob, old_child, added_stop_node

    def undo_replace_random_node(self, new_node, replaced_node_api, old_child, added_stop_node):
        # remove added children if necessary
        if new_node.api_name in {DBRANCH, DEXCEPT, DLOOP}:
            new_node.remove_node(CHILD_EDGE)
            if added_stop_node:
                if new_node.sibling.api_name == STOP:
                    new_node.remove_node(SIBLING_EDGE)
                else:
                    raise ValueError("added_stop_node is True but sibling is not stop node")

        new_node.change_api(replaced_node_api, self.config.vocab2node[replaced_node_api])

        if old_child is not None:
            new_node.add_node(old_child, CHILD_EDGE)

        return new_node

    # def calculate_ln_prob_of_move(self, curr_prog_original, initial_state, added_pos, replaced_node_api, added_edge, is_copy=False):
    #     if not is_copy:
    #         curr_prog = curr_prog_original.copy()
    #     else:
    #         curr_prog = curr_prog_original
    #
    #     # added_node = self.tree_mod.get_node_in_position(curr_prog, added_pos)
    #     #
    #     # # reconstruct original tree
    #     # if (added_node.api_name == DBRANCH and replaced_node_api != DBRANCH) or (
    #     #         added_node.api_name == DLOOP and replaced_node_api != DLOOP) or (
    #     #         added_node.api_name == DEXCEPT and replaced_node_api != DEXCEPT):
    #     #     added_node.remove_node(CHILD_EDGE)
    #     #     added_pos = self.tree_mod.get_nodes_position(curr_prog, added_node)
    #     #
    #     # added_node.change_api(replaced_node_api, self.config.vocab2node[replaced_node_api])
    #     # print("curr prog copy")
    #     # print_verbose_tree_info(curr_prog_copy)
    #
    #     # get original
    #
    #
    #     # return super().calculate_ln_prob_of_move(curr_prog_copy, initial_state, added_pos, added_edge)

    def calculate_reversal_ln_prob(self, curr_prog_original, initial_state, added_pos, replaced_node_api, added_edge,
                                   old_child, added_stop_node, is_copy=False, debug=False):
        if not is_copy:
            curr_prog = curr_prog_original.copy()
        else:
            curr_prog = curr_prog_original

        prev_length = curr_prog.length
        added_node = self.tree_mod.get_node_in_position(curr_prog, added_pos)
        old_node = self.undo_replace_random_node(added_node, replaced_node_api, old_child, added_stop_node)

        if debug:
            print("reversed moved:")
            print_verbose_tree_info(curr_prog)

        prob = self.calculate_ln_prob_of_move(curr_prog, initial_state, added_pos, added_edge, prev_length, is_copy=True)

        # # Calculate prob of move
        # prob -= math.log(curr_prog.length)

        return prob

    # def _get_prob_from_logits(self, curr_prog, initial_state, added_node_pos, added_node, added_edge):
    #     logits = self._get_logits_for_add_node(curr_prog, initial_state, added_node_pos, added_edge)
    #     sorted_logits = np.argsort(-logits)



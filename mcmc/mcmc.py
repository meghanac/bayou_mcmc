import argparse
import math
import os
import textwrap
from copy import deepcopy
import numpy as np
import random
import json
import tensorflow as tf
import sys
from infer import BayesianPredictor

from trainer_vae.model import Model
from trainer_vae.utils import get_var_list
from node import Node, SIBLING_EDGE, CHILD_EDGE, DNODES, DBRANCH, DLOOP, DEXCEPT, START, STOP, EMPTY
from utils import print_verbose_tree_info
from configuration import Configuration, TEMP
from tree_modifier import TreeModifier
from proposals.insert_proposal import InsertProposal
from proposals.delete_proposal import DeleteProposal
from proposals.swap_proposal import SwapProposal
from proposals.add_dnode_proposal import AddDnodeProposal
from proposals.replace_proposal import ReplaceProposal
from proposals.grow_constraint_proposal import GrowConstraintProposal

INSERT = 'insert'
DELETE = 'delete'
SWAP = 'swap'
REPLACE = 'replace'
ADD_DNODE = 'add_dnode'
GROW_CONST = 'grow_constraint'


class TooLongLoopingException(Exception):
    pass


class TooLongBranchingException(Exception):
    pass


class MCMCProgram:
    """

    """

    def __init__(self, save_dir, verbose=False, debug=False):
        """
        Initialize program
        :param save_dir: (string) path to directory in which saved model checkpoints are in
        """
        self.config = Configuration(save_dir)
        self.tree_mod = TreeModifier(self.config)

        # Restore ML model
        self.model = Model(self.config.config_obj)
        self.sess = tf.Session()
        self.restore(save_dir)
        with tf.name_scope("ast_inference"):
            ast_logits = self.model.decoder.ast_logits[:, 0, :]
            self.ast_ln_probs = tf.nn.log_softmax(ast_logits)

        # Initialize variables about program
        self.constraints = []
        self.constraint_nodes = []  # has node numbers of constraints

        self.curr_prog = None
        self.curr_log_prob = -0.0
        self.prev_log_prob = -0.0

        self.initial_state = None
        self.latent_state = None
        self.ret_type = []
        self.fp = [[]]
        self.decoder = None
        self.encoder = None

        # self.proposal_probs = {INSERT: 0.5, DELETE: 0.2, SWAP: 0.1, REPLACE: 0.2, ADD_DNODE: 0.0, GROW_CONST: 0.0}
        # self.proposal_probs = {INSERT: 0.5, DELETE: 0.5, SWAP: 0.00, REPLACE: 0.0, ADD_DNODE: 0.0}
        # self.proposal_probs = {INSERT: 0.1, DELETE: 0.2, SWAP: 0.1, REPLACE: 0.2, ADD_DNODE: 0.0, GROW_CONST: 0.4}
        self.proposal_probs = {INSERT: 0.05, DELETE: 0.05, SWAP: 0.0, REPLACE: 0.0, ADD_DNODE: 0.0, GROW_CONST: 0.9}
        self.proposals = list(self.proposal_probs.keys())
        self.p_probs = [self.proposal_probs[p] for p in self.proposals]
        self.reverse = {INSERT: DELETE, DELETE: INSERT, SWAP: SWAP, REPLACE: REPLACE, ADD_DNODE: DELETE, GROW_CONST:DELETE}

        self.Insert = None
        self.Delete = None
        self.Swap = None
        self.AddDnode = None
        self.Replace = None
        self.GrowConstraint = None

        # Logging  # TODO: change to Logger
        self.accepted = 0
        self.rejected = 0
        self.valid = 0
        self.invalid = 0

        self.posterior_dist = {}

        # Whether to print logs
        self.debug = debug
        self.verbose = (verbose or debug)

    def restore(self, save):
        """
        Restores TF model
        :param save: (string) path to directory in which model checkpoints are stored
        :return:
        """
        # restore the saved model
        vars_ = get_var_list('all_vars')
        old_saver = tf.compat.v1.train.Saver(vars_)
        ckpt = tf.train.get_checkpoint_state(save)
        old_saver.restore(self.sess, ckpt.model_checkpoint_path)
        return

    def add_constraint(self, constraint):
        """
        Updates list of constraints that this program must meet.
        :param constraint: (string) name of api that must appear in the program in order for it to be valid.
        :return:
        """
        try:
            node_num = self.config.vocab2node[constraint]
            if len(self.constraints) < self.config.max_num_api:
                self.constraint_nodes.append(node_num)
                self.constraints.append(constraint)
            else:
                print("Cannot add constraint", constraint, ". Limit reached.")
        except KeyError:
            print("Constraint ", constraint, " is not in vocabulary. Will be skipped.")

    def add_return_type(self, ret_type):
        try:
            ret_num = self.config.rettype2num[ret_type]
            if len(self.ret_type) < self.config.max_num_api:  # Might just be 1
                self.ret_type.append(ret_num)
            else:
                print("Cannot add return type", ret_type, ". Limit reached.")
        except KeyError:
            print("Return type ", ret_type, " is not in the vocabulary. Will be skipped.")

    def add_formal_parameters(self, fp):
        try:
            fp_num = self.config.fp2num[fp]
            if len(self.fp[0]) <= self.config.max_num_api:
                self.fp[0].append(fp_num)
            else:
                print("Cannot add formal parameter", fp, ". Limit reached.")
        except KeyError:
            print("Formal parameter ", fp, "is not in the vocabulary. Will be skipped")

    def init_proposals(self):
        self.Insert = InsertProposal(self.tree_mod, self.decoder, self.sess, verbose=self.verbose, debug=self.debug)
        self.Delete = DeleteProposal(self.tree_mod, verbose=self.verbose, debug=self.debug)
        self.Swap = SwapProposal(self.tree_mod, verbose=self.verbose, debug=self.debug)
        # self.AddDnode = AddDnodeProposal(self.tree_mod, self.decoder)
        self.Replace = ReplaceProposal(self.tree_mod, self.decoder, self.sess, verbose=self.verbose, debug=self.debug)
        self.GrowConstraint = GrowConstraintProposal(self.tree_mod, self.decoder, self.sess, verbose=self.verbose,
                                                     debug=self.debug)

    def init_program(self, constraints, ret_types, fps):
        """
        Creates initial program that satisfies all given constraints.
        :param constraints: (list of strings (api names)) list of apis that must appear in the program for
        it to be valid
        :return:
        """
        # Add given constraints, return types and formal parameters if valid
        for i in constraints:
            self.add_constraint(i)

        for r in ret_types:
            self.add_return_type(r)

        for f in fps:
            self.add_formal_parameters(f)
        if len(self.fp[0]) < self.config.max_num_api:
            for i in range(self.config.max_num_api - len(self.fp[0])):
                self.fp[0].append(0)

        # Initialize tree
        head = self.tree_mod.create_and_add_node(START, None, SIBLING_EDGE)
        self.curr_prog = head

        # Add constraint nodes to tree
        last_node = head
        for i in self.constraints:
            node = self.tree_mod.create_and_add_node(i, last_node, SIBLING_EDGE)
            last_node = node

        # Initialize model states
        self.get_initial_decoder_state()
        # self.get_random_initial_state()

        # Update probabilities of tree
        self.calculate_probability()
        self.prev_log_prob = self.curr_log_prob

        # Initialize proposals
        self.init_proposals()

    def check_validity(self):
        """
        TODO: add logging here
        Check the validity of the current program.
        :return: (bool) whether current program is valid or not
        """
        # Create a list of constraints yet to be met
        constraints = []
        constraints += self.constraint_nodes

        stack = []
        curr_node = self.curr_prog
        last_node = curr_node

        counter = 0

        while curr_node is not None:
            # Update constraint list
            if curr_node.api_num in constraints:
                constraints.remove(curr_node.api_num)

            if counter != 0 and curr_node.api_name == START:
                return False

            # Check that DStop does not have any nodes after it
            if curr_node.api_name == STOP:
                if not (curr_node.sibling is None and curr_node.child is None):
                    if curr_node.parent.api_name != DBRANCH:
                        return False

            # check child edges
            if curr_node.parent_edge == CHILD_EDGE:
                if curr_node.parent.api_name not in {DBRANCH, DLOOP,
                                                     DEXCEPT} and curr_node.parent.parent.api_name not in {DBRANCH,
                                                                                                           DLOOP,
                                                                                                           DEXCEPT}:
                    return False

            # Check that DBranch has the proper form
            if curr_node.api_name == DBRANCH:
                if curr_node.child is None:
                    return False
                if curr_node.child.child is None or curr_node.child.sibling is None \
                        or curr_node.child.child.sibling is None:
                    return False
                if curr_node.child.api_name in (DNODES - {STOP}) or curr_node.child.child.api_name in (DNODES - {STOP}) \
                        or curr_node.child.sibling.api_name in (DNODES - {STOP}):
                    return False
                if curr_node.sibling is None:
                    return False
                # if curr_node.child.child.sibling.api_name != STOP:
                #     return False
                # if curr_node.child.sibling.sibling is None:
                #     return False
                # if curr_node.child.sibling.sibling.api_name != STOP:
                #     return False

            # TODO: basically what is happening is that a DExcept gets added to the end of the program so there's no DStop
            # node but then a node gets added onto the catch node and the required DStop node isn't there
            # easiest solution might just be to allow DStop nodes at the end and discard them when calculating probability

            # Check that DLoop and DExcept have the proper form
            if curr_node.api_name == DLOOP or curr_node.api_name == DEXCEPT:
                if curr_node.child is None:
                    return False
                if curr_node.child.child is None or curr_node.child.sibling is not None:
                    return False
                if curr_node.child.api_name in (DNODES - {STOP}) or curr_node.child.child.api_name in (DNODES - {STOP}):
                    return False
                if curr_node.sibling is None:
                    return False
                # if curr_node.child.child.sibling is None:
                #     return False
                # if curr_node.child.child.sibling.api_name != STOP:
                #     return False

            # Choose next node to explore
            if curr_node.child is not None:
                if curr_node.sibling is not None:
                    stack.append(curr_node.sibling)
                curr_node = curr_node.child
            elif curr_node.sibling is not None:
                curr_node = curr_node.sibling
            else:
                if len(stack) > 0:
                    curr_node = stack.pop()
                else:
                    last_node = curr_node
                    curr_node = None

        # # Last node in program cannot be DStop node
        # # assert last_node.api_name != STOP
        # if last_node.api_name == STOP:
        #     return False

        # Return whether all constraints have been met
        return len(constraints) == 0

    def validate_and_update_program(self, move, ln_proposal_prob, ln_reversal_prob):
        """
        Validate current program and if valid decide whether to accept or reject it.
        :return: (bool) whether to accept or reject current program
        """
        valid = self.check_validity()
        print("valid:", valid)
        if valid:
            self.valid += 1
            return self.accept_or_reject(move, ln_proposal_prob, ln_reversal_prob)

        self.invalid += 1
        return False

    def accept_or_reject(self, move, ln_proposal_prob, ln_reversal_prob):
        """
        Calculates whether to accept or reject current program based on Metropolis Hastings algorithm.
        :return: (bool)
        """
        if self.proposal_probs[self.reverse[move]] != 1.0:
            ln_prob_reverse_move = math.log(self.proposal_probs[self.reverse[move]])
        else:
            ln_prob_reverse_move = 0.0
        ln_prob_move = math.log(self.proposal_probs[move])

        # Calculate acceptance ratio
        alpha = (ln_prob_reverse_move + ln_reversal_prob + self.curr_log_prob) - (
                self.prev_log_prob + ln_prob_move + ln_proposal_prob)
        mu = math.log(random.uniform(0, 1))

        if self.verbose:
            print("accept or reject move:", move)
            print("curr log:", math.exp(self.curr_log_prob))
            print("prev log:", math.exp(self.prev_log_prob))
            print("proposal prob:", math.exp(ln_proposal_prob))
            print("reversal prob:", math.exp(ln_reversal_prob))
            print("move prob:", self.proposal_probs[move])
            print("reverse move prob:", self.proposal_probs[self.reverse[move]])
            print("numerator:", math.exp(ln_prob_reverse_move + ln_reversal_prob + self.curr_log_prob))
            print("denominator:", math.exp(self.prev_log_prob + ln_prob_move + ln_proposal_prob))
            print("alpha:", math.exp(alpha))
            print("mu:", math.exp(mu))

        if mu < alpha:
            self.prev_log_prob = self.curr_log_prob  # TODO: add logging for graph here
            self.accepted += 1
            return True
        else:
            # TODO: add logging
            self.rejected += 1
            return False

    def check_insert(self, added_node, prev_length):
        if added_node.api_name not in DNODES:
            assert self.curr_prog.length == prev_length + 1, "Curr prog length: " + str(
                self.curr_prog.length) + ", prev length: " + str(prev_length) + ", added node length: " + str(
                added_node.length)
        elif added_node.api_name == DBRANCH:
            assert self.curr_prog.length == prev_length + 5 or self.curr_prog.length + 6, "Curr prog length: " + str(
                self.curr_prog.length) + ", prev length: " + str(prev_length) + ", added node length: " + str(
                added_node.length)
        elif added_node.api_name == DLOOP or added_node.api_name == DEXCEPT:
            assert self.curr_prog.length == prev_length + 3 or self.curr_prog.length + 4, "Curr prog length: " + str(
                self.curr_prog.length) + ", prev length: " + str(prev_length) + ", added node length: " + str(
                added_node.length)

    def check_delete(self, node, prev_length):
        if node.api_name not in DNODES:
            assert self.curr_prog.length == prev_length - 1, "Curr prog length: " + str(
                self.curr_prog.length) + ", prev length: " + str(prev_length) + ", deleted node length: " + str(
                node.length)
        elif node.api_name == DBRANCH:
            assert self.curr_prog.length == prev_length - 5 or self.curr_prog.length == prev_length - 6, \
                "Curr prog length: " + str(self.curr_prog.length) + ", prev length: " + str(
                    prev_length) + ", deleted node length: " + str(node.length)
        elif node.api_name == DLOOP or node.api_name == DEXCEPT:
            assert self.curr_prog.length == prev_length - 3 or self.curr_prog.length == prev_length - 4, \
                "Curr prog length: " + str(self.curr_prog.length) + ", prev length: " + str(
                    prev_length) + ", deleted node length: " + str(node.length)

    def insert_proposal(self):
        # Logging and checks
        prev_length = self.curr_prog.length
        self.Insert.attempted += 1

        if self.verbose:
            print("\nADD")
            print("old program:")
            print_verbose_tree_info(self.curr_prog)

        # Add node
        output = self.Insert.add_random_node(self.curr_prog, self.initial_state)
        if output is None:
            return False
        curr_prog, added_node, ln_proposal_prob, added_stop_node = output
        assert curr_prog is not None
        self.curr_prog = curr_prog

        # Calculate reversal probability
        ln_reversal_prob = self.Delete.calculate_ln_prob_of_move(curr_prog.length)

        # Calculate probability of new program
        self.calculate_probability()

        # Print logs
        if self.verbose:
            print("\nnew program:")
            print_verbose_tree_info(self.curr_prog)

        # If no node was added, return False
        if added_node is None:
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False

        # Validate current program
        valid = self.validate_and_update_program(INSERT, ln_proposal_prob, ln_reversal_prob)

        # Undo move if not valid
        if not valid:
            self.Insert.undo_add_random_node(added_node, added_stop_node) # TODO: think about this
            self.curr_log_prob = self.prev_log_prob
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False

        # Check that insertion was valid and that there aren't any bugs
        self.check_insert(added_node, prev_length)

        # Logging
        self.Insert.accepted += 1

        # successful
        return True

    def replace_proposal(self):
        # Logging and checks
        prev_length = self.curr_prog.length
        self.Replace.attempted += 1

        if self.verbose:
            print("\nREPLACE")
            print("old program:")
            print_verbose_tree_info(self.curr_prog)

        # Add node
        output = \
            self.Replace.replace_random_node(self.curr_prog, self.initial_state)

        if output is None:
            return False
        prog, new_node, replaced_node_api, ln_proposal_prob, old_child, added_stop_node = output

        # If no node was added, return False
        if new_node is None or prog is None:
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False
        self.curr_prog = prog

        # Calculate reversal prob
        new_node_pos = self.tree_mod.get_nodes_position(self.curr_prog, new_node)
        ln_reversal_prob = self.Replace.calculate_reversal_ln_prob(self.curr_prog, self.initial_state, new_node_pos,
                                                                   replaced_node_api, new_node.parent_edge, old_child, added_stop_node)

        # Calculate probability of new program
        self.calculate_probability()

        # Print logs
        if self.verbose:
            print("new program:")
            print_verbose_tree_info(self.curr_prog)

        # Validate current program
        valid = self.validate_and_update_program(REPLACE, ln_proposal_prob, ln_reversal_prob)

        # Undo move if not valid
        if not valid:
            self.Replace.undo_replace_random_node(new_node, replaced_node_api, old_child, added_stop_node)
            self.curr_log_prob = self.prev_log_prob
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False

        # # Check that insertion was valid and that there aren't any bugs
        # self.check_replace(new_node, replaced_node, prev_length)

        # Logging
        self.Replace.accepted += 1

    def delete_proposal(self):
        # Logging and checks
        prev_length = self.curr_prog.length
        self.Delete.attempted += 1

        # Cannot delete any nodes if it will result in a tree with just DSubtree
        if prev_length <= 2:
            return False

        # Delete node
        curr_prog, node, parent_node, parent_edge, ln_prob = self.Delete.delete_random_node(self.curr_prog)
        if curr_prog is None or node is None:
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False
        self.curr_prog = curr_prog
        parent_pos = self.tree_mod.get_nodes_position(self.curr_prog, parent_node)

        # Calculate probability of reverse move
        curr_prog_copy = self.curr_prog.copy()
        parent_node_copy = self.tree_mod.get_node_in_position(curr_prog_copy, parent_pos)
        parent_node_copy_neighbor = parent_node_copy.get_neighbor(parent_edge)
        node_copy = node.copy()
        parent_node_copy.add_node(node_copy, parent_edge)
        node_copy.add_node(parent_node_copy_neighbor, parent_edge)
        node_pos = self.tree_mod.get_nodes_position(curr_prog_copy, node_copy)
        ln_reversal_prob = self.Insert.calculate_ln_prob_of_move(curr_prog_copy, self.initial_state, node_pos,
                                                                 parent_edge, prev_length, is_copy=True)
        # parent_node_copy.remove_node(parent_edge)

        # Calculate probability of new program
        self.calculate_probability()

        # Print logs
        if self.verbose:
            print("\nDELETE")
            print("old program:")
            print_verbose_tree_info(curr_prog_copy)
            print("new program:")
            print_verbose_tree_info(self.curr_prog)


        # Undo move if not valid
        if not self.validate_and_update_program(DELETE, ln_prob, ln_reversal_prob):
            self.Delete.undo_delete_random_node(node, parent_node, parent_edge)
            self.curr_log_prob = self.prev_log_prob
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False

        # Check that insertion was valid and that there aren't any bugs
        self.check_delete(node, prev_length)

        # Logging
        self.Delete.accepted += 1

        # successful
        return True

    def swap_proposal(self):
        # Logging and checks
        prev_length = self.curr_prog.length
        self.Swap.attempted += 1

        if self.verbose:
            print("\nSWAP")
            print("old program:")
            print_verbose_tree_info(self.curr_prog)

        # Swap nodes
        curr_prog, node1, node2, ln_prob = self.Swap.random_swap(self.curr_prog)
        if curr_prog is None:
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False
        # Undo move if invalid
        if node1 is None or node2 is None:
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False
        self.curr_prog = curr_prog

        # Calculate probability of reversal
        reversal_ln_prob = self.Swap.calculate_ln_prob_of_move()

        # Calculate probability of new program
        self.calculate_probability()

        # Print logs
        if self.verbose:
            print("new program:")
            print_verbose_tree_info(self.curr_prog)

        # Undo move if not valid
        if not self.validate_and_update_program(SWAP, ln_prob, reversal_ln_prob):
            self.Swap.undo_swap_nodes(node1, node2)
            self.curr_log_prob = self.prev_log_prob
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False

        # Check that there are no bugs in implementation
        assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
            self.curr_prog.length) + " != prev length: " + str(prev_length)

        # Logging
        self.Swap.accepted += 1

        # successful
        return True

    def grow_constraint_proposal(self):
        # Logging and checks
        prev_length = self.curr_prog.length
        self.GrowConstraint.attempted += 1

        if self.verbose:
            print("\nGROW CONSTRAINT")
            print("old program:")
            print_verbose_tree_info(self.curr_prog)

        api = random.choice(self.constraints)
        constraint_node = self.tree_mod.get_node_with_api(self.curr_prog, api)

        output = self.GrowConstraint.grow_constraint(self.curr_prog, self.initial_state, constraint_node,
                                                     len(self.constraints))

        if output is None:
            return False

        curr_prog, first_added_node, last_added_node, ln_proposal_prob, num_sibling_nodes_added = output

        if first_added_node is None or last_added_node is None or num_sibling_nodes_added == 0:
            return False

        ln_reversal_prob = 0.0
        total_nodes_added = first_added_node.length - last_added_node.length + 1
        curr_length = self.curr_prog.length
        curr_node = first_added_node
        for _ in range(num_sibling_nodes_added):
            # probability of choosing curr_node to delete
            ln_reversal_prob += self.Delete.calculate_ln_prob_of_move(curr_length)

            # calculate the length of the remaining tree once deleting curr_node
            child_length = 0
            if curr_node.child is not None:
                child_length = curr_node.child.length
            curr_length -= 1 + child_length

            # update curr_node to next added node
            curr_node = first_added_node.sibling

        # Calculate probability of new program
        self.calculate_probability()

        # Print logs
        if self.verbose:
            print("new program:")
            print_verbose_tree_info(self.curr_prog)

        # Undo move if not valid
        if not self.validate_and_update_program(GROW_CONST, ln_proposal_prob, ln_reversal_prob):
            self.GrowConstraint.undo_grown_constraint(first_added_node, last_added_node)  # TODO: think about this
            self.curr_log_prob = self.prev_log_prob
            assert self.curr_prog.length == prev_length, "Curr prog length: " + str(
                self.curr_prog.length) + " != prev length: " + str(prev_length)
            return False

        # Logging
        self.GrowConstraint.accepted += 1

        # successful
        return True

    # TODO: UNCOMMENT AFTER UPATING CODE
    # def add_dnode_proposal(self, verbose):
    #     # Logging and checks
    #     self.AddDnode.attempted += 1
    #
    #     # Add dnode
    #     self.curr_prog, dnode, ln_proposal_prob = self.AddDnode.add_random_dnode(self.curr_prog, self.initial_state)
    #     reversal_ln_prob = self.Delete.calculate_ln_prob_of_move(self.curr_prog.length)
    #
    #     # Calculate probability of new program
    #     self.calculate_probability()
    #
    #     # Print logs
    #     if verbose:
    #         print_verbose_tree_info(self.curr_prog)
    #
    #     # If not valid, return False
    #     if dnode is None:
    #         return False
    #     if not self.validate_and_update_program(ADD_DNODE, ln_proposal_prob, reversal_ln_prob):
    #         self.AddDnode.undo_add_random_dnode(dnode)
    #         self.curr_log_prob = self.prev_log_prob
    #         return False
    #
    #     # Logging
    #     self.AddDnode.accepted += 1
    #
    #     # successful
    #     return True

    def transform_tree(self):
        """
        Randomly chooses a transformation and transforms the current program if it is accepted.
        :return: (bool) whether current tree was transformed or not
        """
        assert self.check_validity() is True

        move = np.random.choice(self.proposals, p=self.p_probs)

        if move == INSERT:
            return self.insert_proposal()
        elif move == DELETE:
            return self.delete_proposal()
        elif move == SWAP:
            return self.swap_proposal()
        elif move == ADD_DNODE:
            pass
            # return self.add_dnode_proposal(verbose)
        elif move == REPLACE:
            return self.replace_proposal()
        elif move == GROW_CONST:
            return self.grow_constraint_proposal()
        else:
            raise ValueError('move not defined')  # TODO: remove once tested

    def update_random_intial_state(self):
        self.initial_state, self.latent_state = self.decoder.update_random_initial_state(self.latent_state)


    def get_random_initial_state(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--continue_from', type=str, default=self.config.save_dir,
                            help='ignore config options and continue training model checkpointed here')
        clargs = parser.parse_args()
        self.get_initial_decoder_state()
        beam_width = 1
        self.decoder = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)
        self.initial_state, self.latent_state = self.decoder.get_random_initial_state()

    def get_initial_decoder_state(self):
        """
        Get initial state of the decoder given the encoder's latent state
        :return:
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--continue_from', type=str, default=self.config.save_dir,
                            help='ignore config options and continue training model checkpointed here')
        clargs = parser.parse_args()

        self.encoder = BayesianPredictor(clargs.continue_from, batch_size=1)

        nodes, edges = self.tree_mod.get_vector_representation(self.curr_prog)
        nodes = nodes[:self.config.max_num_api]
        edges = edges[:self.config.max_num_api]
        nodes = np.array([nodes])
        edges = np.array([edges])

        self.initial_state = self.encoder.get_initial_state(nodes, edges, np.array(self.ret_type), np.array(self.fp))
        self.initial_state = np.transpose(np.array(self.initial_state), [1, 0, 2])  # batch_first

        beam_width = 1
        self.decoder = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)

    def update_latent_state_and_decoder_state(self):
        nodes, edges = self.tree_mod.get_vector_representation(self.curr_prog)
        nodes = nodes[:self.config.max_num_api]
        edges = edges[:self.config.max_num_api]
        nodes = np.array([nodes])
        edges = np.array([edges])
        self.initial_state = self.encoder.get_initial_state(nodes, edges, np.array(self.ret_type), np.array(self.fp))
        self.initial_state = np.transpose(np.array(self.initial_state), [1, 0, 2])  # batch_first

    # def calculate_probability(self):
    #     """
    #     Calculate probability of current program.
    #     :return:
    #     """
    #     nodes, edges = self.tree_mod.get_vector_representation(self.curr_prog)
    #     node = np.zeros([self.config.batch_size, self.config.max_ast_depth], dtype=np.int32)
    #     edge = np.zeros([self.config.batch_size, self.config.max_ast_depth], dtype=np.bool)
    #     state = self.initial_state
    #     curr_prob = 0.0
    #
    #     for i in range(self.curr_prog.length):
    #         node[0][0] = nodes[i]
    #         edge[0][0] = edges[i]
    #         if i == self.curr_prog.length - 1:
    #             if self.config.node2vocab[node[0][0]] == STOP:
    #                 pass
    #             else:
    #                 # add prob of stop node
    #                 # logits are normalized with log_softmax
    #                 state, ast_prob = self.decoder.get_ast_logits(node, edge, state)
    #                 stop_node = self.config.vocab2node[STOP]
    #                 curr_prob += ast_prob[0][stop_node]
    #         else:
    #             state, ast_prob = self.decoder.get_ast_logits(node, edge, state)
    #             curr_prob += ast_prob[0][nodes[i + 1]]
    #
    #     self.curr_log_prob = curr_prob / self.curr_prog.length
    #
    #     return self.curr_log_prob

    def calculate_probability(self):
        """
        Calculate probability of current program.
        :return:
        """
        nodes, edges, targets = self.tree_mod.get_nodes_edges_targets(self.curr_prog)
        node = np.zeros([self.config.batch_size, self.config.max_ast_depth], dtype=np.int32)
        edge = np.zeros([self.config.batch_size, self.config.max_ast_depth], dtype=np.bool)
        state = self.initial_state
        curr_prob = 0.0

        for i in range(len(nodes)):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            # logits are normalized with log_softmax
            state, ast_prob = self.decoder.get_ast_logits(node, edge, state)
            curr_prob += ast_prob[0][targets[i]]

        self.curr_log_prob = curr_prob
        # self.curr_log_prob = curr_prob - math.log(self.curr_prog.length)

        if self.verbose:
            print(nodes)
            print(edges)
            print(targets)
            print(curr_prob)

        if self.debug:
            print("COULD BE:", math.exp(self.curr_log_prob - math.log(self.curr_prog.length)))
            print("NOW:", math.exp(self.curr_log_prob))

        return self.curr_log_prob

    def mcmc(self):
        """
        Perform one MCMC step.
        1) Try to transform program tree.
        2) If accepted, update the latent space.
        3) Do a random walk in the latent space.
        4) Compute the new initial state of the decoder.
        :return:
        """
        curr_prog = self.tree_mod.get_nodes_edges_targets(self.curr_prog)
        transformed = self.transform_tree()

        # make sure all undos work correctly
        new_prog = self.tree_mod.get_nodes_edges_targets(self.curr_prog)
        if transformed:
            assert new_prog != curr_prog, "Program was transformed but actually remained the same."
        else:
            assert  new_prog == curr_prog, "Program was not transformed yet somehow changed."

        self.update_latent_state_and_decoder_state()

        if new_prog in self.posterior_dist:
            self.posterior_dist[new_prog] += 1
        else:
            self.posterior_dist[new_prog] = 1

        # self.update_random_intial_state()

        # self.initial_state = self.decoder.get_random_initial_state()
        # self.update_latent_state_and_decoder_state()
        # if random.choice([True, False, False, False, False, False, False, False, False, False ]):
        #     self.update_latent_state_and_decoder_state()

        # # Attempt to transform the current program
        # if self.transform_tree():
        # # If successful, update encoder's latent state
        #     self.update_latent_state()
        # # self.transform_tree()
        # self.random_walk_latent_space()
        # self.get_initial_decoder_state()

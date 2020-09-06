import argparse
import datetime
import math
import os
import sys
import textwrap
from copy import deepcopy
import time
# import dill as pickle

from numba import jit, vectorize, njit

import multiprocessing
from joblib import Parallel, delayed

import numpy as np
import random
import json


from node import Node, SIBLING_EDGE, CHILD_EDGE, DNODES, DBRANCH, DLOOP, DEXCEPT, START, STOP, EMPTY
from utils import print_verbose_tree_info
from configuration import TEMP

from infer import BayesianPredictor


class ProposalWithInsertion:
    def __init__(self, tree_modifier, decoder, tf_session, top_k_prob=0.95, verbose=False, debug=False):
        self.decoder = decoder
        self.config = tree_modifier.config
        self.tree_mod = tree_modifier
        self.max_num_api = self.config.max_num_api
        self.max_length = self.config.max_length
        self.top_k_prob = top_k_prob

        # Temporary attributes
        self.curr_prog = None
        self.initial_state = None

        # Logging
        self.attempted = 0
        self.accepted = 0
        self.verbose = (verbose or debug)
        self.debug = debug

        self.use_multinomial = True

        self.sess = tf_session

        self.probs = np.zeros([self.config.vocab_size], dtype=float)

    def _grow_dbranch(self, dbranch):
        """
        Create full DBranch (DBranch, condition, then, else) from parent node.
        :param parent: (Node) parent of DBranch
        :return: (Node) DBranch node
        """
        ln_prob = 0

        # Ensure adding a DBranch won't exceed max depth
        if self.curr_prog.non_dnode_length + 3 > self.max_num_api or self.curr_prog.length + 5 > self.max_length:
            return None, None

        # Create condition as DBranch child
        condition, cond_pos, prob = self._get_new_node(dbranch, CHILD_EDGE, verbose=self.debug)
        assert cond_pos > 0, "Error: Condition node position couldn't be found"
        ln_prob += prob

        # # Add then api as child to condition node
        # then_node, _, prob = self._get_new_node(condition, CHILD_EDGE, verbose=verbose)
        # self.tree_mod.create_and_add_node(STOP, then_node, SIBLING_EDGE)
        # ln_prob += prob
        #
        # # Add else api as sibling to condition node
        # else_node, else_pos, prob = self._get_new_node(condition, SIBLING_EDGE, verbose=verbose)
        # self.tree_mod.create_and_add_node(STOP, else_node, SIBLING_EDGE)
        # ln_prob += prob

        for edge in [CHILD_EDGE, SIBLING_EDGE]:
            parent_node = condition
            counter = 0
            while parent_node.api_name != STOP and counter < 3:
                # Add then api as child to condition node
                parent_node, _, prob = self._get_new_node(parent_node, edge, verbose=self.debug)
                ln_prob += prob
                counter += 1

            if parent_node.api_name != STOP:
                self.tree_mod.create_and_add_node(STOP, parent_node, SIBLING_EDGE)

        added_stop_node = False
        if dbranch.sibling is None:
            self.tree_mod.create_and_add_node(STOP, dbranch, SIBLING_EDGE)
            added_stop_node = True

        return ln_prob, added_stop_node

    def _grow_dloop_or_dexcept(self, dnode):
        """
        Create full DLoop (DLoop, condition, body) from parent node
        :param parent: (Node) parent of DLoop
        :return: (Node) DLoop node
        """
        ln_prob = 0

        # Ensure adding a DBranch won't exceed max depth
        if self.curr_prog.non_dnode_length + 2 > self.max_num_api or self.curr_prog.length + 3 > self.max_length:
            return None, None

        parent_node = dnode
        counter = 0
        while parent_node.api_name != STOP and counter < 2:
            parent_node, cond_pos, prob = self._get_new_node(parent_node, CHILD_EDGE, verbose=self.debug)
            ln_prob += prob
            counter += 1
        if parent_node.api_name != STOP:
            self.tree_mod.create_and_add_node(STOP, parent_node, SIBLING_EDGE)

        # # Create condition as DLoop child
        # condition, cond_pos, prob = self._get_new_node(dnode, CHILD_EDGE, verbose=verbose)
        # assert cond_pos > 0, "Error: Condition node position couldn't be found"
        # ln_prob += prob
        #
        # # Add body api as child to condition node
        # then_node, _, prob = self._get_new_node(condition, CHILD_EDGE, verbose=verbose)
        # self.tree_mod.create_and_add_node(STOP, then_node, SIBLING_EDGE)
        # ln_prob += prob

        added_stop_node = False
        if dnode.sibling is None:
            self.tree_mod.create_and_add_node(STOP, dnode, SIBLING_EDGE)
            added_stop_node = True

        return ln_prob, added_stop_node

    def _get_valid_random_node(self, given_list=None):
        """
        Returns a valid node in the program or in the given list of positions of nodes that can be chosen.
        Valid node is one that is not a 'DSubtree' or 'DStop' nodes. Additionally, it must be one that can have a
        sibling node. Nodes that cannot have a sibling node are the condition/catch nodes that occur right after a DLoop
        or DExcept node.
        :param given_list: (list of ints) list of positions (ints) that represent nodes in the program that can
        be selected
        :return: (Node) the randomly selected node, (int) position of the randomly selected node
        """
        # Boolean flag that checks whether a valid node exists in the program or given list. Is computed at the end of
        # the first iteration
        selectable_node_exists_in_program = None

        # Parent nodes whose children are invalid
        unselectable_parent_dnodes = {self.config.vocab2node[DLOOP],
                                      self.config.vocab2node[
                                          DEXCEPT]}  # TODO: make sure I can actually remove DBranch node

        # Unselectable nodes
        unselectable_nodes = {self.config.vocab2node[START], self.config.vocab2node[STOP],
                              self.config.vocab2node[EMPTY]}

        while True:  # while a valid node exists in the program
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

            # Check validity of selected node
            if (node.parent.api_num not in unselectable_parent_dnodes) and (node.api_num not in unselectable_nodes):
                return node, rand_node_pos

            # If node is invalid, check if valid node exists in program or given list
            if selectable_node_exists_in_program is None:
                nodes, _ = self.tree_mod.get_vector_representation(self.curr_prog)
                if given_list is not None:
                    nodes = [nodes[i] for i in given_list]
                for i in range(len(nodes)):
                    if i == 0:
                        nodes[i] = 0
                    else:
                        if not (nodes[i] not in unselectable_nodes and nodes[i - 1] not in unselectable_parent_dnodes):
                            nodes[i] = 0
                if sum(nodes) == 0:
                    return None, None
                else:
                    selectable_node_exists_in_program = True

    def _get_new_node(self, parent, edge, verbose=False, grow_new_subtree=False):
        # save original tree length
        orig_length = self.curr_prog.length

        # add empty node
        next_nodes = parent.remove_node(edge)
        empty_node = self.tree_mod.create_and_add_node(TEMP, parent, edge)
        empty_node.add_node(next_nodes, edge)
        empty_node_pos = self.tree_mod.get_nodes_position(self.curr_prog, empty_node)

        if verbose:
            print("\nparent subtree:")
            print_verbose_tree_info(parent)

        assert empty_node_pos < self.curr_prog.length, "Parent position must be in the tree but curr_prog.length = " + str(
            self.curr_prog.length) + " and parent_pos = " + str(empty_node_pos)

        if verbose:
            print("empty node api name:", empty_node.api_name, "empty node pos:", empty_node_pos)

        node, node_pos, prob = self._replace_node_api(empty_node, empty_node_pos, edge, verbose=self.debug, grow_new_subtree=grow_new_subtree)

        # calculate probability of move
        prob -= math.log(orig_length)

        return node, node_pos, prob

    def _replace_node_api(self, node, node_pos, parent_edge, verbose=False, grow_new_subtree=False):

        node.change_api(TEMP, self.config.vocab2node[TEMP])

        # return self.get_ast_idx_top_k(parent_pos, non_dnode) # multinomial on top k
        new_node_idx, prob = self._get_ast_idx(node_pos, parent_edge, verbose=verbose, grow_new_subtree=grow_new_subtree)  # randomly choose from top k

        # replace api name
        node.change_api(self.config.node2vocab[new_node_idx], new_node_idx)

        return node, node_pos, prob

    def _get_ast_idx(self, empty_node_pos, added_edge, verbose=False, grow_new_subtree=False):  # TODO: TEST
        """
        Returns api number (based on vocabulary). Uniform randomly selected from top k based on parent node.
        :param parent_pos: (int) position of parent node in current program (by DFS)
        :return: (int) number of api in vocabulary
        """

        import tensorflow as tf


        logits = self._get_logits_for_add_node(self.curr_prog, self.initial_state, empty_node_pos, added_edge, grow_new_subtree=grow_new_subtree)
        sorted_logits = np.argsort(-logits)

        print(logits.shape)

        if self.use_multinomial:
            # logits are already normalized from decoder
            logits = logits.reshape(1, logits.shape[0])
            idx = self.sess.run(tf.multinomial(logits, 1), {})
            ln_prob = self.calculate_multinomial_ln_prob(logits, idx)
            print("idx", idx)
            return idx[0][0], ln_prob
            # return idx, logits[0][idx]/self.curr_prog.length
        else:  # randomly select from top_k
            mu = random.uniform(0, 1)
            if mu <= self.top_k_prob:
                rand_idx = random.randint(0, self.decoder.top_k - 1)  # randint is a,b inclusive
                if verbose:
                    print("topk", [self.config.node2vocab[sorted_logits[i]] for i in range(0, self.decoder.top_k)])
                    print("topk", self.config.node2vocab[sorted_logits[rand_idx]])
                prob = self.top_k_prob * 1.0 / self.decoder.top_k
                return sorted_logits[rand_idx], math.log(prob)
            else:
                rand_idx = random.randint(self.decoder.top_k, len(sorted_logits) - 1)
                if verbose:
                    print("-topk", self.config.node2vocab[sorted_logits[rand_idx]])
                prob = (1 - self.top_k_prob) * 1.0 / (len(logits) - self.decoder.top_k)
                return sorted_logits[rand_idx], math.log(prob)

    # def _get_logits_for_add_node(self, curr_prog, initial_state, empty_node_pos, added_edge, grow_new_subtree=False):
    #     assert empty_node_pos > 0, "Can't replace DSubTree, empty_node_pos must be > 0"
    #
    #     state = initial_state
    #     nodes, edges, targets = self.tree_mod.get_nodes_edges_targets(curr_prog, verbose=self.verbose)
    #     preceding_pos = targets.index(self.config.vocab2node[TEMP])
    #     # print(empty_node_pos)
    #
    #     # # If targets is the last node, modify nodes,edges,targets to include STOP node
    #     # if empty_node_pos == len(targets) - 1:
    #     #     nodes.append(self.config.vocab2node[TEMP])
    #     #     edges.append(SIBLING_EDGE)
    #     #     targets.append(self.config.vocab2node[STOP])
    #
    #     # assert len(empty_node_pos) == 1
    #     # empty_node_pos = empty_node_pos[0]
    #     # print(empty_node_pos)
    #
    #     node = np.zeros([1, 1], dtype=np.int32)
    #     edge = np.zeros([1, 1], dtype=np.bool)
    #
    #     vocab_size = self.config.vocab_size
    #
    #     # stores states and probabilities for each possible added node
    #     # node_num: ast_state
    #     # extra key in logits is probs_key. probs_key : [logit for each node]
    #     logits = {}
    #     probs_key = "probs"
    #
    #     # preceding_pos = max(0, empty_node_pos - 1)
    #
    #     preceding_prob = 0.0
    #
    #     for i in range(len(nodes)):
    #         node[0][0] = nodes[i]
    #         edge[0][0] = edges[i]
    #
    #         # save all logits
    #         if i == preceding_pos:
    #             state, probs = self.decoder.get_ast_logits(node, edge, state)
    #
    #             assert (vocab_size == len(probs[0]), str(vocab_size) + ", " + str(len(probs[0])))
    #
    #             logits[probs_key] = np.zeros(vocab_size)
    #             for j in range(len(probs[0])):
    #                 logits[j] = state
    #                 logits[probs_key][j] += (probs[0][j] + preceding_prob)
    #
    #             if grow_new_subtree or i == len(nodes) - 1:
    #                 return logits[probs_key]
    #
    #         elif preceding_pos < i <= len(nodes) - 1:
    #             for k in range(vocab_size):
    #                 print(k)
    #                 if self.config.node2vocab[nodes[i]] == TEMP:
    #                     node[0][0] = k
    #                 logits[k], probs = self.decoder.get_ast_logits(node, edge, logits[k])
    #                 logits[probs_key][k] += probs[0][targets[i]]
    #
    #             if i == len(nodes) - 1:
    #                 return logits[probs_key]
    #
    #         # pass in nodes up till the node before added node
    #         else:
    #             state, probs = self.decoder.get_ast_logits(node, edge, state)
    #             preceding_prob += probs[0][targets[i]]

    def _get_logits_for_add_node(self, curr_prog, initial_state, empty_node_pos, added_edge, grow_new_subtree=False):
        assert empty_node_pos > 0, "Can't replace DSubTree, empty_node_pos must be > 0"

        state = initial_state
        nodes, edges, targets = self.tree_mod.get_nodes_edges_targets(curr_prog, verbose=self.verbose)
        preceding_pos = targets.index(self.config.vocab2node[TEMP])

        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)

        vocab_size = self.config.vocab_size

        # stores states and probabilities for each possible added node
        # node_num: ast_state
        # extra key in logits is probs_key. probs_key : [logit for each node]
        # logits = {}

        logits = np.zeros([1, 1, self.config.decoder.units], dtype=float)
        self.probs = np.zeros([vocab_size], dtype=float)

        # preceding_pos = max(0, empty_node_pos - 1)

        preceding_prob = 0.0

        for i in range(len(nodes)):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]

            # save all logits
            if i == preceding_pos:
                state, probs = self.decoder.get_ast_logits(node, edge, state)
                # print(len(state[0][0]))
                # print(len(probs[0]))

                # assert (vocab_size == len(probs[0]), str(vocab_size) + ", " + str(len(probs[0])))
                logits = np.array([state[0][0]] * vocab_size)
                # print(logits.shape)
                self.probs = probs[0]
                self.probs += preceding_prob
                # for j in range(len(probs[0])):
                #     logits[j] = state
                #     logits[probs_key][j] += (probs[0][j] + preceding_prob)

                if grow_new_subtree or i == len(nodes) - 1:
                    return self.probs

            elif preceding_pos < i <= len(nodes) - 1:
                if self.config.node2vocab[nodes[i]] == TEMP:
                    nodes_column = np.array(range(vocab_size), dtype=np.int32).reshape(vocab_size, 1)
                else:
                    nodes_column = np.ones(vocab_size, dtype=np.int32).reshape(vocab_size, 1) * nodes[i]

                edges_column = np.ones(vocab_size, dtype=np.bool).reshape(vocab_size, 1) * edges[i]

                targets_column = np.ones(vocab_size, dtype=np.int32).reshape(vocab_size, 1) * targets[i]

                probs_column = np.zeros(vocab_size, dtype=np.int32).reshape(vocab_size, 1)

                # print(targets_column)
                #
                # print("target:", targets[i])

                idxs_column = np.array(range(vocab_size), dtype=np.int32).reshape(vocab_size, 1)

                all_data = np.append(idxs_column, nodes_column, axis=1)
                # print(all_data.shape)
                all_data = np.append(all_data, edges_column, axis=1)
                # print(all_data.shape)
                all_data = np.append(all_data, targets_column, axis=1)
                all_data = np.append(all_data, probs_column, axis=1)
                all_data = np.append(all_data, logits, axis=1)

                print(all_data.shape)

                start_time = time.time()

                num_cores = multiprocessing.cpu_count()

                # print(num_cores)
                #
                # Parallel(n_jobs=num_cores)(delayed(self.get_ast_logits_wrapper)(i) for i in all_data)

                all_data = parallel_apply_along_axis(get_ast_logits_wrapper, 1, all_data, save_dir=self.config.save_dir)

                # self.get_ast_logits_wrapper.parallel_diagnostics(level=4)

                end_time = time.time()

                print("time taken:", end_time - start_time)

                # print("after:", all_data.shape)

                logits_idx = 4
                logits = all_data[:, logits_idx:]

                # print("logits:", logits.shape)

                if i == len(nodes) - 1:
                    return self.probs

            # pass in nodes up till the node before added node
            else:
                state, probs = self.decoder.get_ast_logits(node, edge, state)
                preceding_prob += probs[0][targets[i]]

    # def _get_logits_for_add_node(self, curr_prog, initial_state, empty_node_pos, added_edge, grow_new_subtree=False):
    #     assert empty_node_pos > 0, "Can't replace DSubTree, empty_node_pos must be > 0"
    #
    #     state = initial_state
    #     nodes, edges, targets = self.tree_mod.get_nodes_edges_targets(curr_prog, verbose=self.verbose)
    #     preceding_pos = targets.index(self.config.vocab2node[TEMP])
    #     # print(empty_node_pos)
    #
    #     # # If targets is the last node, modify nodes,edges,targets to include STOP node
    #     # if empty_node_pos == len(targets) - 1:
    #     #     nodes.append(self.config.vocab2node[TEMP])
    #     #     edges.append(SIBLING_EDGE)
    #     #     targets.append(self.config.vocab2node[STOP])
    #
    #     # assert len(empty_node_pos) == 1
    #     # empty_node_pos = empty_node_pos[0]
    #     # print(empty_node_pos)
    #
    #     node = np.zeros([1, 1], dtype=np.int32)
    #     edge = np.zeros([1, 1], dtype=np.bool)
    #
    #     vocab_size = self.config.vocab_size
    #
    #     # stores states and probabilities for each possible added node
    #     # node_num: ast_state
    #     # extra key in logits is probs_key. probs_key : [logit for each node]
    #     logits = {}
    #     probs_key = "probs"
    #
    #     # preceding_pos = max(0, empty_node_pos - 1)
    #
    #     preceding_prob = 0.0
    #
    #     for i in range(len(nodes)):
    #         node[0][0] = nodes[i]
    #         edge[0][0] = edges[i]
    #
    #         # save all logits
    #         if i == preceding_pos:
    #             state, probs = self.decoder.get_ast_logits(node, edge, state)
    #
    #             assert (vocab_size == len(probs[0]), str(vocab_size) + ", " + str(len(probs[0])))
    #
    #             logits[probs_key] = np.zeros(vocab_size)
    #             for j in range(len(probs[0])):
    #                 logits[j] = state
    #                 logits[probs_key][j] += (probs[0][j] + preceding_prob)
    #
    #             if grow_new_subtree or i == len(nodes) - 1:
    #                 return logits[probs_key]
    #
    #         elif preceding_pos < i <= len(nodes) - 1:
    #             for k in range(vocab_size):
    #                 print(k)
    #                 if self.config.node2vocab[nodes[i]] == TEMP:
    #                     node[0][0] = k
    #                 logits[k], probs = self.decoder.get_ast_logits(node, edge, logits[k])
    #                 logits[probs_key][k] += probs[0][targets[i]]
    #
    #             if i == len(nodes) - 1:
    #                 return logits[probs_key]
    #
    #         # pass in nodes up till the node before added node
    #         else:
    #             state, probs = self.decoder.get_ast_logits(node, edge, state)
    #             preceding_prob += probs[0][targets[i]]


    def calculate_multinomial_ln_prob(self, logits, api_num):
        import tensorflow as tf
        norm_logs = self.sess.run(tf.nn.log_softmax(logits[0]), {})
        # print("norm logs:", sum(norm_logs))
        # print("api name:", self.config.node2vocab[api_num])
        # print(sorted(norm_logs, reverse=True))
        # print(norm_logs[api_num] / curr_prog.length)
        # print(norm_logs[api_num])
        # print(math.log(norm_logs[api_num] / curr_prog.length))
        # print(math.log(norm_logs[api_num]))

        # print("logits:", logits[0][added_node.api_num]/curr_prog.length)
        # return math.log(norm_logs[api_num] / curr_prog.length)
        # return math.log(norm_logs[api_num])
        # return logits[0][added_node.api_num]/curr_prog.length

        return norm_logs[api_num]

        # return norm_logs[api_num]

    def _get_prob_from_logits(self, curr_prog, initial_state, added_node_pos, added_node, added_edge):
        added_node_api = added_node.api_name
        added_node.change_api(TEMP, self.config.vocab2node[TEMP])
        logits = self._get_logits_for_add_node(curr_prog, initial_state, added_node_pos, added_edge)
        added_node.change_api(added_node_api, self.config.vocab2node[added_node_api])

        if self.use_multinomial:
            logits = logits.reshape(1, logits.shape[0])
            return self.calculate_multinomial_ln_prob(logits, added_node.api_num)
        else:
            sorted_logits = np.argsort(-logits)
            if np.where(sorted_logits == added_node.api_num)[0] < self.decoder.top_k:
                return math.log(self.top_k_prob * 1.0 / self.decoder.top_k)
            else:
                assert self.top_k_prob < 1.0, "If top_k_prob = 1.0, then it shouldn't be here"
                return math.log((1 - self.top_k_prob) * 1.0 / (len(logits) - self.decoder.top_k))

    def _get_logits(self, curr_prog, initial_state, added_node_pos, added_node, added_edge):
        return self._get_logits_for_add_node(curr_prog, initial_state, added_node_pos, added_edge)

    def calculate_ln_prob_of_move(self, curr_prog_original, initial_state, added_node_pos, added_edge, prev_length,
                                  is_copy=False):
        if not is_copy:
            curr_prog = curr_prog_original.copy()
        else:
            curr_prog = curr_prog_original

        added_node = self.tree_mod.get_node_in_position(curr_prog, added_node_pos)

        if added_node.api_name not in {DBRANCH, DLOOP, DEXCEPT}:
            ln_prob = self._get_prob_from_logits(curr_prog, initial_state, added_node_pos, added_node, added_edge)
            return ln_prob - math.log(prev_length)
        else:
            if added_node.api_name == DBRANCH:
                cond_node = added_node.child
                assert cond_node is not None
                then_node = cond_node.child
                assert then_node is not None
                else_node = cond_node.sibling
                assert else_node is not None

                # remove child and siblings from DBranch and then add DBranch node
                added_node.remove_node(CHILD_EDGE)
                # added_node.remove_node(SIBLING_EDGE)

                # get probability of adding dbranch node
                dbranch_pos = self.tree_mod.get_nodes_position(curr_prog, added_node)
                ln_prob = self._get_prob_from_logits(curr_prog, initial_state, dbranch_pos, added_node, added_edge)
                ln_prob -= math.log(prev_length)
                # get probability of adding condition node
                cond_node.remove_node(CHILD_EDGE)
                cond_node.remove_node(SIBLING_EDGE)
                cond_node = added_node.add_node(cond_node, CHILD_EDGE)
                cond_pos = self.tree_mod.get_nodes_position(curr_prog, cond_node)
                ln_prob += self._get_prob_from_logits(curr_prog, initial_state, cond_pos, cond_node, CHILD_EDGE)

                # get probability of adding then node
                stop_node = then_node.remove_node(SIBLING_EDGE)
                assert (then_node.child is None)
                cond_node.add_node(then_node, CHILD_EDGE)
                then_pos = self.tree_mod.get_nodes_position(curr_prog, then_node)
                ln_prob += self._get_prob_from_logits(curr_prog, initial_state, then_pos, then_node, CHILD_EDGE)
                then_node.add_node(stop_node, SIBLING_EDGE)

                # get probability of adding else node
                stop_node = else_node.remove_node(SIBLING_EDGE)
                assert (else_node.child is None)
                cond_node.add_node(else_node, SIBLING_EDGE)
                else_pos = self.tree_mod.get_nodes_position(curr_prog, else_node)
                ln_prob += self._get_prob_from_logits(curr_prog, initial_state, else_pos, else_node, SIBLING_EDGE)
                else_node.add_node(stop_node, SIBLING_EDGE)

                return ln_prob
            else:
                cond_node = added_node.child
                assert cond_node is not None
                body_node = cond_node.child
                assert body_node is not None

                # remove child from DNode and then find probability of adding dnode
                added_node.remove_node(CHILD_EDGE)
                # assert (added_node.sibling is None)
                dnode_pos = self.tree_mod.get_nodes_position(curr_prog, added_node)
                ln_prob = self._get_prob_from_logits(curr_prog, initial_state, dnode_pos, added_node, added_edge)

                # get probability of adding condition node
                cond_node.remove_node(CHILD_EDGE)
                assert (cond_node.sibling is None)
                added_node.add_node(cond_node, CHILD_EDGE)
                cond_pos = self.tree_mod.get_nodes_position(curr_prog, cond_node)
                ln_prob += self._get_prob_from_logits(curr_prog, initial_state, cond_pos, cond_node, CHILD_EDGE)

                # get probability of adding body node
                cond_node.add_node(body_node, CHILD_EDGE)
                stop_node = body_node.remove_node(SIBLING_EDGE)
                assert (body_node.child is None)
                body_pos = self.tree_mod.get_nodes_position(curr_prog, body_node)
                ln_prob += self._get_prob_from_logits(curr_prog, initial_state, body_pos, body_node, CHILD_EDGE)
                body_node.add_node(stop_node, SIBLING_EDGE)

                return ln_prob


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)
    print("cpus:", multiprocessing.cpu_count())
    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.pool.ThreadPool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def get_ast_logits_wrapper(all_data_row, save_dir):
    # print("HEREEEEE")
    beam_width = 1
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(""))
    parser.add_argument('--continue_from', type=str, default=save_dir,
                        help='ignore config options and continue training model checkpointed here')
    clargs = parser.parse_args()
    decoder = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)

    # print("all data row shape", all_data_row.shape)
    # print(all_data_row)
    api_num_idx = 0
    nodes_idx = 1
    edges_idx = 2
    targets_idx = 3
    probs_idx = 4
    logits_idx = 5
    api_num = int(all_data_row[api_num_idx])
    # print("api num:", api_num)
    node = np.ones([1, 1], dtype=np.int32) * int(all_data_row[nodes_idx])
    edge = np.ones([1, 1], dtype=np.bool) * int(all_data_row[edges_idx])
    logits = all_data_row[logits_idx:]
    logits = np.expand_dims(logits, axis=0)
    logits = np.expand_dims(logits, axis=0)
    logits, probs = decoder.get_ast_logits(node, edge, logits)
    target = int(all_data_row[targets_idx])
    # print(self.probs.shape)
    # print(self.probs[api_num])
    # print(probs.shape)
    # print(probs[0][target])
    # print(probs[0][target])
    all_data_row[probs_idx] += probs[0][target]

    all_data_row[logits_idx:] = logits[0][0]

    decoder.close()

    return all_data_row


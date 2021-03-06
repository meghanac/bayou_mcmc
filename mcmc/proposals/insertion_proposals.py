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

# tf.config.optimizer.set_jit(True)

from node import Node, SIBLING_EDGE, CHILD_EDGE, DNODES, DBRANCH, DLOOP, DEXCEPT, START, STOP, EMPTY
from utils import print_verbose_tree_info
from configuration import TEMP


class ProposalWithInsertion:
    def __init__(self, tree_modifier, decoder, tf_session, top_k_prob=0.995, verbose=False, debug=False):
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

    def _grow_dbranch(self, dbranch, max_consecutive_inserts=3, cond_node=None):
        """
        Create full DBranch (DBranch, condition, then, else) from parent node.
        :param parent: (Node) parent of DBranch
        :return: (Node) DBranch node
        """
        ln_prob = 0

        # Ensure adding a DBranch won't exceed max depth
        if cond_node is None:
            if self.curr_prog.non_dnode_length + 3 > self.max_num_api or self.curr_prog.length + 5 > self.max_length:
                return None, None

        # Create condition as DBranch child
        if cond_node is None:
            condition, cond_pos, prob = self._get_new_node(dbranch, CHILD_EDGE, verbose=self.debug)
            assert cond_pos > 0, "Error: Condition node position couldn't be found"
            ln_prob += prob
        else:
            condition = cond_node
            cond_pos = self.tree_mod.get_nodes_position(self.curr_prog, cond_node)
            assert cond_pos > 0, "Error: Condition node position couldn't be found"

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
            while parent_node.api_name != STOP and counter < max_consecutive_inserts:
                # Add then api as child to condition node
                parent_node, _, prob = self._get_new_node(parent_node, edge, verbose=self.debug)
                ln_prob += prob
                counter +=1

            if parent_node.api_name != STOP:
                self.tree_mod.create_and_add_node(STOP, parent_node, SIBLING_EDGE)

        added_stop_node = False
        if dbranch.sibling is None:
            self.tree_mod.create_and_add_node(STOP, dbranch, SIBLING_EDGE)
            added_stop_node = True

        return ln_prob, added_stop_node

    def _grow_dloop_or_dexcept(self, dnode, max_consecutive_inserts=2, cond_node=None):
        """
        Create full DLoop (DLoop, condition, body) from parent node
        :param parent: (Node) parent of DLoop
        :return: (Node) DLoop node
        """
        ln_prob = 0

        # Ensure adding a DBranch won't exceed max depth
        if cond_node is None:
            if self.curr_prog.non_dnode_length + 2 > self.max_num_api or self.curr_prog.length + 3 > self.max_length:
                return None, None

        if cond_node is None:
            parent_node = dnode
        else:
            parent_node = cond_node
        counter = 0
        while parent_node.api_name != STOP and counter < max_consecutive_inserts:
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

        logits = self._get_logits_for_add_node(self.curr_prog, self.initial_state, empty_node_pos, added_edge, grow_new_subtree=grow_new_subtree)
        sorted_logits = np.argsort(-logits)

        if self.use_multinomial:
            # logits are already normalized from decoder
            logits = logits.reshape(1, logits.shape[0])
            idx = self.sess.run(tf.multinomial(logits, 1)[0][0], {})
            ln_prob = self.calculate_multinomial_ln_prob(logits, idx)
            return idx, ln_prob
            # return idx, logits[0][idx]/self.curr_prog.length
        else:  # randomly select from top_k
            mu = random.uniform(0, 1)
            if mu <= self.top_k_prob:
                rand_idx = random.randint(0, self.decoder.top_k - 1)  # randint is a,b inclusive
                if verbose:
                    print("topk", [self.config.node2vocab[sorted_logits[i]] for i in range(0, self.decoder.top_k)])
                    print("topk", self.config.node2vocab[sorted_logits[rand_idx]])
                for i in range(len(sorted_logits)):
                    if self.config.node2vocab[sorted_logits[i]] not in {STOP, START, DBRANCH, DLOOP, DEXCEPT}:
                        rand_idx = i
                        break
                prob = self.top_k_prob * 1.0 / self.decoder.top_k
                return sorted_logits[rand_idx], math.log(prob)
            else:
                rand_idx = random.randint(self.decoder.top_k, len(sorted_logits) - 1)
                if verbose:
                    print("-topk", self.config.node2vocab[sorted_logits[rand_idx]])
                prob = (1 - self.top_k_prob) * 1.0 / (len(logits) - self.decoder.top_k)
                return sorted_logits[rand_idx], math.log(prob)

    # def _get_logits_for_add_node(self, curr_prog, initial_state, empty_node_pos, added_edge):
    #     #     assert empty_node_pos > 0, "Can't replace DSubTree, empty_node_pos must be > 0"
    #     #
    #     #     state = initial_state
    #     #     nodes, edges = self.tree_mod.get_vector_representation(curr_prog)
    #     #
    #     #     node = np.zeros([1, 1], dtype=np.int32)
    #     #     edge = np.zeros([1, 1], dtype=np.bool)
    #     #
    #     #     vocab_size = self.config.vocab.api_dict_size
    #     #
    #     #     # stores states and probabilities for each possible added node
    #     #     # node_num: ast_state
    #     #     # extra key in logits is probs_key. probs_key : [logit for each node]
    #     #     logits = {}
    #     #     probs_key = "probs"
    #     #
    #     #     preceding_pos = empty_node_pos - 1
    #     #
    #     #     preceding_prob = 0.0
    #     #
    #     #     for i in range(curr_prog.length):
    #     #         node[0][0] = nodes[i]
    #     #         edge[0][0] = edges[i]
    #     #
    #     #         # save all logits
    #     #         if i == preceding_pos:
    #     #             state, probs = self.decoder.get_ast_logits(node, edge, state)
    #     #
    #     #             assert (vocab_size == len(probs[0]), str(vocab_size) + ", " + str(len(probs[0])))
    #     #
    #     #             logits[probs_key] = np.zeros(vocab_size)
    #     #             for j in range(len(probs[0])):
    #     #                 logits[j] = state
    #     #                 logits[probs_key][j] += (probs[0][j] + preceding_prob)
    #     #
    #     #         # pass in each node that could be added into decoder
    #     #         elif i == empty_node_pos:
    #     #             for k in range(vocab_size):
    #     #                 node[0][0] = k
    #     #                 edge[0][0] = added_edge
    #     #
    #     #                 ast_state, probs = self.decoder.get_ast_logits(node, edge, state)
    #     #                 logits[k] = ast_state
    #     #                 if empty_node_pos == curr_prog.length - 1:
    #     #                     logits[probs_key][k] += probs[0][self.config.vocab2node[STOP]]
    #     #                 else:
    #     #                     logits[probs_key][k] += probs[0][nodes[i + 1]]
    #     #
    #     #             if empty_node_pos == curr_prog.length - 1:
    #     #                 return logits[probs_key]
    #     #
    #     #         elif empty_node_pos < i < curr_prog.length - 1:
    #     #             for k in range(vocab_size):
    #     #                 ast_state, probs = self.decoder.get_ast_logits(node, edge, logits[k])
    #     #                 logits[k] = ast_state
    #     #                 logits[probs_key][k] += probs[0][nodes[i + 1]]
    #     #
    #     #         elif i == curr_prog.length - 1:
    #     #             if self.config.node2vocab[nodes[i]] != STOP:
    #     #                 for k in range(vocab_size):
    #     #                     ast_state, probs = self.decoder.get_ast_logits(node, edge, logits[k])
    #     #                     logits[k] = ast_state
    #     #                     logits[probs_key][k] += probs[0][self.config.vocab2node[STOP]]
    #     #
    #     #             return logits[probs_key]
    #     #
    #     #         # pass in nodes up till the node before added node
    #     #         else:
    #     #             state, probs = self.decoder.get_ast_logits(node, edge, state)
    #     #             preceding_prob += probs[0][nodes[i + 1]]

    def _get_logits_for_add_node(self, curr_prog, initial_state, empty_node_pos, added_edge, grow_new_subtree=False):
        assert empty_node_pos > 0, "Can't replace DSubTree, empty_node_pos must be > 0"

        state = initial_state
        nodes, edges, targets = self.tree_mod.get_nodes_edges_targets(curr_prog, verbose=self.verbose)
        preceding_pos = targets.index(self.config.vocab2node[TEMP])
        # print(empty_node_pos)

        # # If targets is the last node, modify nodes,edges,targets to include STOP node
        # if empty_node_pos == len(targets) - 1:
        #     nodes.append(self.config.vocab2node[TEMP])
        #     edges.append(SIBLING_EDGE)
        #     targets.append(self.config.vocab2node[STOP])

        # assert len(empty_node_pos) == 1
        # empty_node_pos = empty_node_pos[0]
        # print(empty_node_pos)

        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)

        vocab_size = self.config.vocab_size

        # stores states and probabilities for each possible added node
        # node_num: ast_state
        # extra key in logits is probs_key. probs_key : [logit for each node]
        logits = {}
        probs_key = "probs"

        # preceding_pos = max(0, empty_node_pos - 1)

        preceding_prob = 0.0

        for i in range(len(nodes)):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]

            # save all logits
            if i == preceding_pos:
                state, probs = self.decoder.get_ast_logits(node, edge, state)

                assert (vocab_size == len(probs[0]), str(vocab_size) + ", " + str(len(probs[0])))

                logits[probs_key] = np.zeros(vocab_size)
                for j in range(len(probs[0])):
                    logits[j] = state
                    logits[probs_key][j] += (probs[0][j] + preceding_prob)

                if grow_new_subtree or i == len(nodes) - 1:
                    return logits[probs_key]

            elif preceding_pos < i <= len(nodes) - 1:
                for k in range(vocab_size):
                    if self.config.node2vocab[nodes[i]] == TEMP:
                        node[0][0] = k
                    logits[k], probs = self.decoder.get_ast_logits(node, edge, logits[k])
                    logits[probs_key][k] += probs[0][targets[i]]

                if i == len(nodes) - 1:
                    return logits[probs_key]

            # pass in nodes up till the node before added node
            else:
                state, probs = self.decoder.get_ast_logits(node, edge, state)
                preceding_prob += probs[0][targets[i]]

    def calculate_multinomial_ln_prob(self, logits, api_num):
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

        # print("calculate ln prob: ")
        # print_verbose_tree_info(curr_prog_original)

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
                print_verbose_tree_info(added_node)
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
                assert (cond_node.sibling is None), cond_node.api_name + " sibling: " + cond_node.sibling.api_name
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
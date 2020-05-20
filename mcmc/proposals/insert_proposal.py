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


class InsertProposal:

    def __init__(self, tree_modifier, decoder):
        self.decoder = decoder
        self.config = tree_modifier.config
        self.tree_mod = tree_modifier
        self.max_num_api = self.config.max_num_api
        self.max_length = self.config.max_length

        # Temporary attributes
        self.curr_prog = None
        self.initial_state = None

        # Logging
        self.attempted = 0
        self.accepted = 0

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
        new_node_idx, prob = self.__get_ast_idx(rand_node_pos, SIBLING_EDGE, non_dnode=False)
        new_node_api = self.config.node2vocab[new_node_idx]

        # If a dnode is chosen, grow it out
        if new_node_api == DBRANCH:
            new_node, ln_prob = self.__grow_dbranch(new_node_parent)
            prob += ln_prob
        elif new_node_api == DLOOP:
            new_node, ln_prob = self.__grow_dloop_or_dexcept(new_node_parent, True)
            prob += ln_prob
        elif new_node_api == DEXCEPT:
            new_node, ln_prob = self.__grow_dloop_or_dexcept(new_node_parent, False)
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
            return self.undo_add_random_dnode(added_node)

        if added_node.sibling is None:
            added_node.parent.remove_node(SIBLING_EDGE)
        else:
            sibling_node = added_node.sibling
            parent_node = added_node.parent
            added_node.parent.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling_node, SIBLING_EDGE)

    def add_random_dnode(self, curr_prog, initial_state):
        """
        Adds a DBranch, DLoop or DExcept to a random node in the current program.
        :return: (Node) the dnode node
        """
        # Temporarily save curr_prog and initial_state
        self.curr_prog = curr_prog
        self.initial_state = initial_state

        dnode_type = random.choice([DBRANCH, DLOOP, DEXCEPT])

        parent, _ = self.__get_valid_random_node(curr_prog)

        if parent is None:
            return None

        assert parent.child is None or parent.parent.api_name == DBRANCH, \
            "WARNING: there's a bug in get_valid_random_node because parent node has child"

        # Grow dnode type
        if dnode_type == DBRANCH:
            dnode, ln_prob = self.__grow_dbranch(parent)
        elif dnode_type == DLOOP:
            dnode, ln_prob = self.__grow_dloop_or_dexcept(parent, True)
        else:
            dnode, ln_prob = self.__grow_dloop_or_dexcept(parent, False)

        # Reset self.curr_prog and self.initial_state
        self.curr_prog = None
        self.initial_state = None

        return curr_prog, dnode, ln_prob

    def undo_add_random_dnode(self, dnode):
        """
        Removes the dnode that was added in add_random_dnode().
        :param dnode: (Node) dnode that is to be removed
        :return:
        """
        dnode_sibling = dnode.sibling
        dnode_parent = dnode.parent
        dnode_parent.remove_node(SIBLING_EDGE)
        dnode_parent.add_node(dnode_sibling, SIBLING_EDGE)

    def __grow_dbranch(self, parent):
        """
        Create full DBranch (DBranch, condition, then, else) from parent node.
        :param parent: (Node) parent of DBranch
        :return: (Node) DBranch node
        """
        ln_prob = 0

        # remove parent's current sibling node if there
        parent_sibling = parent.sibling
        parent.remove_node(SIBLING_EDGE)

        # Create a DBranch node
        dbranch = self.tree_mod.create_and_add_node(DBRANCH, parent, SIBLING_EDGE)
        dbranch_pos = self.tree_mod.get_nodes_position(self.curr_prog, dbranch)
        assert dbranch_pos > 0, "Error: DBranch position couldn't be found"

        # Add parent's old sibling node to DBranch with sibling edge
        dbranch.add_node(parent_sibling, SIBLING_EDGE)

        # Ensure adding a DBranch won't exceed max depth
        if self.curr_prog.non_dnode_length + 3 > self.max_num_api or self.curr_prog.length + 6 > self.max_length:
            # remove added dbranch
            parent.remove_node(SIBLING_EDGE)
            parent.add_node(parent_sibling, SIBLING_EDGE)
            return None

        # Create condition as DBranch child
        cond_idx, prob = self.__get_ast_idx(dbranch_pos, CHILD_EDGE)
        condition = self.tree_mod.create_and_add_node(self.config.node2vocab[cond_idx], dbranch, CHILD_EDGE)
        cond_pos = self.tree_mod.get_nodes_position(self.curr_prog, condition)
        assert cond_pos > 0, "Error: Condition node position couldn't be found"
        ln_prob += prob

        # Add then api as child to condition node
        then_idx, prob = self.__get_ast_idx(cond_pos, CHILD_EDGE)
        then_node = self.tree_mod.create_and_add_node(self.config.node2vocab[then_idx], condition, CHILD_EDGE)
        self.tree_mod.create_and_add_node(STOP, then_node, SIBLING_EDGE)
        ln_prob += prob

        # Add else api as sibling to condition node
        else_idx, prob = self.__get_ast_idx(cond_pos, SIBLING_EDGE)
        else_node = self.tree_mod.create_and_add_node(self.config.node2vocab[else_idx], condition, SIBLING_EDGE)
        ln_prob += prob

        self.tree_mod.create_and_add_node(STOP, else_node, SIBLING_EDGE)

        return dbranch, ln_prob

    def __grow_dloop_or_dexcept(self, parent, create_dloop):
        """
        Create full DLoop (DLoop, condition, body) from parent node
        :param create_dloop: (bool) True to create dloop, False to create dexcept
        :param parent: (Node) parent of DLoop
        :return: (Node) DLoop node
        """
        ln_prob = 0
        # remove parent's current sibling node if there
        parent_sibling = parent.sibling
        parent.remove_node(SIBLING_EDGE)

        if create_dloop:
            # Create a DLoop node
            dnode = self.tree_mod.create_and_add_node(DLOOP, parent, SIBLING_EDGE)
        else:
            # Create a DExcept node
            dnode = self.tree_mod.create_and_add_node(DEXCEPT, parent, SIBLING_EDGE)

        dnode_pos = self.tree_mod.get_nodes_position(self.curr_prog, dnode)
        assert dnode_pos > 0, "Error: DNode position couldn't be found"

        # Add parent's old sibling node to D-node with sibling edge
        dnode.add_node(parent_sibling, SIBLING_EDGE)

        # Ensure adding a DLoop won't exceed max depth
        if self.curr_prog.non_dnode_length + 2 > self.max_num_api or self.curr_prog.length + 4 > self.max_length:
            # remove added dnode
            parent.remove_node(SIBLING_EDGE)
            parent.add_node(parent_sibling, SIBLING_EDGE)
            return None

        # Create condition/try as DNode child
        cond_idx, prob = self.__get_ast_idx(dnode_pos, CHILD_EDGE)
        condition = self.tree_mod.create_and_add_node(self.config.node2vocab[cond_idx], dnode, CHILD_EDGE)
        cond_pos = self.tree_mod.get_nodes_position(self.curr_prog, condition)
        assert cond_pos > 0, "Error: Condition node position couldn't be found"
        ln_prob += prob

        # Add body/catch api as child to condition node
        body_idx, prob = self.__get_ast_idx(cond_pos, CHILD_EDGE)
        body = self.tree_mod.create_and_add_node(self.config.node2vocab[body_idx], condition, CHILD_EDGE)
        ln_prob += prob

        self.tree_mod.create_and_add_node(STOP, body, SIBLING_EDGE)

        return dnode, ln_prob

    def __get_valid_random_node(self, given_list=None):
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
                                      self.config.vocab2node[DEXCEPT]}  # TODO: make sure I can actually remove DBranch node

        # Unselectable nodes
        unselectable_nodes = {self.config.vocab2node[START], self.config.vocab2node[STOP], self.config.vocab2node[EMPTY]}

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

    def __get_ast_idx(self, parent_pos, edge, non_dnode=False):
        # return self.get_ast_idx_all_vocab(parent_pos, non_dnode)  # multinomial on all

        # return self.get_ast_idx_top_k(parent_pos, non_dnode) # multinomial on top k

        return self.__get_ast_idx_random_top_k(parent_pos, non_dnode, edge)  # randomly choose from top k

    def __get_ast_idx_random_top_k(self, parent_pos, added_edge, top_k_prob=0.95):  # TODO: TEST
        """
        Returns api number (based on vocabulary). Uniform randomly selected from top k based on parent node.
        :param parent_pos: (int) position of parent node in current program (by DFS)
        :return: (int) number of api in vocabulary
        """

        logits = self.__get_logits_for_add_node(parent_pos, added_edge)
        sorted_logits = np.argsort(-logits)

        # print(sorted_logits)
        # print(logits[sorted_logits[0]])
        # print(logits[sorted_logits[1]])

        mu = random.uniform(0, 1)
        if mu <= top_k_prob:
            rand_idx = random.randint(0, self.decoder.top_k - 1)  # randint is a,b inclusive
            print("topk", [self.config.node2vocab[sorted_logits[i]] for i in range(0, self.decoder.top_k)])
            print("topk", self.config.node2vocab[sorted_logits[rand_idx]])
            prob = top_k_prob * 1.0 / self.decoder.top_k
            return sorted_logits[rand_idx], prob
        else:
            rand_idx = random.randint(self.decoder.top_k, len(sorted_logits) - 1)
            print("-topk", self.config.node2vocab[sorted_logits[rand_idx]])
            prob = (1 - top_k_prob) * 1.0 / self.decoder.top_k
            return sorted_logits[rand_idx], prob

    # def __get_ast_idx_top_k(self, parent_pos, added_edge, top_k_prob=0.95):
    #     logits = self.get_logits_for_add_node(parent_pos, added_edge)
    #     sorted_logits = np.argsort(-logits)
    #
    #     mu = random.uniform(0, 1)
    #     if mu <= top_k_prob:
    #         top_k = sorted_logits[:self.decoder.top_k]
    #         top_k /= np.linalg.norm(top_k)
    #         # u = random.uniform(0, 1)
    #         # cum_topk = np.cumsum(top_k)
    #         # prob = top_k_prob * 1.0 / self.decoder.top_k
    #         selected = tf.multinomial(top_k)
    #         return sorted_logits[selected], top_k[selected]
    #     else:
    #         not_topk = sorted_logits[self.decoder.top_k:]
    #         not_topk /= np.linalg.norm(not_topk)
    #         selected = tf.multinomial(not_topk)
    #         return sorted_logits[selected], not_topk[selected]

    def __get_logits_for_add_node(self, parent_pos, added_edge):
        state = self.initial_state
        nodes, edges = self.tree_mod.get_vector_representation()

        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)

        vocab_size = self.config.vocab.api_dict_size

        logits = {}

        for i in range(self.curr_prog.length):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            if i == parent_pos:
                state, probs = self.decoder.get_ast_logits(node, edge, state)

                assert (vocab_size == len(probs[0]), str(vocab_size) + ", " + str(len(probs[0])))

                logits["probs"] = np.zeros(vocab_size)
                for j in range(len(probs[0])):
                    logits[j] = state
                    logits["probs"][j] += probs[0][j]

                # pass in each node that could be added into decoder
                for k in range(vocab_size):
                    node[0][0] = k
                    edge[0][0] = added_edge

                    ast_state, probs = self.decoder.get_ast_logits(node, edge, state)
                    logits[k] = ast_state
                    if parent_pos == self.curr_prog.length - 1:
                        logits["probs"][k] += probs[0][self.config.vocab2node[STOP]]
                    else:
                        logits["probs"][k] += probs[0][nodes[i + 1]]

                if parent_pos == self.curr_prog.length - 1:
                    return logits["probs"]

            elif parent_pos < i < self.curr_prog.length - 1:
                for k in range(vocab_size):
                    ast_state, probs = self.decoder.get_ast_logits(node, edge, state)
                    logits[k] = ast_state
                    logits["probs"][k] += probs[0][nodes[i + 1]]

            elif i == self.curr_prog.length - 1:
                if self.config.node2vocab[nodes[i]] != STOP:
                    ast_state, probs = self.decoder.get_ast_logits(node, edge, state)
                    logits[k] = ast_state
                    logits["probs"][k] += probs[0][self.config.vocab2node[STOP]]

                return logits["probs"]

            else:
                state, _ = self.decoder.get_ast_logits(node, edge, state)
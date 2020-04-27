import argparse
import math
import os
import sys
import textwrap
from copy import deepcopy

from ast_helper.beam_searcher.program_beam_searcher import ProgramBeamSearcher
from data_reader import Reader
from data_loader import Loader
from infer import BayesianPredictor
from trainer_vae.model import Model
from trainer_vae.utils import get_var_list, read_config
import numpy as np
import random
import json
import tensorflow as tf

CHILD_EDGE = True
SIBLING_EDGE = False

MAX_LOOP_NUM = 3
MAX_BRANCHING_NUM = 3
MAX_AST_DEPTH = 32

STOP = 'DStop'
DBRANCH = 'DBranch'
DLOOP = 'DLoop'
DEXCEPT = 'DExcept'
START = 'DSubTree'
DNODES = {START, STOP, DBRANCH, DLOOP, DEXCEPT}

EMPTY = '__delim__'


class TooLongLoopingException(Exception):
    pass


class TooLongBranchingException(Exception):
    pass


class Node:
    """
    Class for a single node in a program. Each node represents a single API.
    A program is one or more Nodes, each connected to each other by Sibling or Child Edges.
    There may be at most one sibling and one child node.
    DSubtree is used as the start node of the tree and may only have 1 sibling node representing the first API in
    the program.
    Only DBranch, DLoop and DExcept and their child nodes may have child edges/nodes.
    DStop nodes may not have a sibling or a child node.
    """

    def __init__(self, id, api_name, api_num, parent, parent_edge):
        """
        :param id: [DEPRECATED] (int) unique id for Node object
        :param api_name: (string) name of the api
        :param api_num: (int) number assigned to api in vocab dictionary. Must be > 0.
        :param parent: (Node) parent node
        :param parent_edge: (bool - SIBLING_EDGE or CHILD_EDGE) represents the edge between this node and parent node
        """
        self.api_name = api_name
        self.api_num = api_num
        self.child = None  # pointer to child node
        self.sibling = None  # pointer to sibling node
        self.parent = parent  # pointer to parent node
        self.id = id
        self.length = 1  # Number of nodes in subtree starting at this node (i.e., how many nodes stem out of this node)
        self.parent_edge = parent_edge

        if self.api_name in DNODES:
            self.non_dnode_length = 0
        else:
            self.non_dnode_length = 1

    def update_length(self, length, non_dnode_length, add_or_sub):
        """
        Updates length of this node and recursively updates all node above it so that length of head of the program
        (DSubtree) accounts for all nodes in program
        :param non_dnode_length: length of program excluding all dnodes
        :param length: (int) length to be added to or subtracted from this node
        :param add_or_sub: (string - ust be either 'add' or 'subtract') flag whether to add length or subtract length
        :return:
        """
        # Add length
        if add_or_sub == 'add':
            self.length += length
            self.non_dnode_length += non_dnode_length
        # Subtract length
        else:
            self.length -= length
            self.non_dnode_length -= non_dnode_length
        # Recursively update parent nodes' length
        if self.parent is not None:
            self.parent.update_length(length, non_dnode_length, add_or_sub)

    def add_node(self, node, edge):
        """
        :param node: (Node) node to be added to self. Can be None (i.e., nothing will be added)
        :param edge: (bool- SIBLING_EDGE or CHILD_EDGE) whether node will be self's child or sibling node
        :return:
        """
        if node is not None:
            if edge == SIBLING_EDGE:
                # Remove existing sibling node if there is one
                if self.sibling is not None:
                    self.remove_node(SIBLING_EDGE)
                # Update self
                self.update_length(node.length, node.non_dnode_length, 'add')
                self.sibling = node
                # Update node
                node.parent = self
                node.parent_edge = edge

            elif edge == CHILD_EDGE:
                # Remove existing child node is there is o e
                if self.child is not None:
                    self.remove_node(CHILD_EDGE)
                # Update self
                self.update_length(node.length, node.non_dnode_length, 'add')
                self.child = node
                # Update node
                node.parent = self
                node.parent_edge = edge
            else:
                raise ValueError('edge must be SIBLING_EDGE or CHILD_EDGE')

    def remove_node(self, edge):
        """
        Removes node specified by edge. Removes the entire subtree that stems from the node to be removed.
        :param edge: (bool - SIBLING_EDGE or CHILD_EDGE)
        :return:
        """
        if edge == SIBLING_EDGE:
            if self.sibling is not None:
                old_sibling = self.sibling
                self.sibling = None
                old_sibling.parent = None
                old_sibling.parent_edge = None
                self.update_length(old_sibling.length, old_sibling.non_dnode_length, 'sub')

        elif edge == CHILD_EDGE:
            if self.child is not None:
                child = self.child
                self.child = None
                child.parent = None
                child.parent_edge = None
                self.update_length(child.length, child.non_dnode_length, 'sub')

        else:
            raise ValueError('edge must but a sibling or child edge')

    def copy(self):
        """
        Deep copy self including entire subtree. Recursively copies sibling and child nodes.
        :return: (Node) deep copy of self
        """
        new_node = Node(self.id, self.api_name, self.api_num, self.parent, self.parent_edge)
        if self.sibling is not None:
            new_sibling_node = self.sibling.copy()
            new_node.add_node(new_sibling_node, SIBLING_EDGE)
        if self.child is not None:
            new_child_node = self.child.copy()
            new_node.add_node(new_child_node, CHILD_EDGE)
        return new_node


class MCMCProgram:
    """

    """

    def __init__(self, save_dir):
        """
        Initialize program
        :param save_dir: (string) path to directory in which saved model checkpoints are in
        """
        self.save_dir = save_dir

        # Initialize model
        config_file = os.path.join(save_dir, 'config.json')
        with open(config_file) as f:
            self.config = config = read_config(json.load(f), infer=True)

        # Initialize model configurations
        self.max_num_api = self.config.max_ast_depth
        self.max_length = MAX_AST_DEPTH
        self.config.max_ast_depth = 1
        self.config.max_fp_depth = 1
        self.config.batch_size = 1

        # Restore ML model
        self.model = Model(self.config)
        self.sess = tf.Session()
        self.restore(save_dir)
        with tf.name_scope("ast_inference"):
            ast_logits = self.model.decoder.ast_logits[:, 0, :]
            self.ast_ln_probs = tf.nn.log_softmax(ast_logits)

        # Initialize variables about program
        self.constraints = []
        self.constraint_nodes = []  # has node numbers of constraints
        self.vocab2node = self.config.vocab.api_dict
        self.node2vocab = dict(zip(self.vocab2node.values(), self.vocab2node.keys()))
        self.rettype2num = self.config.vocab.ret_dict
        self.num2rettype = dict(zip(self.rettype2num.values(), self.rettype2num.keys()))
        self.fp2num = self.config.vocab.fp_dict
        self.num2fp = dict(zip(self.fp2num.values(), self.fp2num.keys()))
        self.curr_prog = None
        self.curr_log_prob = -0.0
        self.prev_log_prob = -0.0
        self.node_id_counter = 0
        self.latent_state = None
        self.initial_state = None
        self.encoder_mean = None
        self.encoder_covar = None
        self.predictor = None
        self.searcher = None

        self.ret_type = []
        self.fp = [[]]
        self.decoder = None

        # Logging  # TODO: change to Logger
        self.accepted = 0
        self.valid = 0
        self.invalid = 0
        self.add = 0
        self.delete = 0
        self.swap = 0
        self.rejected = 0
        self.add_accepted = 0
        self.delete_accepted = 0
        self.swap_accepted = 0
        # self.add_invalid = 0
        # self.delete_invalid = 0
        # self.swap_invalid = 0
        self.add_dnode = 0
        self.add_dnode_accepted = 0

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
            node_num = self.vocab2node[constraint]
            if len(self.constraints) < self.max_num_api:
                self.constraint_nodes.append(node_num)
                self.constraints.append(constraint)
            else:
                print("Cannot add constraint", constraint, ". Limit reached.")
        except KeyError:
            print("Constraint ", constraint, " is not in vocabulary. Will be skipped.")

    def add_return_type(self, ret_type):
        try:
            ret_num = self.rettype2num[ret_type]
            if len(self.ret_type) < self.max_num_api:  # Might just be 1
                self.ret_type.append(ret_num)
            else:
                print("Cannot add return type", ret_type, ". Limit reached.")
        except KeyError:
            print("Return type ", ret_type, " is not in the vocabulary. Will be skipped.")

    def add_formal_parameters(self, fp):
        try:
            fp_num = self.fp2num[fp]
            if len(self.fp[0]) <= self.max_num_api:
                self.fp[0].append(fp_num)
            else:
                 print("Cannot add formal parameter", fp, ". Limit reached.")
        except KeyError:
            print("Formal parameter ", fp, "is not in the vocabulary. Will be skipped")

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
        if len(self.fp[0]) < self.max_num_api:
            for i in range(self.max_num_api - len(self.fp[0])):
                self.fp[0].append(0)

        # Initialize tree
        head = self.create_and_add_node(START, None, SIBLING_EDGE)
        self.curr_prog = head

        # Add constraint nodes to tree
        last_node = head
        for i in self.constraints:
            node = self.create_and_add_node(i, last_node, SIBLING_EDGE)
            last_node = node

        # Initialize model states
        # self.update_latent_state()
        self.get_initial_decoder_state()
        # self.get_random_initial_state()

        # Update probabilities of tree
        self.calculate_probability()
        self.prev_log_prob = self.curr_log_prob

    def create_and_add_node(self, api_name, parent, edge):
        """
        Create a new node and add it to the program
        :param api_name: (string) api name of node to be created
        :param parent: (Node) parent of node to be created
        :param edge: (bool- SIBLING_EDGE or CHILD_EDGE) relation of node to be created to parent node
        :return: (Node) node that has been created
        """
        try:
            api_num = self.vocab2node[api_name]
            node = Node(self.node_id_counter, api_name, api_num, parent, edge)

            # Update unique node id generator
            self.node_id_counter += 1

            # Update parent according to edge
            if api_name != START:
                parent.add_node(node, edge)

            return node

        except KeyError:  # api does not exist in vocabulary and hence node cannot be made
            print(api_name, " is not in vocabulary. Node was not added.")
            return None

    # def get_random_api(self):
    #     """
    #     :return: (string) random API from vocab that is not D-node
    #     """
    #     while True:
    #         api = self.node2vocab[random.randint(1, self.model.config.vocab.api_dict_size - 1)]  # exclude 0
    #         if api not in {'DBranch', 'DLoop', 'DExcept', 'DStop', 'DSubtree'}:
    #             return api

    def get_node_in_position(self, pos_num):
        """
        Get Node object for node in given position. Position number is determined by DFS, with child edges taking
        precedence over sibling edges.
        :param pos_num: (int) position number of node required. THIS IS ZERO-INDEXED POSITION NUMBER>
        :return: (Node) required node
        """
        assert pos_num < self.curr_prog.length, "Position number must be less than total length of program"

        num_explored = 0
        stack = []
        curr_node = self.curr_prog

        while num_explored != pos_num and curr_node is not None:
            # Fail safe
            if num_explored > self.max_length + 1:
                raise ValueError("WARNING: Caught in infinite loop")

            # Update curr_node
            if curr_node.child is not None:
                if curr_node.sibling is not None:
                    stack.append(curr_node.sibling)
                curr_node = curr_node.child
                num_explored += 1
            elif curr_node.sibling is not None:
                curr_node = curr_node.sibling
                num_explored += 1
            else:
                if len(stack) > 0:
                    curr_node = stack.pop()
                    num_explored += 1
                else:
                    raise ValueError('DFS failed to find node in postion: ' + str(pos_num))

        return curr_node

    def get_nodes_position(self, node):
        """
        Given a node, return it's position number determined by DFS, with child taking precedence over sibling.
        :param node: (Node) node whose position is to be determined
        :return: (int) position number of given node
        """
        # Fail safe
        if node is None:
            print("WARNING: node fed into self.get_nodes_position is None")
            return node

        stack = []
        curr_node = self.curr_prog
        counter = 0

        # DFS
        while curr_node is not None:
            if curr_node == node:
                return counter

            counter += 1

            if counter > self.max_length + 1:
                raise ValueError("WARNING: Caught in infinite loop")
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
                    raise ValueError('DFS failed to find given node: ' + node.api_name)

        return -1

    def get_vector_representation(self):
        """
        Returns vectors of nodes and edges in the current program that can be fed into the model
        :return: (list of ints) nodes, (list of bools) edges
        """
        stack = []
        curr_node = self.curr_prog
        nodes_dfs = []
        edges_dfs = []

        while curr_node is not None:
            # Add current node to list
            nodes_dfs.append(curr_node.api_num)

            # Update edges list
            if curr_node.api_name != START:
                edges_dfs.append(curr_node.parent_edge)

            # Find next node
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
                    curr_node = None

        # Fill in blank nodes and take only self.max_length number of nodes in the tree
        nodes = np.zeros([1, self.max_length], dtype=np.int32)
        edges = np.zeros([1, self.max_length], dtype=np.bool)
        nodes[0, :len(nodes_dfs)] = nodes_dfs
        edges[0, :len(edges_dfs)] = edges_dfs

        return nodes[0], edges[0]

    def get_node_names_and_edges(self):
        """
        Returns list of apis (nodes) and edges of the current program
        :return: (list of strings) apis of nodes, (list of booleans) edges
        """
        nodes, edges = self.get_vector_representation()
        nodes = [self.node2vocab[node] for node in nodes]
        return nodes, edges

    def get_valid_random_node(self, given_list=None):
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
        unselectable_parent_dnodes = {self.vocab2node[DLOOP],
                                      self.vocab2node[DEXCEPT]}  # TODO: make sure I can actually remove DBranch node

        # Unselectable nodes
        unselectable_nodes = {self.vocab2node[START], self.vocab2node[STOP], self.vocab2node[EMPTY]}

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

            node = self.get_node_in_position(rand_node_pos)

            # Check validity of selected node
            if (node.parent.api_num not in unselectable_parent_dnodes) and (node.api_num not in unselectable_nodes):
                return node, rand_node_pos

            # If node is invalid, check if valid node exists in program or given list
            if selectable_node_exists_in_program is None:
                print("-----------------HERE-----------")
                nodes, _ = self.get_vector_representation()
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

    def get_random_node_to_swap(self, given_list=None):
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

        # Unselectable nodes
        unselectable_nodes = {self.vocab2node[START], self.vocab2node[STOP], self.vocab2node[EMPTY],
                              self.vocab2node[DLOOP], self.vocab2node[DEXCEPT], self.vocab2node[DBRANCH]}

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

            node = self.get_node_in_position(rand_node_pos)

            # Check validity of selected node
            if node.api_num not in unselectable_nodes:
                return node, rand_node_pos

            # If node is invalid, check if valid node exists in program or given list
            if selectable_node_exists_in_program is None:
                print("-----------------HERE-----------")
                nodes, _ = self.get_vector_representation()
                if given_list is not None:
                    nodes = [nodes[i] for i in given_list]
                for i in range(len(nodes)):
                    if i == 0:
                        nodes[i] = 0
                    else:
                        if nodes[i] in unselectable_nodes:
                            nodes[i] = 0
                if sum(nodes) == 0:
                    return None, None
                else:
                    selectable_node_exists_in_program = True

    # def get_non_stop_random_pos(self):
    #     """
    #     Returns a node and its position of a random node in the current program that is not a DStop node.
    #     :return: (Node) selected node, (int) selected node's position
    #     """
    #     while True:  # This shouldn't cause problems because at the very least the program must contain constraint apis
    #         # exclude DSubTree node, randint is [a,b] inclusive
    #         rand_node_pos = random.randint(1, self.curr_prog.length - 1)
    #         node = self.get_node_in_position(rand_node_pos)
    #
    #         if node.api_name != 'DStop':
    #             return node, rand_node_pos

    def get_deletable_node(self):
        """
        Returns a random node and its position in the current program that can be deleted without causing any
        fragmentation in the program.
        :return: (Node) selected node, (int) selected node's position
        """
        while True:  # This shouldn't cause problems because at the very least the program must contain constraint apis
            # exclude DSubTree node, randint is [a,b] inclusive
            rand_node_pos = random.randint(1,
                                           self.curr_prog.length - 1)
            node = self.get_node_in_position(rand_node_pos)

            # Checks parent edge to prevent deleting half a branch or leave dangling D-nodes
            if node.api_name != STOP and node.parent_edge != CHILD_EDGE:
                return node, rand_node_pos

    def add_random_node(self):
        """
        Adds a node to a random position in the current program.
        Node is chosen probabilistically based on all the nodes that come before it (DFS).
        :return:
        """
        # if tree not at max AST depth, can add a node
        if self.curr_prog.non_dnode_length >= self.max_num_api or self.curr_prog.length >= self.max_length:
            return None

        # Get a random position in the tree to be the parent of the new node to be added
        rand_node_pos = random.randint(1,
                                       self.curr_prog.length - 1)  # exclude DSubTree node, randint is [a,b] inclusive
        new_node_parent = self.get_node_in_position(rand_node_pos)

        # Probabilistically choose the node that should appear after selected random parent
        new_node_api = self.node2vocab[self.get_ast_idx(rand_node_pos, non_dnode=False)]
        # new_node_api = self.node2vocab[self.get_ast_idx(rand_node_pos)]

        print(new_node_api)

        # If a dnode is chosen, grow it out
        if new_node_api == DBRANCH:
            return self.grow_dbranch(new_node_parent)
        elif new_node_api == DLOOP:
            return self.grow_dloop(new_node_parent)
        elif new_node_api == DEXCEPT:
            return self.grow_dexcept(new_node_parent)

        # Add node to parent
        if new_node_parent.sibling is None:
            new_node = self.create_and_add_node(new_node_api, new_node_parent, SIBLING_EDGE)
        else:
            old_sibling_node = new_node_parent.sibling
            new_node_parent.remove_node(SIBLING_EDGE)
            new_node = self.create_and_add_node(new_node_api, new_node_parent, SIBLING_EDGE)
            new_node.add_node(old_sibling_node, SIBLING_EDGE)

        # Calculate probability of new program
        self.calculate_probability()

        return new_node

    def undo_add_random_node(self, added_node):
        """
        Undoes add_random_node() and returns current program to state it was in before the given node was added.
        :param added_node: (Node) the node that was added in add_random_node() that is to be removed.
        :return:
        """
        print(added_node.api_name)
        print(added_node.api_name in {DBRANCH, DLOOP, DEXCEPT})
        if added_node.api_name in {DBRANCH, DLOOP, DEXCEPT}:
            return self.undo_add_random_dnode(added_node)

        if added_node.sibling is None:
            added_node.parent.remove_node(SIBLING_EDGE)
        else:
            sibling_node = added_node.sibling
            parent_node = added_node.parent
            added_node.parent.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling_node, SIBLING_EDGE)

        self.curr_log_prob = self.prev_log_prob

    def delete_random_node(self):
        """
        Deletes a random node in the current program.
        :return: (Node) deleted node, (Node) deleted node's parent node,
        (bool- SIBLING_EDGE or CHILD_EDGE) edge between deleted node and its parent
        """
        node, _ = self.get_deletable_node()
        print("deleted node:", node.api_name)
        parent_node = node.parent
        parent_edge = node.parent_edge

        parent_node.remove_node(parent_edge)

        # If a sibling edge was removed and removed node has sibling node, add that sibling node to parent
        if parent_edge == SIBLING_EDGE and node.sibling is not None:
            sibling = node.sibling
            node.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling, SIBLING_EDGE)

        # # Remove dangling DStop node if there is one
        # elif parent_edge == SIBLING_EDGE and node.sibling is None:
        #     last_node = self.get_node_in_position(self.curr_prog.length - 1)
        #     if last_node.api_name == STOP:
        #         last_node.parent.remove_node(last_node.parent_edge)

        self.calculate_probability()

        return node, parent_node, parent_edge

    def undo_delete_random_node(self, node, parent_node, edge):
        """
        Adds back the node deleted from the program in delete_random_node(). Restores program state to what it was
        before delete_random_node() was called.
        :param node: (Node) node that was deleted
        :param parent_node: (Node) parent of the node that was deleted
        :param edge: (bool- SIBLING_EDGE or CHILD_EDGE) edge between deleted node and its parent
        :return:
        """
        sibling = None
        if edge == SIBLING_EDGE:
            if parent_node.sibling is not None:
                sibling = parent_node.sibling
        parent_node.add_node(node, edge)
        if sibling is not None:
            node.add_node(sibling, SIBLING_EDGE)
        self.curr_log_prob = self.prev_log_prob

    def random_swap(self):
        """
        Randomly swaps 2 nodes in the current program. Only the node will be swapped, the subtree will be detached and
        added to the node its being swapped with.
        :return: (Node) node that was swapped, (Node) other node that was swapped
        """
        # get 2 distinct node positions
        # node1, rand_node1_pos = self.get_valid_random_node()  # TODO: FIX THIS BECAUSE IT IS WRONG
        node1, rand_node1_pos = self.get_random_node_to_swap()
        other_nodes = list(range(1, self.curr_prog.length))
        other_nodes.remove(rand_node1_pos)
        # node2, node2_pos = self.get_valid_random_node(given_list=other_nodes)
        node2, node2_pos = self.get_random_node_to_swap(given_list=other_nodes)

        if node1 is None or node2 is None:
            return None, None

        # swap nodes
        self.swap_nodes(node1, node2)

        self.calculate_probability()

        return node1, node2

    def swap_nodes(self, node1, node2):
        """
        Swap given nodes. Only swaps individual nodes and not their subtrees as well.
        :param node1: (Node) node to be swapped
        :param node2: (Node) node to be swapped
        :return:
        """

        # Save parents and parent edges for nodes
        node1_parent = node1.parent
        node2_parent = node2.parent
        node1_edge = node1.parent_edge
        node2_edge = node2.parent_edge

        # If one node is the parent of another
        if node1_parent == node2 or node2_parent == node1:
            if node1_parent == node2:
                parent = node2
                node = node1
            else:
                parent = node1
                node = node2

            # get pointers to parent child and sibling nodes
            parent_edge = node.parent_edge
            if parent_edge == SIBLING_EDGE:
                parent_other_node = parent.child
                parent_other_edge = CHILD_EDGE
            else:
                parent_other_node = parent.sibling
                parent_other_edge = SIBLING_EDGE

            # get grandparent node and edge
            grandparent_node = parent.parent
            grandparent_edge = parent.parent_edge

            # remove nodes from parent
            parent.remove_node(SIBLING_EDGE)
            parent.remove_node(CHILD_EDGE)

            # get pointers to node's child and siblings and remove them
            node_child = node.child
            node_sibling = node.sibling
            node.remove_node(SIBLING_EDGE)
            node.remove_node(CHILD_EDGE)

            # add node to grandparent
            grandparent_node.add_node(node, grandparent_edge)

            # add old parent and other other to new parent node
            node.add_node(parent, parent_edge)
            node.add_node(parent_other_node, parent_other_edge)

            # add node's child and sibling to parent
            parent.add_node(node_child, CHILD_EDGE)
            parent.add_node(node_sibling, SIBLING_EDGE)

        else:
            # remove from parents
            node1_parent.remove_node(node1_edge)
            node2_parent.remove_node(node2_edge)

            # save all siblings and children
            node1_sibling = node1.sibling
            node1_child = node1.child
            node2_sibling = node2.sibling
            node2_child = node2.child

            # remove all siblings and children
            node1.remove_node(SIBLING_EDGE)
            node1.remove_node(CHILD_EDGE)
            node2.remove_node(SIBLING_EDGE)
            node2.remove_node(CHILD_EDGE)

            # add siblings and children to swapped nodes
            node1.add_node(node2_sibling, SIBLING_EDGE)
            node1.add_node(node2_child, CHILD_EDGE)
            node2.add_node(node1_sibling, SIBLING_EDGE)
            node2.add_node(node1_child, CHILD_EDGE)

            # and nodes back to swapped parents
            node1_parent.add_node(node2, node1_edge)
            node2_parent.add_node(node1, node2_edge)

    def undo_swap_nodes(self, node1, node2):
        self.swap_nodes(node1, node2)
        self.curr_log_prob = self.prev_log_prob

    def grow_dbranch(self, parent):
        """
        Create full DBranch (DBranch, condition, then, else) from parent node.
        :param parent: (Node) parent of DBranch
        :return: (Node) DBranch node
        """
        print("HELLOOOO")
        # remove parent's current sibling node if there
        parent_sibling = parent.sibling
        parent.remove_node(SIBLING_EDGE)

        # Create a DBranch node
        dbranch = self.create_and_add_node(DBRANCH, parent, SIBLING_EDGE)
        print("dbranch is none: ", dbranch is None)
        dbranch_pos = self.get_nodes_position(dbranch)
        assert dbranch_pos > 0, "Error: DBranch position couldn't be found"

        # Add parent's old sibling node to DBranch with sibling edge
        dbranch.add_node(parent_sibling, SIBLING_EDGE)

        # If adding DBranch to end of tree, do not add DStop node to last sibling of branch
        add_stop_node_to_end_branch = not (dbranch_pos == (self.curr_prog.length - 1))

        # Ensure adding a DBranch won't exceed max depth
        if add_stop_node_to_end_branch and (
                self.curr_prog.non_dnode_length + 3 > self.max_num_api or self.curr_prog.length + 6 > self.max_length):
            # remove added dbranch
            parent.remove_node(SIBLING_EDGE)
            parent.sibling = parent_sibling
            return None
        elif (not add_stop_node_to_end_branch) and (
                self.curr_prog.non_dnode_length + 3 > self.max_num_api or self.curr_prog.length + 5 > self.max_length):
            # remove added dbranch
            parent.remove_node(SIBLING_EDGE)
            parent.sibling = parent_sibling
            return None

        # Create condition as DBranch child
        condition = self.create_and_add_node(self.node2vocab[self.get_ast_idx(dbranch_pos)], dbranch, CHILD_EDGE)
        cond_pos = self.get_nodes_position(condition)
        assert cond_pos > 0, "Error: Condition node position couldn't be found"

        # Add then api as child to condition node
        then_node = self.create_and_add_node(self.node2vocab[self.get_ast_idx(cond_pos)], condition, CHILD_EDGE)
        self.create_and_add_node(STOP, then_node, SIBLING_EDGE)

        # Add else api as sibling to condition node
        else_node = self.create_and_add_node(self.node2vocab[self.get_ast_idx(cond_pos)], condition, SIBLING_EDGE)
        if add_stop_node_to_end_branch:
            self.create_and_add_node(STOP, else_node, SIBLING_EDGE)

        # Calculate probability of new program
        self.calculate_probability()

        # print(condition.api_name)
        # print(then_node.api_name)
        # print(else_node.api_name)

        return dbranch

    def grow_dloop(self, parent):
        """
        Create full DLoop (DLoop, condition, body) from parent node
        :param parent: (Node) parent of DLoop
        :return: (Node) DLoop node
        """
        # remove parent's current sibling node if there
        parent_sibling = parent.sibling
        parent.remove_node(SIBLING_EDGE)

        # Create a DLoop node
        dloop = self.create_and_add_node(DLOOP, parent, SIBLING_EDGE)
        dloop_pos = self.get_nodes_position(dloop)
        assert dloop_pos > 0, "Error: DLoop position couldn't be found"

        # Add parent's old sibling node to DLoop with sibling edge
        dloop.add_node(parent_sibling, SIBLING_EDGE)

        # If adding DLoop to end of tree, do not add DStop node to last sibling of branch
        add_stop_node_to_end_branch = not (dloop_pos == (self.curr_prog.length - 1))

        # Ensure adding a DLoop won't exceed max depth
        if add_stop_node_to_end_branch and (
                self.curr_prog.non_dnode_length + 2 > self.max_num_api or self.curr_prog.length + 4 > self.max_length):
            # remove added dloop
            parent.remove_node(SIBLING_EDGE)
            parent.sibling = parent_sibling
            return None
        elif (not add_stop_node_to_end_branch) and (
                self.curr_prog.non_dnode_length + 2 > self.max_num_api or self.curr_prog.length + 3 > self.max_length):
            # remove added dloop
            parent.remove_node(SIBLING_EDGE)
            parent.sibling = parent_sibling
            return None

        # Create condition as DLoop child
        condition = self.create_and_add_node(self.node2vocab[self.get_ast_idx(dloop_pos)], dloop, CHILD_EDGE)
        cond_pos = self.get_nodes_position(condition)
        assert cond_pos > 0, "Error: Condition node position couldn't be found"

        # Add body api as child to condition node
        body = self.create_and_add_node(self.node2vocab[self.get_ast_idx(cond_pos)], condition, CHILD_EDGE)
        if add_stop_node_to_end_branch:
            self.create_and_add_node(STOP, body, SIBLING_EDGE)

        # Calculate probability of new program
        self.calculate_probability()

        return dloop

    def grow_dexcept(self, parent):
        """
        Create full DExcept (DExcept, catch, try) from parent node
        :param parent: (Node) parent of DExcept
        :return: (Node) DExcept node
        """
        # remove parent's current sibling node if there
        parent_sibling = parent.sibling
        parent.remove_node(SIBLING_EDGE)

        # Create a DExcept node
        dexcept = self.create_and_add_node(DEXCEPT, parent, SIBLING_EDGE)
        dexcept_pos = self.get_nodes_position(dexcept)
        assert dexcept_pos > 0, "Error: DExcept position couldn't be found"

        # Add parent's old sibling node to DExcept with sibling edge
        dexcept.add_node(parent_sibling, SIBLING_EDGE)

        # If adding DLoop to end of tree, do not add DStop node to last sibling of branch
        add_stop_node_to_end_branch = not (dexcept_pos == (self.curr_prog.length - 1))

        # Ensure adding a DExcept won't exceed max depth
        if add_stop_node_to_end_branch and (
                self.curr_prog.non_dnode_length + 2 > self.max_num_api or self.curr_prog.length + 4 > self.max_length):
            # remove added dexcept
            parent.remove_node(SIBLING_EDGE)
            parent.sibling = parent_sibling
            return None
        elif (not add_stop_node_to_end_branch) and (
                self.curr_prog.non_dnode_length + 2 > self.max_num_api or self.curr_prog.length + 3 > self.max_length):
            # remove added dexcept
            parent.remove_node(SIBLING_EDGE)
            parent.sibling = parent_sibling
            return None

        # Create catch as DExcept child
        catch = self.create_and_add_node(self.node2vocab[self.get_ast_idx(dexcept_pos)], dexcept, CHILD_EDGE)
        catch_pos = self.get_nodes_position(catch)
        assert catch_pos > 0, "Error: Catch node position couldn't be found"

        # Add try api as child to catch node
        try_node = self.create_and_add_node(self.node2vocab[self.get_ast_idx(catch_pos)], catch, CHILD_EDGE)
        if add_stop_node_to_end_branch:
            self.create_and_add_node(STOP, try_node, SIBLING_EDGE)

        # Calculate probability of new program
        self.calculate_probability()

        return dexcept

    def add_random_dnode(self):
        """
        Adds a DBranch, DLoop or DExcept to a random node in the current program.
        :return: (Node) the dnode node
        """
        dnode_type = random.choice([DBRANCH, DLOOP, DEXCEPT])

        parent, _ = self.get_valid_random_node()

        if parent is None:
            return None

        assert parent.child is None or parent.parent.api_name == DBRANCH, \
            "WARNING: there's a bug in get_valid_random_node because parent node has child"

        # Grow dnode type
        if dnode_type == DBRANCH:
            return self.grow_dbranch(parent)
        elif dnode_type == DLOOP:
            return self.grow_dloop(parent)
        else:
            return self.grow_dexcept(parent)

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

        self.curr_log_prob = self.prev_log_prob

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
                    return False

            # Check that DBranch has the proper form
            if curr_node.api_name == DBRANCH:
                if curr_node.child is None:
                    return False
                if curr_node.child.child is None or curr_node.child.sibling is None \
                        or curr_node.child.child.sibling is None:
                    return False
                if curr_node.child.api_name in DNODES or curr_node.child.child.api_name in DNODES \
                        or curr_node.child.sibling.api_name in DNODES:
                    return False
                if curr_node.child.child.sibling.api_name != STOP:
                    return False
                # If this is not the end of program, there should be a DStop node after else node
                if not (curr_node.sibling is None and len(stack) == 0):
                    if curr_node.child.sibling.sibling is None:
                        return False
                    if curr_node.child.sibling.sibling.api_name != STOP:
                        return False

            # Check that DLoop and DExcept have the proper form
            if curr_node.api_name == DLOOP or curr_node.api_name == DEXCEPT:
                if curr_node.child is None:
                    return False
                if curr_node.child.child is None:
                    return False
                if curr_node.child.api_name in DNODES or curr_node.child.child.api_name in DNODES:
                    return False
                # If this is not the end of program, there should be a DStop node
                if not (curr_node.sibling is None and len(stack) == 0):
                    if curr_node.child.child.sibling is None:
                        return False
                    if curr_node.child.child.sibling.api_name != STOP:
                        return False

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
                    curr_node = None

        # Last node in program cannot be DStop node
        if last_node.api_name == STOP:
            print("HERE")
            return False
        print(len(constraints))
        # Return whether all constraints have been met
        return len(constraints) == 0

    def validate_and_update_program(self):
        """
        Validate current program and if valid decide whether to accept or reject it.
        :return: (bool) whether to accept or reject current program
        """
        if self.check_validity():
            self.valid += 1
            return self.accept_or_reject()

        self.invalid += 1
        return False

    def accept_or_reject(self):
        """
        Calculates whether to accept or reject current program based on Metropolis Hastings algorithm.
        :return: (bool)
        """
        print(math.exp(self.curr_log_prob))
        alpha = self.curr_log_prob - self.prev_log_prob
        print(alpha)
        mu = math.log(random.uniform(0, 1))
        # mu = math.log(random.uniform(0, 2 * 10 ** -40))
        print(mu)
        if mu < alpha:
            self.prev_log_prob = self.curr_log_prob  # TODO: add logging for graph here
            self.accepted += 1
            return True
        else:
            # TODO: add logging
            self.rejected += 1
            return False

    def transform_tree(self):
        """
        Randomly chooses a transformation and transforms the current program if it is accepted.
        :return: (bool) whether current tree was transformed or not
        """
        move = random.choice(['add', 'delete'])
        # move = np.random.choice(['add', 'delete', 'swap', 'add_dnode'], p=[0.3, 0.3, 0.3, 0.1])
        print("move:", move)

        if move == 'add':
            self.add += 1
            added_node = self.add_random_node()
            if added_node is None:
                print("node is none")
                return False
            print("valid:", self.validate_and_update_program())
            if not self.validate_and_update_program():
                self.undo_add_random_node(added_node)
                return False
            self.add_accepted += 1
        elif move == 'delete':
            self.delete += 1
            node, parent_node, parent_edge = self.delete_random_node()
            if not self.validate_and_update_program():
                self.undo_delete_random_node(node, parent_node, parent_edge)
                return False
            self.delete_accepted += 1
        elif move == 'swap':
            self.swap += 1
            node1, node2 = self.random_swap()
            if node1 is None or node2 is None:
                return False
            if not self.validate_and_update_program():
                self.undo_swap_nodes(node1, node2)
                return False
            self.swap_accepted += 1
        elif move == 'add_dnode':
            self.add_dnode += 1
            dnode = self.add_random_dnode()
            if dnode is None:
                return False
            if not self.validate_and_update_program():
                self.undo_add_random_dnode(dnode)
                return False
            self.add_dnode_accepted += 1
        else:
            raise ValueError('move not defined')  # TODO: remove once tested

        print("move was successful")
        return True

    def get_next_ast_state_and_prob(self, ast_node, ast_edge, ast_state):
        """
        Calculates next decoder state and logits after feeding in given node and edge.
        :param ast_node: (int) node number (from vocabulary)
        :param ast_edge: (bool- SIBLING_EDGE or CHILD_EDGE) edge between node and parent
        :param ast_state: current state of decoder
        :return: (np.array) current state of decoder, (np.array) normalized logits from decoder
        """
        # Create feed
        feed = {self.model.nodes.name: np.array(ast_node, dtype=np.int32),
                self.model.edges.name: np.array(ast_edge, dtype=np.bool)}
        for i in range(self.config.decoder.num_layers):
            feed[self.model.initial_state[i].name] = np.array(ast_state[i])

        # Pass through model
        [ast_state, ast_ln_prob] = self.sess.run(
            [self.model.decoder.ast_tree.state, self.ast_ln_probs], feed)

        return ast_state, ast_ln_prob

    def get_ast_idx(self, parent_pos, non_dnode=True):
        # return self.get_ast_idx_all_vocab(parent_pos, non_dnode)

        # return self.get_ast_idx_top_k(parent_pos, non_dnode)

        return self.get_ast_idx_random_top_k(parent_pos, non_dnode)

    def get_ast_idx_all_vocab(self, parent_pos, non_dnode):  # TODO: FIX AND TEST
        """
        Returns api number (based on vocabulary). Probabilistically selected based on parent node.
        :param parent_pos: (int) position of parent node in current program (by DFS)
        :return: (int) number of api in vocabulary
        """
        void_ret_type = self.config.vocab.ret_dict["void"]
        nodes, edges = self.get_vector_representation()
        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)
        fp = np.zeros([1, 1], dtype=np.int32)

        # Pass in all nodes that appear before and including the parent through the decoder to get normalized logits
        for i in range(parent_pos + 1):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            feed = {self.model.edges.name: node, self.model.nodes.name: edge,
                    self.model.return_type: [void_ret_type],
                    self.model.formal_params: fp}
            if i < parent_pos:
                self.sess.run(self.model.decoder.ast_logits, feed)
            else:
                # Sample from normalized logits
                ast_idx = tf.multinomial(self.model.decoder.ast_logits[0], 1)
                ast_idx = self.sess.run(ast_idx, feed)
                return ast_idx[0][0]

    def get_ast_idx_top_k(self, parent_pos, non_dnode, top_k=10):  # TODO: FIX AND TEST
        """
                Returns api number (based on vocabulary). Probabilistically selected from top k based on parent node.
                :param parent_pos: (int) position of parent node in current program (by DFS)
                :return: (int) number of api in vocabulary
                """
        void_ret_type = self.config.vocab.ret_dict["void"]
        nodes, edges = self.get_vector_representation()
        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)
        fp = np.zeros([1, 1], dtype=np.int32)

        # Pass in all nodes that appear before and including the parent through the decoder to get normalized logits
        for i in range(parent_pos + 1):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            feed = {self.model.edges.name: node, self.model.nodes.name: edge,
                    self.model.return_type: [void_ret_type],
                    self.model.formal_params: fp}
            if i < parent_pos:
                self.sess.run(self.model.decoder.ast_logits, feed)
            else:
                # get topk logits
                # index twice into logits because logits.shape = [1, 1, vocab_size])

                # # ---- CODE FOR DEBUGGING ------
                # logits = self.model.decoder.ast_logits[0]
                # vals, idxs = tf.math.top_k(logits, k=top_k)
                # chosen_idx, vals, idxs, logits = self.sess.run([tf.multinomial(vals, 1)[0][0], vals[0], idxs[0], logits], feed)
                # names = [self.node2vocab[i] for i in idxs]
                # print(sorted(logits[0], reverse=True))
                # print(chosen_idx)
                # print(self.node2vocab[idxs[chosen_idx]])
                # print(idxs)
                # print(names)
                # print(vals)

                vals, idxs = tf.math.top_k(self.model.decoder.ast_logits[0], k=top_k)
                chosen_idx = self.sess.run(idxs[0][tf.multinomial(vals, 1)[0][0]], feed)

                return chosen_idx

    def get_ast_idx_random_top_k(self, parent_pos, non_dnode, top_k=10):  # TODO: TEST
        """
                Returns api number (based on vocabulary). Uniform randomly selected from top k based on parent node.
                :param parent_pos: (int) position of parent node in current program (by DFS)
                :return: (int) number of api in vocabulary
                """
        # void_ret_type = self.config.vocab.ret_dict["void"]
        # nodes, edges = self.get_vector_representation()
        # node = np.zeros([1, 1], dtype=np.int32)
        # edge = np.zeros([1, 1], dtype=np.bool)
        # fp = np.zeros([1, 1], dtype=np.int32)
        #
        # # Pass in all nodes that appear before and including the parent through the decoder to get normalized logits
        # for i in range(parent_pos + 1):
        #     node[0][0] = nodes[i]
        #     edge[0][0] = edges[i]
        #     feed = {self.model.edges.name: node, self.model.nodes.name: edge,
        #             self.model.return_type: [void_ret_type],
        #             self.model.formal_params: fp}
        #     if i < parent_pos:
        #         self.sess.run(self.model.decoder.ast_logits, feed)
        #     else:
        #         if non_dnode:  # TODO: make this code better
        #             _, idxs = self.sess.run(tf.math.top_k(self.model.decoder.ast_logits[0], k=top_k), feed)
        #             selectable = []
        #             for k in range(top_k):
        #                 if self.node2vocab[idxs[0][k]] not in DNODES:
        #                     selectable.append(idxs[0][k])
        #             rand_idx = random.randint(0, len(selectable) - 1)
        #             return idxs[0][rand_idx]
        #         else:
        #             rand_idx = random.randint(0, top_k - 1)  # Note: randint is a,b inclusive
        #             _, idxs = tf.math.top_k(self.model.decoder.ast_logits[0], k=top_k)
        #             return self.sess.run(idxs[0][rand_idx], feed)

        state = self.initial_state
        nodes, edges = self.get_vector_representation()

        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)

        for i in range(parent_pos + 1):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            if i == parent_pos:
                if non_dnode:  # TODO: make this code better
                    _, idxs, _ = self.decoder.get_next_ast_state(node, edge, state)
                    selectable = []
                    for k in range(self.decoder.top_k):
                        if self.node2vocab[idxs[0][k]] not in DNODES:
                            selectable.append(idxs[0][k])
                    rand_idx = random.randint(0, len(selectable) - 1)
                    return idxs[0][rand_idx]
                else:
                    rand_idx = random.randint(0, self.decoder.top_k - 1)  # Note: randint is a,b inclusive
                    _, idxs, _ = self.decoder.get_next_ast_state(node, edge, state)
                    return idxs[0][rand_idx]
            else:
                state, _ = self.decoder.get_ast_logits(node, edge, state)

    def random_walk_latent_space(self):
        """
        Do a random walk in latent space Z.
        :return:
        """
        samples = tf.random.normal([self.config.batch_size, self.config.latent_size], mean=0., stddev=1.,
                                   dtype=tf.float32)
        latent_state = self.encoder_mean + tf.sqrt(self.encoder_covar) * samples
        self.latent_state = self.sess.run(latent_state, {})

    def update_latent_state(self):
        """
        Update encoder mean, encoder covariance and encoder latent state
        :return: (np.array) new latent state
        """
        void_ret_type = self.config.vocab.ret_dict["void"]
        nodes, edges = self.get_vector_representation()
        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)
        fp = np.zeros([1, 1], dtype=np.int32)

        ret_type = [self.rettype2num["Typeface"]]
        fp_type = np.zeros([1, 1], dtype=np.int32)
        fp_type[0][0] = self.fp2num["String"]
        ret_type = np.array(ret_type)
        fp_type = np.array(fp_type)

        state = None
        for i in range(self.curr_prog.length):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            feed = {self.model.edges.name: node, self.model.nodes.name: edge,
                    self.model.return_type: ret_type,
                    self.model.formal_params: fp_type}
            if i < self.curr_prog.length - 1:
                state = self.sess.run(self.model.latent_state, feed)
            else:  # for last node, save encoder mean and variance as well
                [state, self.encoder_mean, self.encoder_covar] = self.sess.run(
                    [self.model.latent_state, self.model.encoder.output_mean, self.model.encoder.output_covar], feed)
                self.latent_state = state
        return state

    def get_initial_decoder_state(self):
        """
        Get initial state of the decoder given the encoder's latent state
        :return:
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--continue_from', type=str, default=self.save_dir,
                            help='ignore config options and continue training model checkpointed here')
        clargs = parser.parse_args()

        encoder = BayesianPredictor(clargs.continue_from, batch_size=1)

        nodes, edges = self.get_vector_representation()
        nodes = nodes[:self.max_num_api]
        edges = edges[:self.max_num_api]
        nodes = np.array([nodes])
        edges = np.array([edges])
        print(nodes.shape)
        print(edges.shape)
        print(np.array(self.fp).shape)
        print(np.array(self.ret_type).shape)
        self.initial_state = encoder.get_initial_state(nodes, edges, np.array(self.ret_type), np.array(self.fp))
        self.initial_state = np.transpose(np.array(self.initial_state), [1, 0, 2])  # batch_first
        print(self.initial_state)
        encoder.close()

        beam_width = 1
        self.decoder = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)

    def get_random_initial_state(self):
        self.latent_state = np.random.normal(loc=0., scale=1.,
                                        size=(self.config.batch_size, self.config.latent_size))
        self.initial_state = self.sess.run(self.model.initial_state,
                                      feed_dict={self.model.latent_state: self.latent_state})
        self.initial_state = np.transpose(np.array(self.initial_state), [1, 0, 2])  # batch-first
        return self.initial_state

    def calculate_probability(self):
        """
        Calculate probability of current program.
        :return:
        """
        nodes, edges = self.get_vector_representation()
        node = np.zeros([self.config.batch_size, self.config.max_ast_depth], dtype=np.int32)
        edge = np.zeros([self.config.batch_size, self.config.max_ast_depth], dtype=np.bool)
        state = self.initial_state
        curr_prob = 0.0

        for i in range(self.curr_prog.length):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            if i == self.curr_prog.length - 1:
                # add prob of stop node
                state, ast_prob = self.decoder.get_ast_logits(node, edge, state)
                stop_node = self.vocab2node[STOP]
                curr_prob += ast_prob[0][stop_node]
            else:
                state, ast_prob = self.decoder.get_ast_logits(node, edge, state)
                print(ast_prob[0][nodes[i + 1]])
                print(math.exp(ast_prob[0][nodes[i + 1]]))
                curr_prob += ast_prob[0][nodes[i + 1]]

        print(curr_prob / self.curr_prog.length)

        self.curr_log_prob = curr_prob / self.curr_prog.length

        return curr_prob

    def mcmc(self):
        """
        Perform one MCMC step.
        1) Try to transform program tree.
        2) If accepted, update the latent space.
        3) Do a random walk in the latent space.
        4) Compute the new initial state of the decoder.
        :return:
        """
        self.transform_tree()

        # # Attempt to transform the current program
        # if self.transform_tree():
        # # If successful, update encoder's latent state
        #     self.update_latent_state()
        # # self.transform_tree()
        # self.random_walk_latent_space()
        # self.get_initial_decoder_state()

    def reader(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--data', type=str, default='data/',
                            help='load data from here')

        clargs = parser.parse_args()

        reader = Reader(clargs)
        reader.save_data('data/')

    def encode(self):
        nodes, edges = self.get_vector_representation()
        nodes = nodes[:self.max_num_api]
        edges = edges[:self.max_num_api]
        nodes = np.array([nodes])
        edges = np.array([edges])

        ret_type = [self.rettype2num["Typeface"]]

        fp = np.zeros([1, self.max_num_api])
        fp[0][0] = self.fp2num["String"]
        fp[0][1] = self.fp2num["int"]

        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--continue_from', type=str, default=self.save_dir,
                            help='ignore config options and continue training model checkpointed here')
        clargs = parser.parse_args()

        encoder = BayesianPredictor(clargs.continue_from, batch_size=1)
        psi = encoder.get_initial_state(nodes, edges, ret_type, fp)
        self.initial_state = np.transpose(np.array(psi), [1, 0, 2])  # batch_first
        encoder.close()

    def decode_beam_search(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--continue_from', type=str, default=self.save_dir,
                            help='ignore config options and continue training model checkpointed here')
        clargs = parser.parse_args()

        beam_width = 1
        decoder = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)
        program_beam_searcher = ProgramBeamSearcher(decoder)

        ast_paths, fp_paths, ret_types = program_beam_searcher.beam_search(initial_state=self.initial_state)

        print(' ========== AST ==========')
        for ast_path in ast_paths:
            print(ast_path)

        print(' ========== Fp ==========')
        for fp_path in fp_paths:
            print(fp_path)

        print(' ========== Return Type ==========')
        print(ret_types)

    def decode_calculate_prob(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--continue_from', type=str, default=self.save_dir,
                            help='ignore config options and continue training model checkpointed here')
        clargs = parser.parse_args()

        beam_width = 1
        decoder = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)

        state = self.initial_state
        nodes, edges = self.get_vector_representation()

        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)

        curr_prob = 0.0

        for i in range(self.curr_prog.length):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            if i == self.curr_prog.length - 1:
                # add prob of stop node
                state, idxs, ast_prob = decoder.get_next_ast_state(node, edge, state)
                print(ast_prob)
                print([self.node2vocab[i] for i in idxs[0]])
                max_arg = np.argmax(ast_prob[0])
                print(self.node2vocab[idxs[0][max_arg]])
                # stop_node = self.vocab2node[STOP]
                curr_prob += ast_prob[0][max_arg]
            else:
                state, ast_prob = decoder.get_ast_logits(node, edge, state)
                print(ast_prob[0][nodes[i + 1]])
                print(math.exp(ast_prob[0][nodes[i + 1]]))
                curr_prob += ast_prob[0][nodes[i + 1]]

        print(curr_prob / self.curr_prog.length)

        state, idxs, ast_prob = decoder.get_next_ast_state(node, edge, state)
        print(ast_prob)
        print([self.node2vocab[i] for i in idxs[0]])
        max_arg = np.argmax(ast_prob[0])
        print(self.node2vocab[idxs[0][max_arg]])
        print(math.exp(curr_prob))
        print(math.exp(curr_prob + ast_prob[0][max_arg]))
        print("difference", math.exp(ast_prob[0][max_arg]))
        print(math.exp(curr_prob + ast_prob[0][max_arg])/math.exp(curr_prob))
        curr_prob += ast_prob[0][max_arg]

        print(curr_prob / self.curr_prog.length)





    def loader(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--data', type=str, default='data/',
                            help='load data from here')
        parser.add_argument('--continue_from', type=str, default=self.save_dir,
                            help='ignore config options and continue training model checkpointed here')
        parser.add_argument('--vocab_path', type=str, default='../data_extractor/data/1k_vocab_constraint_min_3-600000/')
        clargs = parser.parse_args()

        # print()
        config = deepcopy(self.config)
        config.max_ast_depth = self.max_num_api
        loader = Loader(clargs, config)

        encoder = BayesianPredictor(clargs.continue_from, batch_size=1)

        nodes, edges, targets, \
        ret_type, fp_type, fp_type_targets = loader.next_batch()
        print(ret_type)
        print(fp_type)
        print(fp_type_targets)
        print(type(fp_type))
        print(nodes)

        loader_num2api = dict(zip(loader.config.vocab.api_dict.values(), loader.config.vocab.api_dict.keys()))
        loader_num2ret = dict(zip(loader.config.vocab.ret_dict.values(), loader.config.vocab.ret_dict.keys()))
        loader_num2fp = dict(zip(loader.config.vocab.fp_dict.values(), loader.config.vocab.fp_dict.keys()))

        nodes = [[loader_num2api[i] for i in nodes[0]]]
        ret_type = [loader_num2ret[i] for i in ret_type]
        fp_type = [[loader_num2fp[i] for i in fp_type[0]]]
        print(ret_type)
        print(fp_type)
        print(nodes)

        nodes = np.array([[self.vocab2node[i] for i in nodes[0]]])
        print(nodes.shape)
        # nodes = nodes.T
        # print(nodes.shape)
        ret_type = [self.rettype2num[i] for i in ret_type]
        fp_type = [[self.fp2num[i] for i in fp_type[0]]]
        fp = np.zeros([1,8])
        fp[0][0] = fp_type[0][0]
        fp[0][1] = self.fp2num["int"]
        print(ret_type)
        print(fp)
        print(nodes)
        print(edges)

        nodes[0][2] = 0

        psi = encoder.get_initial_state(nodes, edges, ret_type, fp)
        psi_ = np.transpose(np.array(psi), [1, 0, 2])  # batch_first
        encoder.close()

        # print(psi_)
        self.initial_state = psi_

        beam_width = 1
        decoder = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)
        program_beam_searcher = ProgramBeamSearcher(decoder)

        temp = psi_
        ast_paths, fp_paths, ret_types = program_beam_searcher.beam_search(initial_state=temp)

        print(' ========== AST ==========')
        for ast_path in ast_paths:
            print(ast_path)

        print(' ========== Fp ==========')
        for fp_path in fp_paths:
            print(fp_path)

        print(' ========== Return Type ==========')
        print(ret_types)

        print('\n\n\n\n')
        # decoder.close()
        self.model = decoder.model

        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)
        fp_input = np.zeros([1, 1], dtype=np.bool)
        fp_input[0][0] = fp[0][0]

        parent_pos = 2

        append = 'java.lang.StringBuilder.append(java.lang.String)'

        nodes = nodes[0]
        edges = edges[0]
        node[0][0] = self.vocab2node[append]
        edge[0][0] = edges[2]

        # state = decoder.get_random_initial_state()
        # ast_ln_probs = None
        # state = psi_

        next_state, idxs, probs = decoder.get_next_ast_state(node, edge, psi_)

        print(probs)

        print([self.node2vocab[i] for i in idxs[0]])

        api1 = idxs[0][0]

        node[0][0] = idxs[0][0]
        next_state, idxs, probs = decoder.get_next_ast_state(node, edge, next_state)

        print(probs)

        print([self.node2vocab[i] for i in idxs[0]])

        api2 = idxs[0][0]

        # # Pass in all nodes that appear before and including the parent through the decoder to get normalized logits
        # for i in range(parent_pos + 1):
        #     node[0][0] = nodes[i]
        #     edge[0][0] = edges[i]
        #     state, ast_ln_probs = decoder.get_ast_logits(node, edge, state)
        #     if i == parent_pos:
        #         rand_idx = random.randint(0, 10 - 1)  # Note: randint is a,b inclusive
        #         print(ast_ln_probs.shape)
        #         print(type(ast_ln_probs))
        #         order = ast_ln_probs.argsort()
        #         print(order)
        #         print(order[:, :10])
        #         order = order[0, :10]
        #         # print(ast_ln_probs[0][order])
        #         # _, idxs = self.sess.run(tf.math.top_k(ast_ln_probs, k=10), {})
        #         print([self.node2vocab[i] for i in order])
        #         # print(idx)


    def get_bayesian_predictor(self):
        HELP = """"""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(HELP))
        parser.add_argument('--python_recursion_limit', type=int, default=10000,
                            help='set recursion limit for the Python interpreter')
        parser.add_argument('--continue_from', type=str, default=self.save_dir,
                            help='ignore config options and continue training model checkpointed here')
        parser.add_argument('--saver', type=str, default='plots/beam_search/')
        parser.add_argument('--data', type=str, default='../data_extractor/data/1k_vocab_constraint_min_3-600000',
                            help='load data from here')

        clargs = parser.parse_args()
        if not os.path.exists(clargs.saver):
            os.mkdir(clargs.saver)

        sys.setrecursionlimit(clargs.python_recursion_limit)

        ret_type = [self.rettype2num["Typeface"]]
        fp_type = ["String"]  # "String", "int"
        fp_type = [self.fp2num[i] for i in fp_type]
        nodes, edges = self.get_vector_representation()
        nodes = nodes[:self.max_num_api]
        nodes = np.array([nodes])
        edges = edges[:self.max_num_api]
        edges = np.array([edges])
        ret_type = np.array(ret_type)
        fp_type = np.array([fp_type])
        encoder = BayesianPredictor(clargs.continue_from, batch_size=1)
        psi = encoder.get_initial_state(nodes, edges, ret_type, fp_type)
        psi_ = np.transpose(np.array(psi), [1, 0, 2])  # batch_first
        print(psi_)


        # feed = {self.model.edges.name: edges, self.model.nodes.name: nodes,
        #         self.model.return_type: ret_type,
        #         self.model.formal_params: fp_type}
        # state = self.sess.run(self.model.initial_state, feed)
        # psi = np.transpose(np.array(state), [1, 0, 2])  # batch_first

        # print(psi)

        HELP = """"""

        # reader = Reader(clargs, infer=True)
        # print(reader.read_return_type(ret_type))

        # beam_width = 20
        # self.predictor = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)
        # self.searcher = ProgramBeamSearcher(self.predictor)
        #
        # ast_paths, fp_paths, ret_types = self.searcher.beam_search()
        #
        # print(' ========== AST ==========')
        # for i, ast_path in enumerate(ast_paths):
        #     print(ast_path)
        #
        # print(' ========== Fp ==========')
        # for i, fp_path in enumerate(fp_paths):
        #     print(fp_path)
        #
        # print(' ========== Return Type ==========')
        # print(ret_types)
        #
        # nodes, edges = self.get_vector_representation()
        # return_types = ['boolean', 'String', 'void', 'File', 'OpenForReadResult', 'List<String>', '__UDT__', 'List',
        #                 'Bitmap', 'Long', 'ByteBuffer', 'URI', 'ClassFile', 'ArrayList<String>', 'Page', 'IMOps',
        #                 'double', 'Hashtable', 'Iterator<Object[]>', 'Map']
        # formal_params = ['DSubTree', 'String', 'boolean', 'int']
        # init_state = self.predictor.get_initial_state(nodes, edges, return_types, formal_params)


    # def get_encoder_psi(self):


        # psi = self.model.encoder.get_initial_state(nodes, edges, ret_type, fp_type)
        # psi = np.transpose(np.array(psi), [1, 0, 2])  # batch_first

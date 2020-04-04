import os
from copy import deepcopy
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


class TooLongLoopingException(Exception):
    pass


class TooLongBranchingException(Exception):
    pass


class Node:
    def __init__(self, id, api_name, api_num, parent, parent_edge):
        self.api_name = api_name
        self.api_num = api_num
        self.child = None
        self.sibling = None
        self.valid = True
        self.parent = parent
        self.id = id
        self.length = 1
        self.parent_edge = parent_edge

    def update_length(self, length, add_or_sub):
        if add_or_sub == 'add':
            self.length += length
        else:
            self.length -= length
        if self.parent is not None:
            # print("self:", self.api_name)
            # print("parent:", self.parent.api_name)
            self.parent.update_length(length, add_or_sub)

    def add_node(self, node, edge):
        if node is not None:
            if edge == SIBLING_EDGE:
                if self.sibling is not None:
                    # print("add node: sibling edge not none")
                    self.remove_node(SIBLING_EDGE)
                self.update_length(node.length, 'add')
                self.sibling = node
                node.parent = self
                node.parent_edge = edge
            elif edge == CHILD_EDGE:
                if self.child is not None:
                    self.remove_node(CHILD_EDGE)
                self.update_length(node.length, 'add')
                self.child = node
                node.parent = self
                node.parent_edge = edge
            else:
                raise ValueError('edge must be SIBLING_EDGE or CHILD_EDGE')
        else:
            # print("node is None. Was not added to tree.")
            pass

    # def remove_node(self, edge):  # NOTE: DO NOT USE IT CREATES HARD TO FIND BUGS ALSO DOESN"T
    #                               # LOGICALLY MAKE SENSE TO HAVE 2 TYPES OF REMOVES
    #     if edge == SIBLING_EDGE:
    #         if self.sibling is not None:
    #             old_sibling = self.sibling
    #             old_sibling.parent = None
    #             old_sibling.parent_edge = None
    #             self.update_length(old_sibling.length, 'sub')
    #
    #             # If old sibling has sibling node, add that sibling node to self to end up removing only 1 node
    #             new_sibling = old_sibling.sibling
    #             self.sibling = new_sibling
    #             if new_sibling is not None:
    #                 new_sibling.parent = self
    #                 new_sibling.parent_edge = edge
    #                 self.update_length(new_sibling.length, 'add')
    #         else:
    #             print("Tried to remove sibling edge but it is not there.")
    #
    #     elif edge == CHILD_EDGE:
    #         if self.child is not None:
    #             child = self.child
    #             self.child = None
    #             child.parent = None
    #             child.parent_edge = None
    #             self.update_length(child.length, 'sub')
    #         else:
    #             print("Tried to remove child edge but it is not there.")
    #
    #     else:
    #         raise ValueError('edge must but a sibling or child edge')

    def remove_node(self, edge):
        if edge == SIBLING_EDGE:
            if self.sibling is not None:
                old_sibling = self.sibling
                self.sibling = None
                old_sibling.parent = None
                old_sibling.parent_edge = None
                self.update_length(old_sibling.length, 'sub')

        elif edge == CHILD_EDGE:
            if self.child is not None:
                child = self.child
                self.child = None
                child.parent = None
                child.parent_edge = None
                self.update_length(child.length, 'sub')

        else:
            raise ValueError('edge must but a sibling or child edge')

    # def skip_dsubtree_header(self):
    #     assert self.api_name == 'DSubTree'
    #     self = self.sibling
    #     assert self.val != 'DSubTree'
    #     return self

    def copy(self):
        new_node = Node(self.id, self.api_name, self.api_num, self.parent, self.parent_edge)
        if self.sibling is not None:
            new_sibling_node = self.sibling.copy()
            new_node.add_node(new_sibling_node, SIBLING_EDGE)
        if self.child is not None:
            new_child_node = self.child.copy()
            new_node.add_node(new_child_node, CHILD_EDGE)
        return new_node

    def depth_first_search(self):

        buffer = []
        stack = []
        dfs_id = None
        parent_id = 0
        if self is not None:
            stack.append((self, parent_id, SIBLING_EDGE))
            dfs_id = 0

        while (len(stack) > 0):

            item_triple = stack.pop()
            item = item_triple[0]
            parent_id = item_triple[1]
            edge_type = item_triple[2]

            buffer.append((item.val, parent_id, edge_type))

            if item.sibling is not None:
                stack.append((item.sibling, dfs_id, SIBLING_EDGE))

            if item.child is not None:
                stack.append((item.child, dfs_id, CHILD_EDGE))

            dfs_id += 1

        return buffer


class MCMCProgram:
    def __init__(self, save_dir):
        # Initialize model
        config_file = os.path.join(save_dir, 'config.json')
        with open(config_file) as f:
            self.config = config = read_config(json.load(f), infer=True)

        self.max_depth = self.config.max_ast_depth

        self.config.max_ast_depth = 1
        self.config.max_fp_depth = 1
        self.config.batch_size = 1

        self.model = Model(self.config)
        self.sess = tf.Session()
        self.restore(save_dir)

        self.constraints = []
        self.constraint_nodes = []  # has node numbers of constraints
        self.vocab2node = self.config.vocab.api_dict
        self.node2vocab = dict(zip(self.vocab2node.values(), self.vocab2node.keys()))
        self.curr_prog = None
        self.curr_log_prob = -0.0
        self.prev_log_prob = -0.0
        self.node_id_counter = 0
        self.latent_state = None
        self.initial_state = None
        self.encoder_mean = None
        self.encoder_covar = None

        # logging
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
        self.add_invalid = 0
        self.delete_invalid = 0
        self.swap_invalid = 0
        # self.prev_state = self.curr_state.copy()

        with tf.name_scope("ast_inference"):
            ast_logits = self.model.decoder.ast_logits[:, 0, :]
            self.ast_ln_probs = tf.nn.log_softmax(ast_logits)
            # self.ast_idx = tf.multinomial(ast_logits, 1)
            # self.ast_top_k_values, self.ast_top_k_indices = tf.nn.top_k(self.ast_ln_probs,
            #                                                             k=self.config.batch_size)
            # print(self.ast_ln_probs.shape)
            # print(self.ast_idx.shape)
            # print(self.config.vocab.api_dict_size)

    def restore(self, save):
        # restore the saved model
        vars_ = get_var_list('all_vars')
        old_saver = tf.compat.v1.train.Saver(vars_)
        ckpt = tf.train.get_checkpoint_state(save)
        old_saver.restore(self.sess, ckpt.model_checkpoint_path)
        return

    def get_next_ast_state_and_prob(self, ast_node, ast_edge, ast_state):
        feed = {self.model.nodes.name: np.array(ast_node, dtype=np.int32),
                self.model.edges.name: np.array(ast_edge, dtype=np.bool)}
        for i in range(self.config.decoder.num_layers):
            feed[self.model.initial_state[i].name] = np.array(ast_state[i])

        [ast_state, ast_ln_prob] = self.sess.run(
            [self.model.decoder.ast_tree.state, self.ast_ln_probs], feed)

        return ast_state, ast_ln_prob

    def get_ast_idx(self, parent_pos):
        void_ret_type = self.config.vocab.ret_dict["void"]
        nodes, edges = self.get_vector_representation()
        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)
        fp = np.zeros([1, 1], dtype=np.int32)
        for i in range(parent_pos+1):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            feed = {self.model.edges.name: node, self.model.nodes.name: edge,
                    self.model.return_type: [void_ret_type],
                    self.model.formal_params: fp}
            if i < parent_pos:
                self.sess.run(self.model.decoder.ast_logits, feed)
            else:
                ast_idx = tf.multinomial(self.model.decoder.ast_logits[0], 1)
                ast_idx = self.sess.run(ast_idx, feed)
                return ast_idx[0][0]

    def random_walk_latent_space(self):
        samples = tf.random.normal([self.config.batch_size, self.config.latent_size], mean=0., stddev=1.,
                                   dtype=tf.float32)
        latent_state = self.encoder_mean + tf.sqrt(self.encoder_covar) * samples
        self.latent_state = self.sess.run(latent_state, {})

    def update_latent_state(self):
        void_ret_type = self.config.vocab.ret_dict["void"]
        nodes, edges = self.get_vector_representation()
        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)
        fp = np.zeros([1, 1], dtype=np.int32)
        state = None
        for i in range(self.curr_prog.length):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            feed = {self.model.edges.name: node, self.model.nodes.name: edge,
                    self.model.return_type: [void_ret_type],
                    self.model.formal_params: fp}
            if i < self.curr_prog.length - 1:
                state = self.sess.run(self.model.latent_state, feed)
            else:  # for last node, save encoder mean and variance as well
                [state, self.encoder_mean, self.encoder_covar] = self.sess.run(
                    [self.model.latent_state, self.model.encoder.output_mean, self.model.encoder.output_covar], feed)
                self.latent_state = state
        return state

    def get_initial_decoder_state(self):
        # latent_state = np.random.normal(loc=0., scale=1.,
        #                  size=(self.config.batch_size, self.config.latent_size))
        initial_state = self.sess.run(self.model.initial_state,
                                      feed_dict={self.model.latent_state: self.latent_state})
        # print("init shape", np.array(initial_state).shape)
        initial_state = np.transpose(np.array(initial_state), [1, 0, 2])  # batch-first
        self.initial_state = initial_state
        return initial_state

    def init_program(self):
        # Initialize tree
        head = self.createAndAddNode('DSubTree', None, SIBLING_EDGE)
        self.curr_prog = head
        # self.prev_prog = head.copy()

        # Add constraint nodes to tree
        last_node = head
        for i in self.constraints:
            node = self.createAndAddNode(i, last_node, SIBLING_EDGE)
            last_node = node

        # Add stop node
        # self.createAndAddNode('DStop', last_node, SIBLING_EDGE)

        # curr_node = self.curr_prog
        # while curr_node is not None:
        #     curr_node = curr_node.sibling

        self.update_latent_state()
        self.get_initial_decoder_state()
        # print("latent state:", self.latent_state)
        # print("encoder mean:", self.encoder_mean)
        # print("encoder covar:", self.encoder_covar)
        # print("decoder init state:", self.get_initial_decoder_state())
        # self.random_walk_latent_space()

        # Update probabilities of tree
        self.calculate_probability()
        self.prev_log_prob = self.curr_log_prob

    def createAndAddNode(self, api_name, parent, edge):
        # Create node
        try:
            api_num = self.vocab2node[api_name]
            node = Node(self.node_id_counter, api_name, api_num, parent, edge)

            # Update unique node id generator
            self.node_id_counter += 1

            # Update parent according to edge
            if api_name != 'DSubTree':
                parent.add_node(node, edge)

            return node

        except KeyError:
            print(api_name, " is not in vocabulary. Node was not added.")
            return None

    def add_constraint(self, constraint):
        try:
            node_num = self.vocab2node[constraint]
            self.constraint_nodes.append(node_num)
            self.constraints.append(constraint)
        except KeyError:
            print("Constraint ", constraint, " is not in vocabulary. Will be skipped.")

    def create_program(self, constraints):
        for i in constraints:
            self.add_constraint(i)

        # Create initial tree
        self.init_program()

    # def update_latent_space(self):
    #     samples = tf.random.normal([self.config.batch_size, self.config.latent_size], mean=0., stddev=1., dtype=tf.float32)
    #     self.latent_state = self.model.encoder.output_mean + tf.sqrt(self.model.encoder.output_covar) * samples

    def get_random_api(self):
        while True:
            api = self.node2vocab[random.randint(1, self.model.config.vocab.api_dict_size - 1)]
            if api not in {'DBranch', 'DLoop', 'DExcept', 'DStop'}:
                return api

    def get_node_in_position(self, pos_num):
        # print("prog length:", self.curr_prog.length, "pos num:", pos_num)
        assert pos_num < self.curr_prog.length

        num_explored = 0
        stack = []
        curr_node = self.curr_prog

        counter = 0

        while num_explored != pos_num and curr_node is not None:
            counter += 1
            if counter > self.max_depth + 1:
                raise ValueError("WARNING: Caught in infinite loop")
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

    def check_if_stop(self, node):
        if node.api_name == 'DStop':
            node.remove_edge(CHILD_EDGE)
            node.remove_edge(SIBLING_EDGE)

    def get_non_branched_random_pos(self, given_list=None):
        while True:
            if given_list is None:
                rand_node_pos = random.randint(1,
                                               self.curr_prog.length - 1)  # exclude DSubTree node, randint is [a,b] inclusive
            elif len(given_list) == 0:
                return None, None
            else:
                rand_node_pos = random.choice(given_list)

            node = self.get_node_in_position(rand_node_pos)

            # if node.api_name not in {'DBranch', 'DStop', 'DLoop', 'DExcept'}:
            return node, rand_node_pos

    def get_non_stop_random_pos(self):
        while True:
            rand_node_pos = random.randint(1,
                                           self.curr_prog.length - 1)  # exclude DSubTree node, randint is [a,b] inclusive
            node = self.get_node_in_position(rand_node_pos)

            if node.api_name != 'DStop':
                return node, rand_node_pos

    def add_random_node(self):
        # if tree not at max AST depth
        if self.curr_prog.length >= self.max_depth:
            return None

        rand_node_pos = random.randint(1,
                                       self.curr_prog.length - 1)  # exclude DSubTree node, randint is [a,b] inclusive
        # new_node_api = self.get_random_api()
        new_node_parent = self.get_node_in_position(rand_node_pos)

        # print("ast_idx:", self.get_ast_idx(rand_node_pos))
        new_node_api = self.node2vocab[self.get_ast_idx(rand_node_pos)]

        # NOTE: what to do if I pick a DBranch or DLoop or DExcept?
        if new_node_parent.sibling is None:
            new_node = self.createAndAddNode(new_node_api, new_node_parent, SIBLING_EDGE)
        else:
            old_sibling_node = new_node_parent.sibling
            new_node_parent.remove_node(SIBLING_EDGE)
            new_node = self.createAndAddNode(new_node_api, new_node_parent, SIBLING_EDGE)
            new_node.add_node(old_sibling_node, SIBLING_EDGE)

        # # remove any children for Stop nodes
        # self.check_if_stop(new_node_parent)
        # self.check_if_stop(new_node)

        self.calculate_probability()

        return new_node

    def undo_add_random_node(self, added_node):
        if added_node.sibling is None:
            added_node.parent.remove_node(SIBLING_EDGE)
        else:
            sibling_node = added_node.sibling
            parent_node = added_node.parent
            added_node.parent.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling_node, SIBLING_EDGE)

        self.curr_log_prob = self.prev_log_prob

    def delete_random_node(self):
        node, _ = self.get_non_stop_random_pos()
        parent_node = node.parent
        parent_edge = node.parent_edge

        parent_node.remove_node(parent_edge)

        # If a sibling edge was removed and removed node has sibling node, add that sibling node to parent
        if parent_edge == SIBLING_EDGE and node.sibling is not None:
            sibling = node.sibling
            node.remove_node(SIBLING_EDGE)
            parent_node.add_node(sibling, SIBLING_EDGE)

        self.calculate_probability()

        return node, parent_node, parent_edge

    def undo_delete_random_node(self, node, parent_node, edge):
        sibling = None
        if edge == SIBLING_EDGE:
            if parent_node.sibling is not None:
                sibling = parent_node.sibling
        parent_node.add_node(node, edge)
        if sibling is not None:
            node.add_node(sibling, SIBLING_EDGE)
        self.curr_log_prob = self.prev_log_prob

    def random_swap(self):
        # get 2 distinct node positions
        node1, rand_node1_pos = self.get_non_branched_random_pos()
        other_nodes = list(range(1, self.curr_prog.length))
        other_nodes.remove(rand_node1_pos)
        node2, node2_pos = self.get_non_branched_random_pos(given_list=other_nodes)

        if node1 is None or node2 is None:
            return None, None

        # swap nodes
        self.swap_nodes(node1, node2)

        self.calculate_probability()

        return node1, node2

    def swap_subtrees(self, node1, node2):
        node1_parent = node1.parent
        node2_parent = node2.parent
        node1_edge = node1.parent_edge
        node2_edge = node2.parent_edge

        if not (node1_parent == node2 or node2_parent == node1):
            # remove from parents
            node1_parent.remove_node(node1_edge)
            node2_parent.remove_node(node2_edge)

            # and nodes back to swapped parents
            node1_parent.add_node(node2, node1_edge)
            node2_parent.add_node(node1, node2_edge)

        else:
           print("Can't swap subtrees of child-parent nodes")

    def swap_nodes(self, node1, node2):
        node1_parent = node1.parent
        node2_parent = node2.parent
        node1_edge = node1.parent_edge
        node2_edge = node2.parent_edge

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

    def check_validity(self):
        constraints = []
        constraints += self.constraint_nodes
        # print("constraints:", constraints)

        stack = []
        curr_node = self.curr_prog
        last_node = curr_node

        while curr_node is not None:
            last_node = curr_node
            # print("curr_node:", curr_node.api_name, curr_node.api_num)
            if curr_node.api_num in constraints:
                constraints.remove(curr_node.api_num)
                # print("updated constraints:", constraints)
            if curr_node.api_name == 'DStop':
                print("api is DStop")
                print("branches are none:", curr_node.sibling is None and curr_node.child is None)
                if not (curr_node.sibling is None and curr_node.child is None):
                    return False
            if curr_node.child is not None:
                if curr_node.sibling is not None:
                    stack.append(curr_node.sibling)
                curr_node = curr_node.child

            elif curr_node.sibling is not None:
                # print("curr node sibling:", curr_node.sibling.api_name)
                curr_node = curr_node.sibling
            else:
                if len(stack) > 0:
                    curr_node = stack.pop()
                else:
                    curr_node = None
            # print("stack:", [node.api_name for node in stack])

        # TODO: add logging for whether this is valid or not
        if last_node.api_name == 'DStop':
            return False

        return len(constraints) == 0

    def validate_and_update_program(self):
        if self.check_validity():
            self.valid += 1
            return self.accept_or_reject()

        self.invalid += 1
        return False

    def accept_or_reject(self):
        alpha = self.curr_log_prob / self.prev_log_prob
        # print("curr prob:", self.curr_log_prob)
        # print("prev prob:", self.prev_log_prob)
        # print("alpha:", alpha)
        mu = random.uniform(0, 1)
        if mu <= alpha:
            self.prev_log_prob = self.curr_log_prob  # TODO: add logging for graph here
            self.accepted += 1
            return True
        else:
            # TODO: add logging
            self.rejected += 1
            return False

    def transform_tree(self):
        # print("-------------------")
        nodes, edges = self.get_node_names_and_edges()
        # print("nodes:", nodes)
        # print("edges:", edges)
        move = random.choice(['add', 'delete', 'swap'])
        # print("move:", move)

        if move == 'add':
            self.add += 1
            added_node = self.add_random_node()
            if added_node is None:
                # add logging that tree is at max capacity
                print("added_node is None")
                return False
            print("added node:", added_node.api_name)
            # print("added node parent:", added_node.parent.api_name)
            print("add node valid: ", self.validate_and_update_program())
            if not self.validate_and_update_program():
                # print("undo add random node")
                self.undo_add_random_node(added_node)
                return False
            self.add_accepted += 1
        elif move == 'delete':
            self.delete += 1
            node, parent_node, parent_edge = self.delete_random_node()
            # print("deleted node is None:", node is None)
            if not self.validate_and_update_program():
                # print("undo delete random node:", node.api_name)
                self.undo_delete_random_node(node, parent_node, parent_edge)
                return False
            # print("deleted node:", node.api_name)
            self.delete_accepted += 1
        elif move == 'swap':
            self.swap += 1
            node1, node2 = self.random_swap()
            # print("node 1:", node1 is None, "node 2:", node2 is None)
            if node1 is None or node2 is None:
                # print("node1 or node2 is None")
                return False
            # print("swap node1:", node1.api_name, "swap node2:", node2.api_name)
            if not self.validate_and_update_program():
                # print("undo swap nodes")
                self.swap_nodes(node1, node2)
                self.curr_log_prob = self.prev_log_prob
                return False
            self.swap_accepted += 1
        else:
            raise ValueError('move not defined')  # TODO: remove once tested
            return False

        print("move was successful")
        # nodes, edges = self.get_node_names_and_edges()
        # print("new nodes:", nodes)
        # print("new edges:", edges)
        return True

    def mcmc(self):
        if (self.transform_tree()):
            self.update_latent_state()

        self.random_walk_latent_space()
        self.get_initial_decoder_state()

    def get_vector_representation(self):
        stack = []
        curr_node = self.curr_prog
        nodes_dfs = []
        edges_dfs = []

        while curr_node is not None:
            nodes_dfs.append(curr_node.api_num)
            if curr_node.child is not None:
                if curr_node.sibling is not None:
                    stack.append(curr_node.sibling)
                edges_dfs.append(CHILD_EDGE)
                curr_node = curr_node.child

            elif curr_node.sibling is not None:
                edges_dfs.append(SIBLING_EDGE)
                curr_node = curr_node.sibling
            else:
                if len(stack) > 0:
                    curr_node = stack.pop()
                else:
                    curr_node = None

        nodes = np.zeros([1, self.max_depth], dtype=np.int32)
        edges = np.zeros([1, self.max_depth], dtype=np.bool)
        nodes[0, :len(nodes_dfs)] = nodes_dfs
        edges[0, :len(edges_dfs)] = edges_dfs

        return nodes[0], edges[0]

    def get_node_names_and_edges(self):
        nodes, edges = self.get_vector_representation()
        nodes = [self.node2vocab[node] for node in nodes]
        return nodes, edges

    def calculate_probability(self):
        nodes, edges = self.get_vector_representation()
        node = np.zeros([self.config.batch_size, self.config.max_ast_depth], dtype=np.int32)
        edge = np.zeros([self.config.batch_size, self.config.max_ast_depth], dtype=np.bool)
        state = self.initial_state
        curr_prob = 0.0

        for i in range(self.curr_prog.length):
            node[0][0] = nodes[i]
            edge[0][0] = edges[i]
            state, ast_prob = self.get_next_ast_state_and_prob(node, edge, state)
            state = np.array(state)
            if i == self.curr_prog.length - 1:
                # add prob of stop node
                stop_node = self.vocab2node['DStop']
                curr_prob += ast_prob[0][stop_node]
            else:
                curr_prob += ast_prob[0][nodes[i + 1]]  # should I be taking this node or the next node?
            # if i == self.curr_prog.length - 1:
            #     stop_node = self.vocab2node['DStop']
            #     curr_prob += ast_prob[0][stop_node]

        # TODO: this makes sense right vs trying to calculate the probability of the whole tree. unsure how that would work tbh

        self.curr_log_prob = curr_prob

    # def infer(self):

# mcmc = MCMCProgram(model)
# constraints = ["java.lang.StringBuffer.StringBuffer()", "java.lang.StringBuffer.append(java.lang.String)", "java.io.BufferedReader.readLine()", "hi"]
# mcmc.create_program(constraints)
# mcmc.testing()

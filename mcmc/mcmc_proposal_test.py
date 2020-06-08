import math
import random
import unittest
# from node import Node
from utils import print_verbose_tree_info
# from tree_modifier import TreeModifier

import numpy as np
import tensorflow as tf

import unittest.mock as mock

from test_utils import STR_BUF, STR_APP, READ_LINE, CLOSE, STR_LEN, STR_BUILD, STR_BUILD_APP, create_base_program, \
    create_str_buf_base_program, create_eight_node_program, create_dbranch, create_dloop, create_dexcept, \
    create_all_dtypes_program, DBRANCH, DLOOP, DEXCEPT, SIBLING_EDGE, CHILD_EDGE

from mcmc import INSERT, DELETE, REPLACE, SWAP, ADD_DNODE

from proposals.insertion_proposals import ProposalWithInsertion
from proposals.insert_proposal import InsertProposal

# SAVED MODEL
SAVED_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/1k_vocab_constraint_min_3-600000'


class ProposalTests(unittest.TestCase):
    @mock.patch.object(random, 'randint')
    def test_insert_proposal(self, mock_randint):
        test_prog, _, _ = create_base_program(SAVED_MODEL_PATH, [STR_BUILD, STR_BUILD_APP],
                                              ["Typeface"],
                                              ["String", "int"])
        prog = test_prog.prog
        curr_prog = test_prog.prog.curr_prog
        prog.proposal_probs = {INSERT: 0.5, DELETE: 0.5, SWAP: 0.0, REPLACE: 0.0, ADD_DNODE: 0.0}
        expected_nodes, expected_edges = prog.tree_mod.get_vector_representation(curr_prog)
        print("prev program")
        print_verbose_tree_info(curr_prog)

        # Logging and checks
        prev_length = curr_prog.length
        print("prev length:", prev_length)

        # Add node
        for i in range(1, curr_prog.length):
            print("\ni:", i)
            mock_randint.return_value = i
            output = prog.Insert.add_random_node(curr_prog, prog.initial_state)

            if output is None:
                print("OUTPUT IS NONE")

            curr_prog, added_node, ln_proposal_prob = output
            ln_reversal_prob = prog.Delete.calculate_ln_prob_of_move()

            # Calculate probability of new program
            prog.calculate_probability()
            print("")
            print("probability of old program", math.exp(prog.prev_log_prob))
            print("probability of new program", math.exp(prog.curr_log_prob))
            print("")

            # Print logs
            print("new program")
            print_verbose_tree_info(curr_prog)

            # If no node was added, return False
            if added_node is None:
                self.assertEqual(curr_prog.length, prev_length, "Curr prog length: " + str(
                    curr_prog.length) + " != prev length: " + str(prev_length))

            # Validate current program
            valid = prog.validate_and_update_program(INSERT, ln_proposal_prob, ln_reversal_prob)

            print("valid:", valid)

            if valid:
                # Check that insertion was valid and that there aren't any bugs
                prog.check_insert(added_node, prev_length)

            # Undo move
            prog.Insert.undo_add_random_node(added_node)
            prog.curr_log_prob = prog.prev_log_prob
            print("after undo move:")
            print_verbose_tree_info(curr_prog)
            self.assertEqual(curr_prog.length, prev_length, "Curr prog length: " + str(
                curr_prog.length) + " != prev length: " + str(prev_length))
            nodes, edges = prog.tree_mod.get_vector_representation(curr_prog)
            self.assertListEqual(list(expected_nodes), list(nodes))
            self.assertListEqual(list(expected_edges), list(edges))

    def test_insert_proposal_for_dnode(self):

        test_prog, _, _ = create_base_program(SAVED_MODEL_PATH, [STR_BUILD, STR_BUILD_APP],
                                              ["Typeface"],
                                              ["String", "int"])
        prog = test_prog.prog
        curr_prog = test_prog.prog.curr_prog
        prog.proposal_probs = {INSERT: 0.5, DELETE: 0.5, SWAP: 0.0, REPLACE: 0.0, ADD_DNODE: 0.0}
        expected_nodes, expected_edges = prog.tree_mod.get_vector_representation(curr_prog)
        print("prev program")
        print_verbose_tree_info(curr_prog)

        # Logging and checks
        prev_length = curr_prog.length
        print("prev length:", prev_length)

        dnodes = [DBRANCH, DLOOP, DEXCEPT]

        # prog.Insert = ProposalWithInsertion(prog.tree_mod, prog.decoder)

        prog.Insert.curr_prog = curr_prog
        prog.Insert.initial_state = prog.initial_state

        insertion_class = ProposalWithInsertion(prog.tree_mod, prog.decoder)
        insertion_class.curr_prog = curr_prog
        insertion_class.initial_state = prog.initial_state

        # Add node
        for i in range(1, curr_prog.length):
            for DNODE in dnodes:
                print("\ni:", i)
                parent = prog.tree_mod.get_node_in_position(curr_prog, i)
                parent_sibling = parent.sibling
                parent.remove_node(SIBLING_EDGE)
                dnode = prog.tree_mod.create_and_add_node(DNODE, parent, SIBLING_EDGE)
                dnode.add_node(parent_sibling, SIBLING_EDGE)
                self.assertIsNotNone(dnode)

                if DNODE == DBRANCH:
                    insertion_class._grow_dbranch(dnode, verbose=True)
                elif DNODE == DLOOP:
                    insertion_class._grow_dloop_or_dexcept(dnode, verbose=True)
                else:
                    insertion_class._grow_dloop_or_dexcept(dnode, verbose=True)

                # Calculate probability of new program
                prog.calculate_probability()
                print("")
                print("probability of old program", math.exp(prog.prev_log_prob))
                print("probability of new program", math.exp(prog.curr_log_prob))
                print("")

                # Print logs
                print("new program")
                print_verbose_tree_info(curr_prog)

                # Validate current program
                valid = prog.validate_and_update_program(INSERT, -5, -5)

                print("valid:", valid)

                if valid:
                    # Check that insertion was valid and that there aren't any bugs
                    prog.check_insert(dnode, prev_length)

                # Undo move
                prog.Insert.undo_add_random_node(dnode)
                prog.curr_log_prob = prog.prev_log_prob
                print("after undo move:")
                print_verbose_tree_info(curr_prog)
                self.assertEqual(curr_prog.length, prev_length, "Curr prog length: " + str(
                    curr_prog.length) + " != prev length: " + str(prev_length))
                nodes, edges = prog.tree_mod.get_vector_representation(curr_prog)
                self.assertListEqual(list(expected_nodes), list(nodes))
                self.assertListEqual(list(expected_edges), list(edges))

    @mock.patch.object(random, 'randint')
    def test_delete_proposal(self, mock_randint):
        test_prog, _, _ = create_base_program(SAVED_MODEL_PATH, [STR_BUILD, STR_BUILD_APP],
                                              ["Typeface"],
                                              ["String", "int"])
        prog = test_prog.prog
        curr_prog = test_prog.prog.curr_prog
        prog.proposal_probs = {INSERT: 0.5, DELETE: 0.5, SWAP: 0.0, REPLACE: 0.0, ADD_DNODE: 0.0}
        expected_nodes, expected_edges = prog.tree_mod.get_vector_representation(curr_prog)
        print("prev program")
        print_verbose_tree_info(curr_prog)

        # Logging and checks
        prev_length = curr_prog.length
        print("prev length:", prev_length)

        # Add node
        for i in range(1, curr_prog.length):
            print("\ni:", i)
            mock_randint.return_value = i
            curr_prog, node, parent_node, parent_edge, ln_prob = prog.Delete.delete_random_node(curr_prog)

            parent_pos = prog.tree_mod.get_nodes_position(curr_prog, parent_node)

            curr_prog_copy = curr_prog.copy()
            parent_node_copy = prog.tree_mod.get_node_in_position(curr_prog_copy, parent_pos)
            parent_node_copy_neighbor = parent_node_copy.get_neighbor(parent_edge)
            parent_node_copy.add_node(node, parent_edge)
            node.add_node(parent_node_copy_neighbor, parent_edge)
            node_pos = prog.tree_mod.get_nodes_position(curr_prog_copy, node)

            nodes_copy, edges_copy = prog.tree_mod.get_vector_representation(curr_prog_copy)
            self.assertListEqual(list(expected_nodes), list(nodes_copy))
            self.assertListEqual(list(expected_edges), list(edges_copy))

            ln_reversal_prob = prog.Insert.calculate_ln_prob_of_move(curr_prog_copy, prog.initial_state, node_pos,
                                                                     parent_edge, is_copy=True)
            parent_node_copy.remove_node(parent_edge)

            # Calculate probability of new program
            prog.calculate_probability()
            print("")
            print("probability of old program", math.exp(prog.prev_log_prob))
            print("probability of new program", math.exp(prog.curr_log_prob))
            print("")

            # Print logs
            print("new program")
            print_verbose_tree_info(curr_prog)

            valid = prog.validate_and_update_program(DELETE, ln_prob, ln_reversal_prob)

            print("valid:", valid)

            # Undo move if not valid
            if not valid:
                prog.Delete.undo_delete_random_node(node, parent_node, parent_edge)
                prog.curr_log_prob = prog.prev_log_prob
                print("after undo move:")
                print_verbose_tree_info(curr_prog)
                self.assertEqual(curr_prog.length, prev_length, "Curr prog length: " + str(
                    curr_prog.length) + " != prev length: " + str(prev_length))
                nodes, edges = prog.tree_mod.get_vector_representation(curr_prog)
                self.assertListEqual(list(expected_nodes), list(nodes))
                self.assertListEqual(list(expected_edges), list(edges))
            else:
                # Check that insertion was valid and that there aren't any bugs
                prog.check_delete(node, prev_length)

    @mock.patch.object(random, 'randint')
    # @mock.patch.object(ProposalWithInsertion, 'get_ast_idx')
    def test_insert_delete_calculations(self, mock_randint):
        # Insert Proposal
        test_prog, _, _ = create_base_program(SAVED_MODEL_PATH, [STR_BUILD, STR_BUILD_APP],
                                              ["Typeface"],
                                              ["String", "int"])
        pos = 1
        mock_randint.return_value = pos
        prog = test_prog.prog
        curr_prog = test_prog.prog.curr_prog
        prog.proposal_probs = {INSERT: 0.5, DELETE: 0.5, SWAP: 0.0, REPLACE: 0.0, ADD_DNODE: 0.0}
        curr_prog, added_node, ln_proposal_prob = prog.Insert.add_random_node(curr_prog, prog.initial_state)

        print("\ncurr prog from insert proposal:")
        print_verbose_tree_info(curr_prog)
        nodes, edges = prog.tree_mod.get_vector_representation(curr_prog)

        # Calculate move of same move done in insert proposal
        new_test_prog, _, _ = create_base_program(SAVED_MODEL_PATH, [STR_BUILD, STR_BUILD_APP],
                                                  ["Typeface"],
                                                  ["String", "int"])
        new_prog = new_test_prog.prog
        new_curr_prog = new_prog.curr_prog
        parent_node = new_prog.tree_mod.get_node_in_position(new_curr_prog, pos)
        new_added_node = new_prog.tree_mod.create_and_add_node(added_node.api_name, parent_node,
                                                               added_node.parent_edge, save_neighbors=True)
        print("\nmanual curr prog:")
        print_verbose_tree_info(new_curr_prog)
        new_nodes, new_edges = new_prog.tree_mod.get_vector_representation(new_curr_prog)
        self.assertListEqual(list(nodes), list(new_nodes))
        self.assertListEqual(list(edges), list(new_edges))

        new_added_node_pos = new_prog.tree_mod.get_nodes_position(new_curr_prog, new_added_node)
        print("\nnode pos:", new_added_node_pos)
        new_prog.Insert.curr_prog = new_curr_prog
        new_prog.Insert.initial_state = prog.initial_state
        new_ln_prob = new_prog.Insert.calculate_ln_prob_of_move(new_curr_prog, prog.initial_state,
                                                                new_added_node_pos,
                                                                new_added_node.parent_edge, is_copy=True)
        print("\nprob from insert proposal:", ln_proposal_prob)
        print("prob from calc move:", new_ln_prob)
        self.assertEqual(ln_proposal_prob, new_ln_prob)

    @mock.patch.object(random, 'randint')
    def test_replace_proposal(self, mock_randint):
        test_prog, _, _ = create_base_program(SAVED_MODEL_PATH, [STR_BUILD, STR_LEN],
                                              ["Typeface"],
                                              ["String", "int"])
        prog = test_prog.prog
        curr_prog = test_prog.prog.curr_prog
        expected_nodes, expected_edges = prog.tree_mod.get_vector_representation(curr_prog)
        prog.proposal_probs = {INSERT: 0.0, DELETE: 0.0, SWAP: 0.0, REPLACE: 1.0, ADD_DNODE: 0.0}
        print("prev program")
        print_verbose_tree_info(curr_prog)

        # Logging and checks
        prev_length = curr_prog.length
        print("prev length:", prev_length)

        # Add node
        for i in range(1, curr_prog.length):
            print("\ni:", i)
            mock_randint.return_value = i
            curr_prog, new_node, replaced_node_api, ln_proposal_prob = \
                prog.Replace.replace_random_node(curr_prog, prog.initial_state)

            # If no node was added, return False
            if new_node is None:
                self.assertEqual(curr_prog.length, prev_length, "Curr prog length: " + str(
                    curr_prog.length) + " != prev length: " + str(prev_length))
                print("NEW NODE IS NONE")
                continue

            # Print logs
            print("\nnew curr program:")
            print_verbose_tree_info(curr_prog)

            # Calculate reversal prob
            new_node_pos = prog.tree_mod.get_nodes_position(curr_prog, new_node)
            ln_reversal_prob = prog.Replace.calculate_reversal_ln_prob(curr_prog, prog.initial_state, new_node_pos,
                                                                       replaced_node_api, new_node.parent_edge)

            # Calculate probability of new program
            prog.calculate_probability()

            print("\nprev prob:", prog.prev_log_prob)
            print("curr prob:", prog.curr_log_prob)
            print("replace prob:", ln_proposal_prob)
            print("reversal replace prob:", ln_reversal_prob)

            self.assertEqual(prog.prev_log_prob, ln_reversal_prob)
            self.assertEqual(prog.curr_log_prob, ln_proposal_prob)

            # Validate current program
            valid = prog.validate_and_update_program(REPLACE, ln_proposal_prob, ln_reversal_prob)

            print("\nvalid:", valid)

            # Undo move if not valid
            prog.Replace.undo_replace_random_node(new_node, replaced_node_api)
            prog.curr_log_prob = prog.prev_log_prob
            print("after undo move:")
            print_verbose_tree_info(curr_prog)
            self.assertEqual(curr_prog.length, prev_length, "Curr prog length: " + str(
                curr_prog.length) + " != prev length: " + str(prev_length))
            nodes, edges = prog.tree_mod.get_vector_representation(curr_prog)
            self.assertListEqual(list(expected_nodes), list(nodes))
            self.assertListEqual(list(expected_edges), list(edges))

    def test_swap_proposal(self):
        pass

    def test_logits(self):
        test_prog, _, _ = create_base_program(SAVED_MODEL_PATH, [STR_BUILD, STR_BUILD_APP],
                                              ["Typeface"],
                                              ["String", "int"])
        prog = test_prog.prog
        curr_prog = test_prog.prog.curr_prog
        expected_nodes, expected_edges = prog.tree_mod.get_vector_representation(curr_prog)
        print("prev program")
        print_verbose_tree_info(curr_prog)

        added_node_pos = 2
        prog.Insert.curr_prog = curr_prog
        prog.Insert.initial_state = prog.initial_state
        added_node = prog.tree_mod.get_node_in_position(curr_prog, added_node_pos)

        logits = prog.Insert._get_logits(curr_prog, prog.initial_state, added_node_pos, added_node, SIBLING_EDGE)
        logits = logits.reshape(1, logits.shape[0])
        print(type(logits))
        print(logits.shape)
        # # chosen_idx = prog.sess.run([tf.multinomial(logits, 1)], {logits: logits})
        # logits = logits.reshape(1, logits.shape[0])
        # vals, idxs = tf.math.top_k(logits, k=prog.decoder.top_k)
        #
        # # print(logits.shape)
        # print(type(vals))
        # chosen_idx = prog.sess.run(tf.multinomial(vals, 1), {logits: logits})
        # print(chosen_idx)
        # print(prog.config.node2vocab[chosen_idx])

        nodes, edges = prog.tree_mod.get_vector_representation(curr_prog)
        node = np.zeros([1, 1], dtype=np.int32)
        edge = np.zeros([1, 1], dtype=np.bool)
        node[0][0] = nodes[0]
        edge[0][0] = edges[0]
        # feed = {prog.decoder.edges.name: edge, prog.decoder.nodes.name: node,
        #         prog.decoder.return_type: ["Typeface"],
        #         prog.decoder.formal_params: ["String", "int"]}

        feed = {prog.decoder.model.edges.name: edge, prog.decoder.model.nodes.name: node,
                prog.decoder.model.return_type: [prog.config.rettype2num["Typeface"]],
                prog.decoder.model.formal_params: [[prog.config.fp2num["int"]]]}

        ast_logits = prog.sess.run(prog.decoder.model.decoder.ast_logits[0], feed)
        print(ast_logits)
        print(type(ast_logits))
        print(ast_logits.shape)

        vals, idxs = tf.math.top_k(logits, k=prog.decoder.top_k)
        # vals, idxs, chosen_idx = prog.sess.run([vals, idxs, idxs[0][tf.multinomial(vals, 1)[0][0]]], feed)
        vals, idxs, chosen_idx = prog.sess.run([vals, idxs, idxs[0][tf.multinomial(vals, 1)[0][0]]], {})

        print(vals)
        print(type(vals))
        print(idxs)
        print(type(idxs))
        print(chosen_idx)
        print(prog.config.node2vocab[chosen_idx])

        print("\nmultinomial:")
        idx = prog.sess.run(tf.multinomial(logits, 1)[0][0], {})
        print(idx)
        print(prog.config.node2vocab[idx])

        logs = logits.tolist()[0]
        logs = [math.exp(i) for i in logs]
        print(logs)
        print(sum(logs))

        norm_logs = prog.sess.run(tf.nn.log_softmax(logits[0]), {})
        norm_logs = [math.exp(i) for i in norm_logs]
        print(norm_logs)
        print(norm_logs[idx])
        print(sum(norm_logs))

        # TODO: need to figure out if this is how I should sample from the multinomial or not. My logits are sums of
        #  normalized logits. The sum in unnormalized and hence can be used by tf.multinomial. Need to make sure the
        #  line below is whats happening in tf.multinomial, which I think is the case.
        print("np multinomial:", np.argmax(np.random.multinomial(1, norm_logs, size=1)))


if __name__ == '__main__':
    unittest.main()

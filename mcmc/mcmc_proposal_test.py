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

from mcmc import INSERT, DELETE, REPLACE

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
        new_prog.Insert.initial_state = new_prog.initial_state
        new_ln_prob = new_prog.Insert.calculate_ln_prob_of_move(new_curr_prog, new_prog.initial_state,
                                                                new_added_node_pos,
                                                                new_added_node.parent_edge, is_copy=True)
        print("\nprob from insert proposal:", ln_proposal_prob)
        print("prob from calc move:", new_ln_prob)
        self.assertEqual(ln_proposal_prob, new_ln_prob)


if __name__ == '__main__':
    unittest.main()

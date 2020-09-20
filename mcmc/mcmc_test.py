import argparse
import math
import datetime
import os
import random
import sys
import unittest

from ast_helper.beam_searcher.program_beam_searcher import ProgramBeamSearcher
from data_extractor.data_loader import Loader
from mcmc import Node, MCMCProgram, SIBLING_EDGE, CHILD_EDGE, START, STOP, DBRANCH, DLOOP, DEXCEPT
from utils import print_verbose_tree_info
from trainer_vae.infer import BayesianPredictor
from trainer_vae.model import Model

import numpy as np
import tensorflow as tf

from test_utils import STR_BUF, STR_APP, READ_LINE, CLOSE, STR_LEN, STR_BUILD, STR_BUILD_APP, create_base_program, \
    create_str_buf_base_program, create_eight_node_program, create_dbranch, create_dloop, create_dexcept, \
    create_all_dtypes_program

from mcmc import INSERT, DELETE, REPLACE, SWAP, ADD_DNODE, GROW_CONST

from test_suite import MOST_COMMON_APIS, MID_COMMON_APIS, UNCOMMON_APIS, MID_COMMON_DISJOINT_PAIRS, \
    MOST_COMMON_DISJOINT_PAIRS, UNCOMMON_DISJOINT_PAIRS

from data_extractor.graph_analyzer import GraphAnalyzer

TOP = 'top'
MID = 'mid'
LOW = 'low'

import unittest.mock as mock

# SAVED MODEL
SAVED_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/1k_vocab_constraint_min_3-600000'

ALL_DATA_1K_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_1k_vocab'

ALL_DATA_TRAINING_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_training_data_1.38m_large_config'

ALL_DATA_1K_05_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_1k_vocab_0.5_KL_beta'

ALL_DATA_1K_025_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_1k_vocab_0.25_KL_beta'


class MCMCProgramTest(unittest.TestCase):

    def test_init(self):
        # Test basic program
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, len(expected_nodes) - 1)

        # test_prog.prog.calculate_probability()
        # print(test_prog.prog.curr_log_prob)

        # Test 8 node program
        test_prog, expected_nodes, expected_edges = create_eight_node_program(SAVED_MODEL_PATH)
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, len(expected_nodes) - 1)

        # test_prog.prog.calculate_probability()
        # print(test_prog.prog.curr_log_prob)

        # Test program with DBranch
        test_prog, _, _ = create_str_buf_base_program()
        test_prog, _ = create_dbranch(test_prog)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DBRANCH, STR_BUF, STR_APP, STOP, READ_LINE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 4)

        # Test program with DLoop
        test_prog, _, _ = create_str_buf_base_program()
        test_prog, _ = create_dloop(test_prog)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DLOOP, READ_LINE, CLOSE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 3)

        # Test program with DExcept
        test_prog, _, _ = create_str_buf_base_program()
        test_prog, _ = create_dexcept(test_prog)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DEXCEPT, STR_BUF, CLOSE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 3)

        # Test program will all dnodes
        test_prog, expected_nodes, expected_edges = create_all_dtypes_program()
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 8)

    def test_create_and_add_node(self):
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program()

        # Test adding sibling node
        test_prog.add_to_first_available_node(STR_BUF, SIBLING_EDGE)
        expected_nodes = [START, STR_BUF, STR_BUF]
        expected_edges = [False, False]

        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 3)

        # Test adding child node
        test_prog.add_to_first_available_node(STR_APP, CHILD_EDGE)
        expected_nodes = [START, STR_APP, STR_BUF, STR_BUF]
        expected_edges = [True, False, False]

        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 4)

        # Test adding invalid node
        test_prog.add_to_first_available_node('abc', CHILD_EDGE)

        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 4)

    def test_get_node_in_position(self):
        test_prog, expected_nodes, expected_edges = create_eight_node_program(SAVED_MODEL_PATH)

        self.assertEqual(START, test_prog.prog.get_node_in_position(0).api_name)
        self.assertEqual(STR_APP, test_prog.prog.get_node_in_position(1).api_name)
        self.assertEqual(READ_LINE, test_prog.prog.get_node_in_position(2).api_name)
        self.assertEqual(STR_BUF, test_prog.prog.get_node_in_position(3).api_name)
        self.assertEqual(READ_LINE, test_prog.prog.get_node_in_position(4).api_name)
        self.assertEqual(STR_APP, test_prog.prog.get_node_in_position(5).api_name)
        self.assertEqual(STR_BUF, test_prog.prog.get_node_in_position(6).api_name)
        self.assertEqual(STR_BUF, test_prog.prog.get_node_in_position(7).api_name)

    def test_get_vector_representation(self):
        # Test non-branched program
        test_prog, expected_nodes, expected_edges = create_eight_node_program(SAVED_MODEL_PATH)
        node_nums = [test_prog.vocab2node[i] for i in expected_nodes]
        exp_node_nums = np.zeros([1, test_prog.prog.max_length], dtype=np.int32)
        exp_edges = np.zeros([1, test_prog.prog.max_length], dtype=np.bool)
        exp_node_nums[0, :len(node_nums)] = node_nums
        exp_edges[0, :len(expected_edges)] = expected_edges
        exp_node_nums = exp_node_nums[0].tolist()
        exp_edges = exp_edges[0].tolist()

        nodes, edges = test_prog.prog.get_vector_representation()

        self.assertListEqual(nodes.tolist(), exp_node_nums, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(edges.tolist(), exp_edges, "Edges must be equal to expected nodes in program.")

        # Test all dnodes types program
        test_prog, expected_nodes, expected_edges = create_all_dtypes_program(SAVED_MODEL_PATH)
        node_nums = [test_prog.vocab2node[i] for i in expected_nodes]
        exp_node_nums = np.zeros([1, test_prog.prog.max_length], dtype=np.int32)
        exp_edges = np.zeros([1, test_prog.prog.max_length], dtype=np.bool)
        exp_node_nums[0, :len(node_nums)] = node_nums
        exp_edges[0, :len(expected_edges)] = expected_edges
        exp_node_nums = exp_node_nums[0].tolist()
        exp_edges = exp_edges[0].tolist()

        nodes, edges = test_prog.prog.get_vector_representation()

        self.assertListEqual(nodes.tolist(), exp_node_nums, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(edges.tolist(), exp_edges, "Edges must be equal to expected nodes in program.")

    # def test_validity(self):


    @mock.patch.object(MCMCProgram, 'get_ast_idx')
    @mock.patch.object(random, 'randint')
    def test_add_and_undo_random_node(self, mock_randint, mock_get_ast_idx):
        test_prog, expected_nodes, expected_edges = create_eight_node_program(SAVED_MODEL_PATH)
        mock_get_ast_idx.return_value = test_prog.vocab2node[CLOSE]
        mock_randint.return_value = 5

        test_prog.prog.max_num_api = 11

        # add a random node 1
        new_node = test_prog.prog.add_random_node()
        test_prog.update_nodes_and_edges()
        self.assertEqual(new_node.api_name, CLOSE)

        new_expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_APP, CLOSE, STR_BUF, STR_BUF]
        new_expected_edges = [True, True, True, False, False, False, False, False]

        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 9)

        # add random node 2
        mock_randint.return_value = 8
        new_node2 = test_prog.prog.add_random_node()
        test_prog.update_nodes_and_edges()
        self.assertEqual(new_node2.api_name, CLOSE)

        new_expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_APP, CLOSE, STR_BUF, STR_BUF, CLOSE]
        new_expected_edges = [True, True, True, False, False, False, False, False, False]

        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 10)

        # remove first random node
        new_expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_APP, STR_BUF, STR_BUF, CLOSE]
        new_expected_edges = [True, True, True, False, False, False, False, False]

        test_prog.prog.undo_add_random_node(new_node)
        test_prog.update_nodes_and_edges()
        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 9)

        # remove second random node
        test_prog.prog.undo_add_random_node(new_node2)
        test_prog.update_nodes_and_edges()
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 8)

        # test that if max depth is exceeded no node will be added
        test_prog.prog.max_num_api = 7
        mock_randint.return_value = 4
        new_node = test_prog.prog.add_random_node()
        test_prog.update_nodes_and_edges()
        self.assertEqual(new_node, None)
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 8)

        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        mock_randint.return_value = 1
        mock_get_ast_idx.return_value = test_prog.vocab2node[DBRANCH]
        dbranch = test_prog.prog.add_random_node()
        test_prog.prog.undo_add_random_node(dbranch)
        test_prog.update_nodes_and_edges()
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 2)

    @mock.patch.object(random, 'randint')
    def test_delete_and_undo_random_node(self, mock_randint):
        test_prog, expected_nodes, expected_edges = create_eight_node_program(SAVED_MODEL_PATH)
        mock_randint.return_value = 5

        # test delete a node
        node, parent_node, parent_edge = test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertEqual(node.api_name, STR_APP)
        self.assertEqual(parent_node.api_name, STR_APP)
        self.assertEqual(parent_edge, SIBLING_EDGE)
        new_expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_BUF, STR_BUF]
        new_expected_edges = [True, True, True, False, False, False]
        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 7)

        # test delete node with replaced sibling
        node2, parent_node2, parent_edge2 = test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertEqual(node2.api_name, STR_BUF)
        self.assertEqual(parent_node2.api_name, START)
        self.assertEqual(parent_edge2, SIBLING_EDGE)
        self.assertEqual(parent_node2.sibling.api_name, STR_BUF)
        self.assertEqual(parent_node2.sibling.sibling, None)
        new_expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_BUF]
        new_expected_edges = [True, True, True, False, False]
        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 6)

        # test undo delete node
        test_prog.prog.undo_delete_random_node(node, parent_node, parent_edge)
        test_prog.update_nodes_and_edges()
        new_expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_APP, STR_BUF]
        new_expected_edges = [True, True, True, False, False, False]
        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 7)

        # test removing entire child subtree
        mock_randint.return_value = 1
        node3, parent_node3, parent_edge3 = test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        new_expected_nodes_3 = [START, STR_BUF]
        new_expected_edges_3 = [SIBLING_EDGE]
        self.assertListEqual(test_prog.nodes, new_expected_nodes_3, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges_3, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 2)
        test_prog.prog.undo_delete_random_node(node3, parent_node3, parent_edge3)
        test_prog.update_nodes_and_edges()
        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 7)

    def test_swap_nodes(self):
        test_prog, expected_nodes, expected_edges = create_eight_node_program(SAVED_MODEL_PATH)
        node1 = test_prog.prog.get_node_in_position(1)
        node2 = test_prog.prog.get_node_in_position(6)
        self.assertEqual(node1.api_name, STR_APP)
        self.assertEqual(node2.api_name, STR_BUF)

        test_prog.prog.swap_nodes(node1, node2)
        test_prog.update_nodes_and_edges()
        swapped_expected_nodes = [START, STR_BUF, READ_LINE, STR_BUF, READ_LINE, STR_APP, STR_APP, STR_BUF]
        self.assertListEqual(test_prog.nodes, swapped_expected_nodes,
                             "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 8)

        # test swapping nodes again to get original tree back
        test_prog.prog.swap_nodes(node1, node2)
        test_prog.update_nodes_and_edges()
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 8)

        # test switching a parent and child node
        node2 = test_prog.prog.get_node_in_position(2)
        swapped_expected_nodes = [START, READ_LINE, STR_APP, STR_BUF, READ_LINE, STR_APP, STR_BUF, STR_BUF]
        test_prog.prog.swap_nodes(node1, node2)
        test_prog.update_nodes_and_edges()
        self.assertListEqual(test_prog.nodes, swapped_expected_nodes,
                             "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 8)

        # test switching parent and child back to get the original tree
        test_prog.prog.swap_nodes(node1, node2)
        test_prog.update_nodes_and_edges()
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 8)

    @mock.patch.object(random, 'randint')
    def test_check_validity(self, mock_randint):
        test_prog, expected_nodes, expected_edges = create_eight_node_program(SAVED_MODEL_PATH)
        self.assertTrue(test_prog.prog.check_validity())

        test_prog.prog.add_constraint(STR_APP)
        self.assertTrue(test_prog.prog.check_validity())
        test_prog.prog.add_constraint(STR_APP)
        self.assertTrue(test_prog.prog.check_validity())

        # Test invalid node
        test_prog.prog.add_constraint('abc')
        self.assertTrue(test_prog.prog.check_validity())

        # Fail case
        test_prog.prog.add_constraint(STR_APP)
        self.assertFalse(test_prog.prog.check_validity())

        # All branches valid case
        test_prog, expected_nodes, expected_edges = create_all_dtypes_program(SAVED_MODEL_PATH)
        self.assertTrue(test_prog.prog.check_validity())

        test_prog, expected_nodes, expected_edges = create_all_dtypes_program(SAVED_MODEL_PATH)
        self.assertTrue(test_prog.prog.check_validity())

        mock_randint.return_value = 12
        test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertTrue(test_prog.prog.check_validity())
        self.assertEqual(test_prog.prog.curr_prog.length, 11)

        mock_randint.return_value = 8
        test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertTrue(test_prog.prog.check_validity())
        self.assertEqual(test_prog.prog.curr_prog.length, 7)

        mock_randint.return_value = 2
        test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertTrue(test_prog.prog.check_validity())

        mock_randint.return_value = 1
        test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertFalse(test_prog.prog.check_validity())

        # Test DBranch fail
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        dbranch = test_prog.add_to_first_available_node(DBRANCH, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        cond = test_prog.prog.create_and_add_node(DLOOP, dbranch, CHILD_EDGE)
        then = test_prog.prog.create_and_add_node(STR_APP, cond, CHILD_EDGE)
        else_node = test_prog.prog.create_and_add_node(STR_APP, cond, SIBLING_EDGE)
        test_prog.prog.create_and_add_node(STOP, then, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())

        # Test DBranch
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        dbranch = test_prog.add_to_first_available_node(DBRANCH, SIBLING_EDGE)
        cond = test_prog.prog.create_and_add_node(STR_BUF, dbranch, CHILD_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        then = test_prog.prog.create_and_add_node(STR_APP, cond, CHILD_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        else_node = test_prog.prog.create_and_add_node(STR_APP, cond, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        test_prog.prog.create_and_add_node(STOP, then, SIBLING_EDGE)
        self.assertTrue(test_prog.prog.check_validity())

        # Test DLoop fail
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        dloop = test_prog.add_to_first_available_node(DLOOP, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        cond = test_prog.prog.create_and_add_node(DEXCEPT, dloop, CHILD_EDGE)
        body = test_prog.prog.create_and_add_node(STR_APP, cond, CHILD_EDGE)
        test_prog.prog.create_and_add_node(STOP, body, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())

        # Test DLoop
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        dloop = test_prog.add_to_first_available_node(DLOOP, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        cond = test_prog.prog.create_and_add_node(STR_BUF, dloop, CHILD_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        body = test_prog.prog.create_and_add_node(STR_APP, cond, CHILD_EDGE)
        test_prog.prog.create_and_add_node(STOP, body, SIBLING_EDGE)
        self.assertTrue(test_prog.prog.check_validity())

        # Test DExcept fail
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        dexcept = test_prog.add_to_first_available_node(DEXCEPT, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        cond = test_prog.prog.create_and_add_node(DBRANCH, dexcept, CHILD_EDGE)
        body = test_prog.prog.create_and_add_node(STR_APP, cond, CHILD_EDGE)
        test_prog.prog.create_and_add_node(STOP, body, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())

        # Test DExcept
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        dexcept = test_prog.add_to_first_available_node(DEXCEPT, SIBLING_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        cond = test_prog.prog.create_and_add_node(STR_BUF, dexcept, CHILD_EDGE)
        self.assertFalse(test_prog.prog.check_validity())
        body = test_prog.prog.create_and_add_node(STR_APP, cond, CHILD_EDGE)
        test_prog.prog.create_and_add_node(STOP, body, SIBLING_EDGE)
        self.assertTrue(test_prog.prog.check_validity())


    @mock.patch.object(MCMCProgram, 'get_ast_idx')
    @mock.patch.object(random, 'randint')
    def test_add_and_swap_node(self, mock_randint, mock_get_ast_idx):
        test_prog, expected_nodes, expected_edges = create_eight_node_program(SAVED_MODEL_PATH)
        mock_get_ast_idx.return_value = test_prog.vocab2node[CLOSE]
        mock_randint.return_value = 5

        test_prog.prog.max_num_api = 11

        # add a random node 1
        new_node = test_prog.prog.add_random_node()
        test_prog.update_nodes_and_edges()
        self.assertEqual(new_node.api_name, CLOSE)
        self.assertEqual(new_node.parent.api_name, STR_APP)

        new_expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_APP, CLOSE, STR_BUF, STR_BUF]
        new_expected_edges = [True, True, True, False, False, False, False, False]

        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 9)

        # get newly added node and its parent and swap
        node1 = test_prog.prog.get_node_in_position(6)
        node2 = test_prog.prog.get_node_in_position(5)
        self.assertEqual(node1.api_name, CLOSE)
        self.assertEqual(node2.api_name, STR_APP)
        test_prog.prog.swap_nodes(node1, node2)
        test_prog.update_nodes_and_edges()

        new_expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, CLOSE, STR_APP, STR_BUF, STR_BUF]
        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 9)

        # get newly added node and some other node and swap
        node1 = test_prog.prog.get_node_in_position(5)
        node2 = test_prog.prog.get_node_in_position(2)
        self.assertEqual(node1.api_name, CLOSE)
        self.assertEqual(node2.api_name, READ_LINE)
        test_prog.prog.swap_nodes(node1, node2)
        test_prog.update_nodes_and_edges()

        new_expected_nodes = [START, STR_APP, CLOSE, STR_BUF, READ_LINE, READ_LINE, STR_APP, STR_BUF, STR_BUF]
        self.assertListEqual(test_prog.nodes, new_expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, new_expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 9)

    def test_get_valid_random_node(self):
        pass

    # @mock.patch.object(MCMCProgram, 'get_valid_random_node')
    @mock.patch.object(MCMCProgram, 'get_ast_idx')
    @mock.patch.object(random, 'choice')
    def test_add_dnode_node(self, mock_rand_choice, mock_get_ast_idx):
        test_prog, _, _ = create_str_buf_base_program(SAVED_MODEL_PATH)
        test_prog, dbranch = create_dbranch(test_prog)
        test_prog, dloop = create_dloop(test_prog, parent=dbranch)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DBRANCH, STR_BUF, STR_APP, STOP, READ_LINE, STOP, DLOOP, READ_LINE, CLOSE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE, SIBLING_EDGE,
                          SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 6)

        test_prog.prog.undo_add_random_dnode(dbranch)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DLOOP, READ_LINE, CLOSE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 3)

        test_prog.prog.undo_add_random_dnode(dloop)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF]
        expected_edges = [SIBLING_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 1)

        # Test adding and undoing a DBranch
        mock_get_ast_idx.return_value = test_prog.vocab2node[READ_LINE]
        mock_rand_choice.return_value = 'DBranch'
        dbranch = test_prog.prog.add_random_dnode()
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DBRANCH, READ_LINE, READ_LINE, STOP, READ_LINE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 4)
        test_prog.prog.undo_add_random_dnode(dbranch)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF]
        expected_edges = [SIBLING_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 1)

        # Test adding and undoing a DLoop
        mock_rand_choice.return_value = 'DLoop'
        dloop = test_prog.prog.add_random_dnode()
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DLOOP, READ_LINE, READ_LINE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 3)
        test_prog.prog.undo_add_random_dnode(dloop)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF]
        expected_edges = [SIBLING_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 1)

        # Test adding and undoing a DExcept
        mock_rand_choice.return_value = 'DExcept'
        dexcept = test_prog.prog.add_random_dnode()
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DEXCEPT, READ_LINE, READ_LINE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 3)
        test_prog.prog.undo_add_random_dnode(dexcept)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF]
        expected_edges = [SIBLING_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, len(expected_nodes))
        self.assertEqual(test_prog.prog.curr_prog.non_dnode_length, 1)

    def test_nodes_edges_targets(self):
        test_prog, expected_nodes, expected_edges = create_str_buf_base_program(SAVED_MODEL_PATH)
        create_dbranch(test_prog)
        nodes, edges, targets = test_prog.prog.tree_mod.get_nodes_edges_targets(test_prog.prog.curr_prog)
        print([test_prog.prog.config.node2vocab[i] for i in nodes[:8]])
        print(edges[:8])
        print([test_prog.prog.config.node2vocab[i] for i in targets[:8]])
        print(nodes)
        print(edges)
        print(targets)

        # self.assertListEqual(expected_nodes, nodes)
        # self.assertListEqual(expected_edges, edges)

    def test_mcmc(self):
        # test_prog, expected_nodes, expected_edges = create_base_program(SAVED_MODEL_PATH,
        #                                                                 ['java.lang.StringBuilder.StringBuilder()',
        #                                                                  'java.util.Map<java.lang.String,java.lang.String>.entrySet()'],
        #                                                                 ["String"],
        #                                                                 ['DSubTree', 'Map<String,String>', 'String'])

        # test_prog, expected_nodes, expected_edges = create_base_program(SAVED_MODEL_PATH,
        #                                                                 [STR_BUF],
        #                                                                 ['String'],
        #                                                                 ['DSubTree', 'String'])

        # test_prog, expected_nodes, expected_edges = create_base_program(ALL_DATA_1K_05_MODEL_PATH,
        #                                                                 ['java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()',
        #                                                                  'java.lang.StringBuilder.append(long)'],
        #                                                                 ['__UDT__'],
        #                                                                 ['DSubTree', 'String'])

        # test_prog, expected_nodes, expected_edges = create_base_program(ALL_DATA_1K_05_MODEL_PATH,
        #                                                                 ['java.io.FileInputStream.read(byte[])',
        #                                                                  'java.nio.ByteBuffer.getInt(int)',
        #                                                                  'java.lang.String.format(java.lang.String,java.lang.Object[])'],
        #                                                                 ['void'],
        #                                                                 ['DSubTree', 'String'], ordered=True)

        test_prog, expected_nodes, expected_edges = create_base_program(ALL_DATA_1K_05_MODEL_PATH,
                                                                        ['java.util.Vector<Tau_E>.Vector(int)', 'DLoop'],
                                                                        ['__UDT__'],
                                                                        ['DSubTree', 'Element', '__UDT__', 'boolean'], ordered=True, exclude=['DBranch'])



        test_prog.prog.verbose = True

        # test_prog.prog.proposal_probs = {INSERT: 0.05, DELETE: 0.05, SWAP: 0.0, REPLACE: 0.0, ADD_DNODE: 0.0, GROW_CONST: 0.9}

        # test_prog.add_to_first_available_node('java.awt.image.BufferedImage.getWidth(java.awt.image.ImageObserver)', SIBLING_EDGE)
        # test_prog.add_to_first_available_node('java.util.logging.Logger.setResourceBundle(java.util.logging.LogRecord)', SIBLING_EDGE)
        # test_prog.add_to_first_available_node('java.util.ArrayList<E>.ensureCapacity(int)', SIBLING_EDGE)
        # test_prog.add_to_first_available_node(STR_BUILD, SIBLING_EDGE)
        # test_prog.add_to_first_available_node(STR_BUILD, SIBLING_EDGE)

        # test_prog.prog.max_depth = 15

        # last_node = test_prog.prog.tree_mod.get_node_with_api(test_prog.prog.curr_prog, 'java.lang.StringBuilder.append(long)')
        # test_prog.prog.tree_mod.create_and_add_node(STOP, last_node, SIBLING_EDGE)

        num_iter = 330

        # print(test_prog.prog.curr_prog.length)
        for i in range(num_iter):
            print("\n\n---------------")
            print(i)
            test_prog.prog.mcmc(i)
            print_verbose_tree_info(test_prog.prog.curr_prog)
            # if i % 1 == 0:
            #     test_prog.update_nodes_and_edges(verbose=True)

        # test_prog.update_nodes_and_edges(verbose=True)
        test_prog.print_summary_logs()

    def test_dev(self):
        constraints = ['java.lang.String.length()', 'java.io.BufferedWriter.newLine()']
        initial_state, test_prog, rt, fp = self.test_vae_beam_search(constraints, beam_width=1)

        test_prog, expected_nodes, expected_edges = create_base_program(SAVED_MODEL_PATH, constraints, rt, fp,
                                                                        debug=False, verbose=False)

        # print(initial_state.shape)
        # print(test_prog.prog.initial_state.shape)

        test_prog.prog.initial_state = initial_state


        # test_prog.prog.undo_add_random_node(added_node)

        # test_prog.prog.get_random_initial_state()
        #
        # test_prog.prog.add_random_node()

        num_iter = 100

        for i in range(num_iter):
            print(i)
            test_prog.prog.mcmc()

        test_prog.print_summary_logs()

    def test_individual_pair(self):
        constraints = [STR_APP]
        num_iter = 330
        NEW_VOCAB = 'new_1k_vocab_min_3-600000'
        graph_analyzer = GraphAnalyzer(NEW_VOCAB, load_reader=True)
        rt, fp = graph_analyzer.get_top_k_rt_fp(constraints)
        rt = [graph_analyzer.num2rettype[rt[0][0]]]
        fp = [graph_analyzer.num2fp[fp[0][0]], graph_analyzer.num2fp[fp[1][0]]]
        test_prog, expected_nodes, expected_edges = create_base_program(SAVED_MODEL_PATH, constraints, rt, fp,
                                                                        debug=False, verbose=False)
        for i in range(num_iter):
            if i % 100 == 0:
                print("i:", str(i))
            test_prog.prog.mcmc()

        test_prog.print_summary_logs()

    def test_pairs(self):
        api_lists = [MOST_COMMON_APIS, MID_COMMON_APIS, UNCOMMON_APIS]
        NEW_VOCAB = 'new_1k_vocab_min_3-600000'
        graph_analyzer = GraphAnalyzer(NEW_VOCAB, load_reader=True)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = 'all_disjoint_pairs_test.txt'
        file_path = dir_path + "/lofi_testing/" + filename
        logs_f = open(os.path.join(file_path), 'a+')
        logs_f.write("\nModel: " + NEW_VOCAB)
        logs_f.write("\nDate: " + str(datetime.datetime.now()))
        num_iter = 330
        logs_f.write("\nNumber of MCMC Steps: " + str(num_iter))
        logs_f.flush()
        for api_list in api_lists:
            for api in api_list:
                for level in [TOP, MID, LOW]:
                    if api_list == MOST_COMMON_APIS:
                        disjoint_pair = MOST_COMMON_DISJOINT_PAIRS
                    elif api_list == MID_COMMON_APIS:
                        disjoint_pair = MID_COMMON_DISJOINT_PAIRS
                    elif api_list == UNCOMMON_APIS:
                        disjoint_pair = UNCOMMON_DISJOINT_PAIRS
                    else:
                        raise ValueError("api_list must be MOST, MID or UNCOMMON")
                    constraints = [api, disjoint_pair[api][level][0][0]]
                    # print("\n\n\n-------------------------\nconstraints:", constraints)
                    rt, fp = graph_analyzer.get_top_k_rt_fp(constraints)
                    rt = [graph_analyzer.num2rettype[rt[0][0]]]
                    fp = [graph_analyzer.num2fp[fp[0][0]], graph_analyzer.num2fp[fp[1][0]]]
                    test_prog, expected_nodes, expected_edges = create_base_program(SAVED_MODEL_PATH, constraints, rt, fp,
                                                                                    debug=False, verbose=False)
                    test_prog.prog.debug = False
                    test_prog.prog.verbose = False

                    for i in range(num_iter):
                        if i % 100 == 0:
                            print("i:", str(i))
                        test_prog.prog.mcmc()

                    test_prog.print_summary_logs()
                    test_prog.save_summary_logs(logs_f)
                    logs_f.flush()

    def test_vae_beam_search(self, constraints=None, beam_width=5):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--python_recursion_limit', type=int, default=10000,
                            help='set recursion limit for the Python interpreter')
        parser.add_argument('--continue_from', type=str,
                            default='../trainer_vae/save/1k_vocab_constraint_min_3-600000/',
                            help='ignore config options and continue training model checkpointed here')
        parser.add_argument('--data', default='../data_extractor/data/new_1k_vocab_constraint_min_3-600000/')
        clargs = parser.parse_args()
        sys.setrecursionlimit(clargs.python_recursion_limit)

        size = beam_width

        encoder = BayesianPredictor(clargs.continue_from, batch_size=size)

        if constraints is None:
            constraints = ['java.util.Map<java.lang.String,byte[]>.hashCode()', 'java.lang.String.String(byte[])']
        NEW_VOCAB = 'new_1k_vocab_min_3-600000'
        graph_analyzer = GraphAnalyzer(NEW_VOCAB, load_reader=True)
        rt, fp = graph_analyzer.get_top_k_rt_fp(constraints)
        rt = [graph_analyzer.num2rettype[rt[0][0]]]
        fp = [graph_analyzer.num2fp[fp[0][0]], graph_analyzer.num2fp[fp[1][0]]]
        test_prog, expected_nodes, expected_edges = create_base_program(SAVED_MODEL_PATH, constraints, rt, fp,
                                                                        debug=False, verbose=False)

        nodes, edges = test_prog.prog.tree_mod.get_vector_representation(test_prog.prog.curr_prog)
        nodes = nodes[:test_prog.prog.config.max_num_api]
        edges = edges[:test_prog.prog.config.max_num_api]
        nodes = np.array([nodes])
        edges = np.array([edges])

        input_nodes = np.zeros([size, 8])
        input_edges = np.zeros([size, 8])
        input_nodes[0] = nodes
        input_edges[0] = edges
        input_rt = np.zeros([size])
        input_rt[0] = test_prog.prog.ret_type[0]
        input_fp = np.zeros([size, 8])
        input_fp[0] = np.array(test_prog.prog.fp)

        psi = encoder.get_initial_state(input_nodes, input_edges, input_rt.T, input_fp)

        # psi = encoder.get_initial_state(nodes, edges, np.array(test_prog.prog.ret_type), np.array(test_prog.prog.fp))
        psi_ = np.transpose(np.array(psi), [1, 0, 2])  # batch_first
        encoder.close()

        predictor = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)
        searcher = ProgramBeamSearcher(predictor)
        ast_paths, fp_paths, ret_types = searcher.beam_search(initial_state=psi_)
        print(' ========== AST ==========')
        for i, ast_path in enumerate(ast_paths):
            print(ast_path)

        print(' ========== Fp ==========')
        for i, fp_path in enumerate(fp_paths):
            print(fp_path)

        print(' ========== Return Type ==========')
        print(ret_types)

        predictor.close()

        return psi_, test_prog, rt, fp


if __name__ == '__main__':
    unittest.main()
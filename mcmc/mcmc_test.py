import random
import unittest
from mcmc import Node, MCMCProgram, SIBLING_EDGE, CHILD_EDGE
from trainer_vae.model import Model

import numpy as np
import tensorflow as tf

import unittest.mock as mock

# Shorthand for nodes
STR_BUF = 'java.lang.StringBuffer.StringBuffer()'
STR_APP = 'java.lang.StringBuffer.append(java.lang.String)'
READ_LINE = 'java.io.BufferedReader.readLine()'
START = 'DSubTree'
CLOSE = 'java.io.InputStream.close()'
STOP = 'DStop'
DBRANCH = 'DBranch'
DLOOP = 'DLoop'
DEXCEPT = 'DExcept'
STR_LEN = 'java.lang.String.length()'

# SAVED MODEL
SAVED_MODEL_PATH = '/Users/meghanachilukuri/Documents/GitHub/bayou_mcmc/trainer_vae/save/aws/save/'


class MCMCProgramTest(unittest.TestCase):
    def create_base_program(self, constraints):
        test_prog = MCMCProgramWrapper(SAVED_MODEL_PATH, constraints)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START]
        expected_edges = []
        for i in test_prog.constraints:
            expected_nodes.append(i)
            expected_edges.append(False)
        return test_prog, expected_nodes, expected_edges

    def create_str_buf_base_program(self):
        return self.create_base_program([STR_BUF, 'abc'])

    def create_eight_node_program(self):
        test_prog, expected_nodes, expected_edges = self.create_str_buf_base_program()
        test_prog.add_to_first_available_node(STR_BUF, SIBLING_EDGE)
        test_prog.add_to_first_available_node(STR_APP, CHILD_EDGE)
        test_prog.add_to_first_available_node(READ_LINE, CHILD_EDGE)
        test_prog.add_to_first_available_node(STR_APP, SIBLING_EDGE)
        test_prog.add_to_first_available_node(READ_LINE, SIBLING_EDGE)
        test_prog.add_to_first_available_node(STR_BUF, CHILD_EDGE)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_APP, STR_BUF, STR_BUF]
        expected_edges = [True, True, True, False, False, False, False]

        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")

        return test_prog, expected_nodes, expected_edges

    def create_dbranch(self, test_prog, parent):
        # expected nodes = [DBRANCH, STR_BUF, STR_APP, STOP, READ_LINE, STOP]
        # expected edges = [CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE, SIBLING_EDGE]
        dbranch = test_prog.prog.create_and_add_node(DBRANCH, parent, SIBLING_EDGE)
        cond = test_prog.prog.create_and_add_node(STR_BUF, dbranch, CHILD_EDGE)
        then = test_prog.prog.create_and_add_node(STR_APP, cond, CHILD_EDGE)
        test_prog.prog.create_and_add_node(STOP, then, SIBLING_EDGE)
        else_node = test_prog.prog.create_and_add_node(READ_LINE, cond, SIBLING_EDGE)
        test_prog.prog.create_and_add_node(STOP, else_node, SIBLING_EDGE)
        return test_prog, dbranch

    def create_dloop(self, test_prog, parent):
        # expected nodes = [DLOOP, READ_LINE, CLOSE, STOP]
        # expected edges = [CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE]
        dloop = test_prog.prog.create_and_add_node(DLOOP, parent, SIBLING_EDGE)
        cond = test_prog.prog.create_and_add_node(READ_LINE, dloop, CHILD_EDGE)
        body = test_prog.prog.create_and_add_node(CLOSE, cond, CHILD_EDGE)
        test_prog.prog.create_and_add_node(STOP, body, SIBLING_EDGE)
        return test_prog, dloop

    def create_dexcept(self, test_prog, parent):
        # expected nodes = [DEXCEPT, STR_BUF, CLOSE, STOP]
        # expected edges = [CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE]
        dexcept = test_prog.prog.create_and_add_node(DEXCEPT, parent, SIBLING_EDGE)
        catch = test_prog.prog.create_and_add_node(STR_BUF, dexcept, CHILD_EDGE)
        try_node = test_prog.prog.create_and_add_node(CLOSE, catch, CHILD_EDGE)
        test_prog.prog.create_and_add_node(STOP, try_node, SIBLING_EDGE)
        return test_prog, dexcept

    def create_all_dtypes_program(self):
        test_prog, expected_nodes, expected_edges = self.create_str_buf_base_program()
        dbranch_parent = test_prog.prog.get_node_in_position(1)
        test_prog, dbranch = self.create_dbranch(test_prog, dbranch_parent)
        test_prog, dloop = self.create_dloop(test_prog, dbranch)
        test_prog, dexcept = self.create_dexcept(test_prog, dloop)
        test_prog.update_nodes_and_edges()
        expected_nodes = [START, STR_BUF, DBRANCH, STR_BUF, STR_APP, STOP, READ_LINE, STOP, DLOOP, READ_LINE, CLOSE,
                          STOP, DEXCEPT, STR_BUF, CLOSE]
        expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE, SIBLING_EDGE,
                          SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE]
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")

        return test_prog, expected_nodes, expected_edges

    def test_init(self):
        # Test basic program
        test_prog, expected_nodes, expected_edges = self.create_str_buf_base_program()

        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 2)

        test_prog.prog.calculate_probability()
        print(test_prog.prog.curr_log_prob)

        # Test 8 node program
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()

        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 8)

        test_prog.prog.calculate_probability()
        print(test_prog.prog.curr_log_prob)

    def test_create_and_add_node(self):
        test_prog, expected_nodes, expected_edges = self.create_str_buf_base_program()

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
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()

        self.assertEqual(START, test_prog.prog.get_node_in_position(0).api_name)
        self.assertEqual(STR_APP, test_prog.prog.get_node_in_position(1).api_name)
        self.assertEqual(READ_LINE, test_prog.prog.get_node_in_position(2).api_name)
        self.assertEqual(STR_BUF, test_prog.prog.get_node_in_position(3).api_name)
        self.assertEqual(READ_LINE, test_prog.prog.get_node_in_position(4).api_name)
        self.assertEqual(STR_APP, test_prog.prog.get_node_in_position(5).api_name)
        self.assertEqual(STR_BUF, test_prog.prog.get_node_in_position(6).api_name)
        self.assertEqual(STR_BUF, test_prog.prog.get_node_in_position(7).api_name)

    def test_get_vector_representation(self):
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()
        node_nums = [test_prog.vocab2node[i] for i in expected_nodes]
        exp_node_nums = np.zeros([1, test_prog.prog.max_depth], dtype=np.int32)
        exp_edges = np.zeros([1, test_prog.prog.max_depth], dtype=np.bool)
        exp_node_nums[0, :len(node_nums)] = node_nums
        exp_edges[0, :len(expected_edges)] = expected_edges
        exp_node_nums = exp_node_nums[0]
        exp_edges = exp_edges[0]
        exp_node_nums = exp_node_nums.tolist()
        exp_edges = exp_edges.tolist()

        nodes, edges = test_prog.prog.get_vector_representation()

        self.assertListEqual(nodes.tolist(), exp_node_nums, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(edges.tolist(), exp_edges, "Edges must be equal to expected nodes in program.")

    def test_validity(self):
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()
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

    @mock.patch.object(MCMCProgram, 'get_ast_idx')
    @mock.patch.object(random, 'randint')
    def test_add_and_undo_random_node(self, mock_randint, mock_get_ast_idx):
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()
        mock_get_ast_idx.return_value = test_prog.vocab2node[CLOSE]
        mock_randint.return_value = 5

        test_prog.prog.max_depth = 11

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
        test_prog.prog.max_depth = 8
        mock_randint.return_value = 4
        new_node = test_prog.prog.add_random_node()
        test_prog.update_nodes_and_edges()
        self.assertEqual(new_node, None)
        self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
        self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
        self.assertEqual(test_prog.prog.curr_prog.length, 8)

    @mock.patch.object(random, 'randint')
    def test_delete_and_undo_random_node(self, mock_randint):
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()
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
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()
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
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()
        self.assertTrue(test_prog.prog.check_validity())

        mock_randint.return_value = 1
        test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertTrue(test_prog.prog.check_validity())

        test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertTrue(test_prog.prog.check_validity())

        test_prog.prog.delete_random_node()
        test_prog.update_nodes_and_edges()
        self.assertFalse(test_prog.prog.check_validity())

    @mock.patch.object(MCMCProgram, 'get_ast_idx')
    @mock.patch.object(random, 'randint')
    def test_add_and_swap_node(self, mock_randint, mock_get_ast_idx):
        test_prog, expected_nodes, expected_edges = self.create_eight_node_program()
        mock_get_ast_idx.return_value = test_prog.vocab2node[CLOSE]
        mock_randint.return_value = 5

        test_prog.prog.max_depth = 11

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

    @mock.patch.object(MCMCProgram, 'get_valid_random_node')
    @mock.patch.object(MCMCProgram, 'get_ast_idx')
    @mock.patch.object(random, 'choice')
    def test_add_dnode_node(self, mock_randint, mock_get_ast_idx, mock_get_pos):
        pass

    def test_mcmc(self):
        test_prog, expected_nodes, expected_edges = self.create_base_program([STR_LEN, 'slkfje'])

        test_prog.prog.max_depth = 15

        num_iter = 50

        for i in range(num_iter):
            print("\n", i)
            # print(i)
            test_prog.prog.mcmc()
            test_prog.update_nodes_and_edges(verbose=True)
        #
        test_prog.print_summary_logs()

    def test_all_dtypes_program(self):
        test_prog, expected_nodes, expected_edges = self.create_all_dtypes_program()
        self.assertTrue(test_prog.prog.check_validity())

    @mock.patch.object(random, 'choice')
    def test_dev(self, mock_rand_choice):
        mock_rand_choice.return_value = 'add'

        test_prog, expected_nodes, expected_edges = self.create_base_program([STR_LEN, 'slkfje'])

        test_prog.prog.max_depth = 15

        num_iter = 1

        for i in range(num_iter):
            print("\n", i)
            # print(i)
            test_prog.prog.mcmc()
            test_prog.update_nodes_and_edges(verbose=True)
        #
        test_prog.print_summary_logs()


class MCMCProgramWrapper:
    def __init__(self, save_dir, constraints):
        # init MCMCProgram
        self.prog = MCMCProgram(save_dir)
        self.prog.init_program(constraints)

        self.constraints = self.prog.constraints
        self.vocab2node = self.prog.vocab2node
        self.node2vocab = self.prog.node2vocab

        # init nodes, edges and parents
        self.nodes = []
        self.edges = []
        self.parents = []
        self.update_nodes_and_edges()

    def add_to_first_available_node(self, api_name, edge):
        curr_node = self.prog.curr_prog
        stack = []
        while curr_node is not None:
            if edge == SIBLING_EDGE and curr_node.sibling is None and curr_node.api_name != STOP:
                break

            elif edge == CHILD_EDGE and curr_node.child is None and curr_node.api_name != STOP:
                break

            else:
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

        parent = curr_node

        self.create_and_add_node(api_name, parent, edge)

    def create_and_add_node(self, api_name, parent, edge):
        self.prog.create_and_add_node(api_name, parent, edge)

        self.update_nodes_and_edges()

    def update_nodes_and_edges(self, verbose=False):
        curr_node = self.prog.curr_prog

        stack = []
        nodes = []
        edges = []
        parents = [None]

        pos_counter = 0

        while curr_node is not None:
            nodes.append(curr_node.api_name)

            if verbose:
                self.verbose_node_info(curr_node, pos=pos_counter)

            pos_counter += 1

            if curr_node.api_name != START:
                edges.append(curr_node.parent_edge)
                parents.append(curr_node.parent.api_name)

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
                    # remove last DSTOP node
                    if curr_node.api_name == STOP:
                        curr_node.parent.remove_node(curr_node.parent_edge)
                        nodes.pop()
                        edges.pop()
                        parents.pop()
                    curr_node = None

        self.nodes = nodes
        self.edges = edges
        self.parents = parents

    def verbose_node_info(self, node, pos=None):
        node_info = {"api name": node.api_name, "length": node.length, "api num": node.api_num,
                     "parent edge": node.parent_edge}

        if pos is not None:
            node_info["position"] = pos

        if node.parent is not None:
            node_info["parent"] = node.parent.api_name
        else:
            node_info["parent"] = node.parent

            if node.api_name != 'DSubTree':
                print("WARNING: node does not have a parent", node.api_name)

        if node.sibling is not None:
            node_info["sibling"] = node.sibling.api_name

            if node.sibling.parent is None:
                print("WARNING: sibling parent is None for node", node.api_name, "in pos", pos)
                node_info["sibling parent"] = node.sibling.parent
            else:
                node_info["sibling parent"] = node.sibling.parent.api_name

            node_info["sibling parent edge"] = node.sibling.parent_edge
        else:
            node_info["sibling"] = node.sibling

        if node.child is not None:
            node_info["child"] = node.child.api_name

            if node.child.parent is None:
                print("WARNING: child parent is None for node", node.api_name, "in pos", pos)
                node_info["child parent"] = node.child.parent
            else:
                node_info["child parent"] = node.child.parent.api_name

            node_info["child parent edge"] = node.child.parent_edge

        print(node_info)

        return node_info

    def print_summary_logs(self):
        self.update_nodes_and_edges()
        nodes, edges = self.prog.get_node_names_and_edges()
        print("\n", "-------------------LOGS:-------------------")
        print("Nodes:", nodes)
        print("Edges:", edges)
        print("Parents:", self.parents)
        print("Total accepted transformations:", self.prog.accepted)
        print("Total rejected transformations:", self.prog.rejected)
        print("Total valid transformations:", self.prog.valid)
        print("Total invalid transformations:", self.prog.invalid)
        print("Total attempted add transforms:", self.prog.add)
        print("Total accepted add transforms:", self.prog.add_accepted)
        print("Total attempted delete transforms:", self.prog.delete)
        print("Total accepted delete transforms:", self.prog.delete_accepted)
        print("Total attempted swap transforms:", self.prog.swap)
        print("Total accepted swap transforms:", self.prog.swap_accepted)



if __name__ == '__main__':
    unittest.main()
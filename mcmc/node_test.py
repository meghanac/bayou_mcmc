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
    create_all_dtypes_program

# SAVED MODEL
SAVED_MODEL_PATH = '/Users/meghanachilukuri/Documents/GitHub/bayou_mcmc/trainer_vae/save/1k_vocab_constraint_min_3-600000'


class NodeTest(unittest.TestCase):
    def test_add_node(self):
        pass

    def test_remove_node(self):
        pass

    def test_update_length(self):
        pass

    def test_copy(self):
        test_prog, _, _ = create_eight_node_program(SAVED_MODEL_PATH)
        self.copy_test(test_prog.prog.curr_prog)
        test_prog, _, _ = create_all_dtypes_program(SAVED_MODEL_PATH)
        self.copy_test(test_prog.prog.curr_prog)

    def copy_test(self, prog_orig):
        print("original:")
        print_verbose_tree_info(prog_orig)
        prog_copy = prog_orig.copy()
        print("copy:")
        print_verbose_tree_info(prog_copy)

        self.assertEqual(prog_orig.length, prog_copy.length)

        stack = []

        node1 = prog_orig
        node2 = prog_copy
        for i in range(prog_copy.length):
            self.assertEqual(node1.api_name, node2.api_name)
            self.assertEqual(node1.api_num, node2.api_num)
            if node1 == prog_orig:
                self.assertIsNone(node1.parent)
                self.assertIsNone(node1.parent_edge)
                self.assertIsNone(node2.parent)
                self.assertIsNone(node2.parent_edge)
            else:
                self.assertIsNotNone(node1.parent)
                self.assertIsNotNone(node1.parent_edge)
                self.assertIsNotNone(node2.parent)
                self.assertIsNotNone(node2.parent_edge)
                self.assertEqual(node1.parent.api_name, node2.parent.api_name)
                self.assertEqual(node1.parent_edge, node2.parent_edge)
            self.assertEqual(node1.length, node2.length)
            self.assertEqual(node1.non_dnode_length, node2.non_dnode_length)
            if node1.child is None:
                self.assertIsNone(node2.child)
            else:
                self.assertIsNotNone(node2.child)

            if node1.sibling is None:
                self.assertIsNone(node2.sibling)
            else:
                self.assertIsNotNone(node2.sibling)

            next_node1 = None
            next_node2 = None

            if node1.child is not None:
                next_node1 = node1.child
                next_node2 = node2.child
                if node1.sibling is not None:
                    stack.append((node1.sibling, node2.sibling))
            elif node1.sibling is not None:
                next_node1 = node1.sibling
                next_node2 = node2.sibling
            else:
                if len(stack) > 0:
                    next_node1, next_node2 = stack.pop()
                else:
                    if i != prog_copy.length - 1:
                        raise ValueError("Something went wrong with traversal")

            node1 = next_node1
            node2 = next_node2

            if (node1 is None or node2 is None) and i != prog_copy.length - 1:
                raise ValueError("Something went wrong with traversal")


if __name__ == '__main__':
    unittest.main()

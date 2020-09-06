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

ALL_TRAINING_DATA_GPU_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_normal_gpu'

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

test_prog, expected_nodes, expected_edges = create_base_program(ALL_TRAINING_DATA_GPU_PATH,
                                                                ['java.util.Random.Random(long)',
                                                                 'java.io.OutputStream.write(byte[])'],
                                                                ['void'],
                                                                ['DSubTree'])

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
    test_prog.prog.mcmc()
    print_verbose_tree_info(test_prog.prog.curr_prog)
    # if i % 1 == 0:
    #     test_prog.update_nodes_and_edges(verbose=True)

# test_prog.update_nodes_and_edges(verbose=True)
test_prog.print_summary_logs()

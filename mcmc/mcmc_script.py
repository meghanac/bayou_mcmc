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
from infer import BayesianPredictor
from trainer_vae.model import Model

import numpy as np
import tensorflow as tf

from test_utils import STR_BUF, STR_APP, READ_LINE, CLOSE, STR_LEN, STR_BUILD, STR_BUILD_APP, create_base_program, \
    create_str_buf_base_program, create_eight_node_program, create_dbranch, create_dloop, create_dexcept, \
    create_all_dtypes_program

from mcmc import INSERT, DELETE, REPLACE, SWAP, ADD_DNODE, GROW_CONST

from test_suite import MOST_COMMON_APIS, MID_COMMON_APIS, UNCOMMON_APIS, MID_COMMON_DISJOINT_PAIRS, \
    MOST_COMMON_DISJOINT_PAIRS, UNCOMMON_DISJOINT_PAIRS, ONE_OF_EACH

from data_extractor.graph_analyzer import GraphAnalyzer

TOP = 'top'
MID = 'mid'
LOW = 'low'

import unittest.mock as mock

# SAVED MODEL
SAVED_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/1k_vocab_constraint_min_3-600000'

ALL_DATA_1K_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_1k_vocab'

ALL_DATA_1K_05_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_1k_vocab_0.5_KL_beta'

ALL_DATA_1K_025_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_1k_vocab_0.25_KL_beta'


NEW_VOCAB = 'new_1k_vocab_min_3-600000'
graph_analyzer = GraphAnalyzer(NEW_VOCAB, load_reader=True)
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = 'one_of_each_test_results_3.txt'
file_path = dir_path + "/lofi_testing/" + filename
logs_f = open(os.path.join(file_path), 'a+')
logs_f.write("\nModel: " + NEW_VOCAB)
logs_f.write("\nDate: " + str(datetime.datetime.now()))
num_iter = 330
logs_f.write("\nNumber of MCMC Steps: " + str(num_iter))
logs_f.flush()
for constraints in ONE_OF_EACH[10:15]:
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
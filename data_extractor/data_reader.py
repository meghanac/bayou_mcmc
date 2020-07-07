# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import json
import sys
import textwrap

import ijson  # .backends.yajl2_cffi as ijson
import numpy as np
import random
import os
import pickle

from data_extractor.utils import dump_vocab
from ast_helper.ast_reader import AstReader
from ast_helper.ast_traverser import AstTraverser, TooLongBranchingException, TooLongLoopingException
from data_extractor.dictionary import Dictionary

MAX_AST_DEPTH = 32
MAX_FP_DEPTH = 8
MAX_KEYWORDS = 10


class TooLongPathError(Exception):
    pass


class InvalidSketchError(Exception):
    pass


HELP = """{}"""

API_TO_PROG_IDS = 'api_to_prog_ids'
RT_TO_PROG_IDS = 'rt_to_prog_ids'
FP_TO_PROG_IDS = 'fp_to_prog_ids'


class Reader:
    def __init__(self, clargs, infer=False, create_database=False, vocab=None):
        self.ast_traverser = AstTraverser()
        self.ast_reader = AstReader()
        self.infer = infer
        self.vocab = argparse.Namespace()

        if self.infer:
            assert vocab is not None
            self.vocab = vocab

        random.seed(12)
        # read the raw evidences and targets
        print('Reading data file...')
        if self.infer:
            self.api_dict = Dictionary(self.vocab.api_dict)
            self.ret_dict = Dictionary(self.vocab.ret_dict)
            self.fp_dict = Dictionary(self.vocab.fp_dict)
            self.keyword_dict = Dictionary(self.vocab.keyword_dict)
        else:
            self.api_dict = Dictionary()
            self.ret_dict = Dictionary()
            self.fp_dict = Dictionary()
            self.keyword_dict = Dictionary()

        ast_programs, return_types, formal_params, keywords = self.read_data(clargs.data + clargs.data_filename + ".json")

        # setup input and target chars/vocab
        if not self.infer:
            self.vocab.api_dict, self.vocab.api_dict_size = self.api_dict.get_call_dict()
            self.vocab.ret_dict, self.vocab.ret_dict_size = self.ret_dict.get_call_dict()
            self.vocab.fp_dict, self.vocab.fp_dict_size = self.fp_dict.get_call_dict()
            self.vocab.keyword_dict, self.vocab.keyword_dict_size = self.keyword_dict.get_call_dict()

        sz = len(ast_programs)

        ## Wrangle Program ASTs
        # wrangle the evidences and targets into numpy arrays
        self.nodes = np.zeros((sz, MAX_AST_DEPTH), dtype=np.int32)
        self.edges = np.zeros((sz, MAX_AST_DEPTH), dtype=np.bool)
        self.targets = np.zeros((sz, MAX_AST_DEPTH), dtype=np.int32)

        for i, api_path in enumerate(ast_programs):
            len_path = min(len(api_path), MAX_AST_DEPTH)
            mod_path = api_path[:len_path]
            self.nodes[i, :len_path] = [p[0] for p in mod_path]
            self.edges[i, :len_path] = [p[1] for p in mod_path]
            self.targets[i, :len_path] = [p[2] for p in mod_path]

        ## Wrangle Return Types
        self.return_types = np.zeros(sz, dtype=np.int32)
        for i, rt in enumerate(return_types):
            self.return_types[i] = rt

        ## Wrangle Formal Param Types
        self.fp_types = np.zeros((sz, MAX_FP_DEPTH), dtype=np.int32)
        self.fp_type_targets = np.zeros((sz, MAX_FP_DEPTH), dtype=np.int32)
        for i, fp_list in enumerate(formal_params):
            len_list = min(len(fp_list), MAX_FP_DEPTH)
            mod_list = fp_list[:len_list]
            self.fp_types[i, :len_list] = mod_list
            self.fp_type_targets[i, 0:(len_list - 1)] = mod_list[1:len_list]


        ## Wrangle Keywords
        self.keywords = np.zeros((sz, MAX_KEYWORDS), dtype=np.int32)
        for i, kw in enumerate(keywords):
            len_list = min(len(kw), MAX_KEYWORDS)
            mod_list = kw[:len_list]
            self.keywords[i, :len_list] = mod_list

        self.create_database = create_database
        if create_database:
            self.database = {}
            self.save_prog_database(sz)

        # self.js_programs = js_programs
        self.save_data(clargs.data)

        # reset batches
        print('Done!')

    def read_data(self, filename):

        data_points = []
        done, ignored_for_branch, ignored_for_loop = 0, 0, 0

        f = open(filename, 'rb')

        for program in ijson.items(f, 'programs.item'):
            if 'ast' not in program:
                continue
            try:

                parsed_api_array = self.read_ast(program['ast'])
                return_type_id = self.read_return_type(program['returnType'])
                parsed_fp_array = self.read_formal_params(program['formalParam'])
                keywords = self.read_keywords(program['keywords'])

                data_points.append((parsed_api_array, return_type_id, parsed_fp_array, keywords))
                done += 1

            except TooLongLoopingException as e1:
                ignored_for_loop += 1

            except TooLongBranchingException as e2:
                ignored_for_branch += 1

            if done % 1000000 == 0:
                print('Extracted data for {} programs'.format(done), end='\n')
                # break

        print('{:8d} programs/asts in training data'.format(done))
        print('{:8d} programs/asts missed in training data for loop'.format(ignored_for_loop))
        print('{:8d} programs/asts missed in training data for branch'.format(ignored_for_branch))

        # randomly shuffle to avoid bias towards initial data points during training
        random.shuffle(data_points)
        parsed_api_array, return_type_ids, formal_param_ids, keywords = zip(*data_points)  # unzip

        return parsed_api_array, return_type_ids, formal_param_ids, keywords

    def save_prog_database(self, sz):
        # program_ids = {}
        api_to_prog_ids = {}
        rt_to_prog_ids = {}
        fp_to_prog_ids = {}
        stored_programs = set([])
        # counter = 0
        repeat_prog_counter = 0

        for i in range(sz):
            # add apis
            for api in self.nodes[i]:
                if api not in api_to_prog_ids:
                    api_to_prog_ids[api] = {i}
                else:
                    api_to_prog_ids[api].add(i)
            for api in self.targets[i]:
                if api not in api_to_prog_ids:
                    api_to_prog_ids[api] = {i}
                else:
                    api_to_prog_ids[api].add(i)

            # add return types
            if self.return_types[i] in rt_to_prog_ids:
                rt_to_prog_ids[self.return_types[i]].add(i)
            else:
                rt_to_prog_ids[self.return_types[i]] = {i}

            # add formal parameters
            for fp in self.fp_types[i]:  # TODO: should I  + self.fp_type_targets[i]?
                if fp in fp_to_prog_ids:
                    fp_to_prog_ids[fp].add(i)
                else:
                    fp_to_prog_ids[fp] = {i}

        # # freeze sets
        # api_to_prog_ids = dict([(api, frozenset(progs)) for api, progs in api_to_prog_ids.items()])
        # rt_to_prog_ids = dict([(rt, frozenset(progs)) for rt, progs in rt_to_prog_ids.items()])
        # fp_to_prog_ids = dict([(fp, frozenset(progs)) for fp, progs in fp_to_prog_ids.items()])

        # self.database['program_ids'] = program_ids
        self.database[API_TO_PROG_IDS] = api_to_prog_ids
        self.database[RT_TO_PROG_IDS] = rt_to_prog_ids
        self.database[FP_TO_PROG_IDS] = fp_to_prog_ids

    def save_data(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/ast_apis.pickle', 'wb') as f:
            pickle.dump([self.nodes, self.edges, self.targets], f)

        with open(path + '/return_types.pickle', 'wb') as f:
            pickle.dump(self.return_types, f)

        with open(path + '/formal_params.pickle', 'wb') as f:
            pickle.dump([self.fp_types, self.fp_type_targets], f)

        with open(path + '/keywords.pickle', 'wb') as f:
            pickle.dump(self.keywords, f)

        with open(os.path.join(path + '/vocab.json'), 'w') as f:
            json.dump(dump_vocab(self.vocab), fp=f, indent=2)

        if self.create_database:
            self.save_database(path)

        # with open(path + '/js_programs.json', 'w') as f:
        #     json.dump({'programs': self.js_programs}, fp=f, indent=2)

    def save_database(self, path):
        with open(path + '/program_database.pickle', 'wb') as f:
            pickle.dump(self.database, f)


    def read_ast(self, program_ast_js):
        # Read the Program AST to a sequence
        ast_node_graph = self.ast_reader.get_ast_from_json(program_ast_js['_nodes'])
        self.ast_traverser.check_nested_branch(ast_node_graph)
        self.ast_traverser.check_nested_loop(ast_node_graph)
        path = self.ast_traverser.depth_first_search(ast_node_graph)
        parsed_api_array = []
        for i, (curr_node_val, parent_node_id, edge_type) in enumerate(path):
            if self.infer:
                curr_node_id = self.api_dict.get_node_val(curr_node_val)
            else:
                curr_node_id = self.api_dict.conditional_add_node_val(curr_node_val)
            # now parent id is already evaluated since this is top-down breadth_first_search
            parent_call = path[parent_node_id][0]
            parent_call_id = self.api_dict.get_node_val(parent_call)

            if i > 0 and not (
                    curr_node_id is None or parent_call_id is None):  # I = 0 denotes DSubtree ----sibling---> DSubTree
                parsed_api_array.append((parent_call_id, edge_type, curr_node_id))
        return parsed_api_array

    def read_return_type(self, program_ret_js):
        if self.infer:
            return_type_id = self.ret_dict.get_node_val(program_ret_js)
        else:
            return_type_id = self.ret_dict.conditional_add_node_val(program_ret_js)
        return return_type_id

    def read_formal_params(self, program_fp_js):
        # Read the formal params
        formal_param_array = program_fp_js
        if self.infer:
            parsed_fp_array = [self.fp_dict.get_node_val('DSubTree')]
        else:
            parsed_fp_array = [self.fp_dict.conditional_add_node_val('DSubTree')]
        for fp_call in formal_param_array:
            if self.infer:
                fp_call_id = self.fp_dict.get_node_val(fp_call)
            else:
                fp_call_id = self.fp_dict.conditional_add_node_val(fp_call)
            parsed_fp_array.append(fp_call_id)
        return parsed_fp_array

    def read_keywords(self, program_keyword_js):
        # Read the keywords
        keywords = []
        for kw in program_keyword_js:
            if self.infer:
                kw_id = self.keyword_dict.get_node_val(kw)
            else:
                kw_id = self.keyword_dict.conditional_add_node_val(kw)
            keywords.append(kw_id)
        return keywords


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--data', type=str, default='data/1k_vocab_constraint_min_3-600000',
                        help='data to be saved here')

    clargs_ = parser.parse_args()
    if not os.path.exists(clargs_.data):
        os.makedirs(clargs_.data)
    sys.setrecursionlimit(clargs_.python_recursion_limit)
    Reader(clargs_)

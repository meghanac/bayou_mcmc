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

from data_extractor.utils import dump_vocab, read_vocab
from trainer_vae.utils import read_config
from ast_helper.ast_reader import AstReader
from ast_helper.ast_traverser import AstTraverser, TooLongBranchingException, TooLongLoopingException
from data_extractor.dictionary import Dictionary
from data_extractor.dataset_creator import IN_CS, IN_API, MAX_EQ, MIN_EQ, EX_API, EX_CS

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


class BayouTestResultsReader:
    def __init__(self, data_dir_path, data_filename, config_path, infer=True, save=False):
        self.ast_traverser = AstTraverser()
        self.ast_reader = AstReader()
        self.infer = infer
        self.vocab = argparse.Namespace()

        if self.infer:
            js = json.load(open(config_path, "rb"))
            vocab = argparse.Namespace()
            for attr_dict in js['evidence']:
                attr = attr_dict['name']
                vocab.__setattr__(attr, attr_dict['vocab'])
                vocab.__setattr__('chars_' + attr, attr_dict['chars'])
            assert vocab is not None
            self.vocab = vocab

        random.seed(12)
        # read the raw evidences and targets
        print('Reading data file...')
        if self.infer:
            self.api_dict = Dictionary(self.vocab.apicalls)
            self.type_dict = Dictionary(self.vocab.types)

        categories = self.read_data(data_dir_path + data_filename + ".json")

        if save:
            save_f = open(data_dir_path + "/" + data_filename + "_posterior_dist.pickle", "wb")
            save_f.write(pickle.dumps(categories))
            save_f.close()

    def read_data(self, filename):
        done, ignored_for_branch, ignored_for_loop = 0, 0, 0

        f = open(filename, 'rb')

        empty_post_dist = 0
        empty_out_asts = 0
        categories = {IN_API: {}, IN_CS: {}, EX_CS: {}, EX_API: {}, MIN_EQ: {}, MAX_EQ: {}}
        for data_point in ijson.items(f, 'programs.item'):
            if 'out_asts' not in data_point:
                continue
            test_result = {'types': data_point['types'], 'posterior_dist': {}, 'apicalls': data_point['apicalls']}
            for program in data_point['out_asts']:
                try:
                    parsed_api_array = self.read_ast(program['ast'])
                    nodes, edges, targets = zip(*parsed_api_array)
                    test_result['posterior_dist'][(nodes, edges, targets)] = program['probability']

                except TooLongLoopingException as e1:
                    ignored_for_loop += 1

                except TooLongBranchingException as e2:
                    ignored_for_branch += 1

            if len(data_point['out_asts']) == 0:
                # print("\n\n\n")
                # print(data_point.keys())
                empty_out_asts += 1
            if len(test_result['posterior_dist']) == 0:
                empty_post_dist += 1

            done += 1
            if done % 100 == 0:
                print('Extracted data for {} programs'.format(done), end='\n')
                print('Number of empty posteriors:', empty_post_dist)
                print('Number of empty out_asts:', empty_out_asts)

            key = data_point['key']
            key[1] = tuple(key[1])
            key = tuple(key)
            categories[data_point['category']][key] = test_result

        print('{:8d} programs/asts in training data'.format(done))
        print('{:8d} programs/asts missed in training data for loop'.format(ignored_for_loop))
        print('{:8d} programs/asts missed in training data for branch'.format(ignored_for_branch))
        print('{:8d} had empty out_asts.'.format(empty_post_dist))

        f.close()

        return categories
        # return parsed_api_array, return_type_ids, formal_param_ids

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
            for fp in self.fp_type_targets[i]:
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
            f.close()

        with open(path + '/return_types.pickle', 'wb') as f:
            pickle.dump(self.return_types, f)
            f.close()

        with open(path + '/formal_params.pickle', 'wb') as f:
            pickle.dump([self.fp_types, self.fp_type_targets], f)
            f.close()

        with open(os.path.join(path + '/vocab.json'), 'w') as f:
            json.dump(dump_vocab(self.vocab), fp=f, indent=2)
            f.close()

        if self.create_database:
            self.save_database(path)

        # with open(path + '/js_programs.json', 'w') as f:
        #     json.dump({'programs': self.js_programs}, fp=f, indent=2)

    def save_database(self, path):
        with open(path + '/program_database.pickle', 'wb') as f:
            pickle.dump(self.database, f)
            f.close()

    def read_ast(self, program_ast_js):
        # Read the Program AST to a sequence
        ast_node_graph = self.ast_reader.get_ast_from_json(program_ast_js['_nodes'])
        self.ast_traverser.check_nested_branch(ast_node_graph)
        self.ast_traverser.check_nested_loop(ast_node_graph)
        path = self.ast_traverser.depth_first_search(ast_node_graph)
        parsed_api_array = []
        for i, (curr_node_val, parent_node_id, edge_type) in enumerate(path):
            # if self.infer:
            #     curr_node_id = self.api_dict.get_node_val(curr_node_val)
            # else:
            #     curr_node_id = self.api_dict.conditional_add_node_val(curr_node_val)

            curr_node_id = curr_node_val
            # now parent id is already evaluated since this is top-down breadth_first_search
            parent_call = path[parent_node_id][0]
            # parent_call_id = self.api_dict.get_node_val(parent_call)
            parent_call_id = parent_call

            if i > 0 and not (
                    curr_node_id is None or parent_call_id is None):  # I = 0 denotes DSubtree ----sibling---> DSubTree
                parsed_api_array.append((parent_call_id, edge_type, curr_node_id))
        if len(parsed_api_array) == 0:
            print(path)
        return parsed_api_array

    def read_type(self, program_fp_js):
        # Read the formal params
        formal_param_array = program_fp_js
        if self.infer:
            parsed_fp_array = [self.type_dict.get_node_val('DSubTree')]
        else:
            parsed_fp_array = [self.type_dict.conditional_add_node_val('DSubTree')]
        for fp_call in formal_param_array:
            if self.infer:
                fp_call_id = self.type_dict.get_node_val(fp_call)
            else:
                fp_call_id = self.type_dict.conditional_add_node_val(fp_call)
            parsed_fp_array.append(fp_call_id)
        return parsed_fp_array


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

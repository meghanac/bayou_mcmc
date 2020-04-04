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

import ijson
import json
import argparse
import os

# VAR_TRACK_TYPES = ['ret_var_id', 'expr_var_id', 'fp_var_id']
VAR_TRACK_TYPES = ['expr_var_id']


def read_data(filename):
    f = open(filename, 'rb')
    new_programs = []
    for program in ijson.items(f, 'programs.item'):
        if 'ast' not in program:
            continue
        ast_nodes = program['ast']['_nodes'] #list
        ast_nodes.reverse()

        new_ast_nodes = []
        useful_var_ids = set()
        for ast_node in ast_nodes:
            ast_node_type = ast_node['node']

            # Check useful APICall nodes and update their reqd var ids
            if ast_node_type == 'DAPICall':
                for item in VAR_TRACK_TYPES:
                    # if isinstance(item, 'int'):
                    #     useful_var_ids.add(item)
                    # elif isinstance(item, 'list'):
                    #     useful_var_ids.update(item)
                    useful_var_ids.add(item)
            # Check VarCall id if ever used
            elif ast_node_type == 'DVarCall':
                _id = ast_node['_id']
                if _id not in useful_var_ids:
                    continue
            else:
                pass

            new_ast_nodes.append(ast_node)

        new_ast_nodes.reverse()
        new_program = program
        new_program['ast'] = {"node":"DSubTree", "_nodes": new_ast_nodes}


        if len(new_ast_nodes) > 0:
            new_programs.append(new_program)

        # print(new_programs)

        # with open('data_manipulated.json', 'w') as f:
        #     json.dump({new_programs}, fp=f, indent=2)

    return

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument('--python_recursion_limit', type=int, default=100000,
#                         help='set recursion limit for the Python interpreter')
#     parser.add_argument('--filename', type=str)
#     clargs = parser.parse_args()
#     read_data(clargs.filename)

read_data('data/vocab_constraint_1k-1000.json')
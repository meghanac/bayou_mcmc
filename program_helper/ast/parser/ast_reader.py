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

import numpy as np
import pickle

from program_helper.ast.ops import DType, DVarAccess, DAPICall, DSymtabMod, DClsInit, DVarAssign, DAPIInvoke, \
    DAPICallMulti
from program_helper.ast.ops.leaf_ops.DOp import DOp
from program_helper.ast.parser.ast_parser import AstParser
from program_helper.ast.parser.ast_checker import AstChecker
from program_helper.ast.parser.ast_traverser import AstTraverser


class AstReader:

    def __init__(self, max_depth=32,
                 max_loop_num=None,
                 max_branching_num=None,
                 max_variables=None,
                 max_trys=None,
                 concept_vocab=None,
                 api_vocab=None,
                 type_vocab=None,
                 var_vocab=None,
                 op_vocab=None,
                 infer=True):
        self.ast_parser = AstParser()
        self.ast_checker = AstChecker(
            max_depth=max_depth,
            max_loop_num=max_loop_num,
            max_branching_num=max_branching_num,
            max_trys=max_trys,
            max_variables=max_variables)

        self.max_depth = max_depth
        self.infer = infer

        self.concept_vocab = concept_vocab
        self.api_vocab = api_vocab
        self.type_vocab = type_vocab
        self.var_vocab = var_vocab
        self.op_vocab = op_vocab

        self.nodes = None
        self.edges = None
        self.targets = None

        self.var_decl_ids = None

        self.varOrNot = None
        self.typeOrNot = None
        self.apiOrNot = None

        self.type_helper_val = None
        self.expr_type_val = None
        self.ret_type_val = None
        return

    def read_while_vocabing(self, program_ast_js, symtab=None, fp_type_head=None, field_head=None, repair_mode=True):
        # Read the Program AST to a sequence
        if symtab is None:
            symtab = dict()

        self.ast_node_graph = self.ast_parser.get_ast_with_memory(program_ast_js['_nodes'], symtab,
                                                                  fp_type_head=fp_type_head,
                                                                  field_head=field_head,
                                                                  repair_mode=repair_mode)

        if not self.infer:
            self.ast_checker.check(self.ast_node_graph)

        path = AstTraverser.depth_first_search(self.ast_node_graph)

        parsed_ast_array = []
        parent_call_val = 0
        for i, (curr_node_val, curr_node_type, curr_node_validity, curr_node_var_decl_ids,
                parent_node_id,
                edge_type, expr_type, type_helper, return_type) in enumerate(path):

            assert curr_node_validity is True

            type_or_not, var_or_not, api_or_not, symtabmod_or_not, op_or_not = False, False, False, False, False
            expr_type_val, type_helper_val, ret_type_val = 0, 0, 0

            if curr_node_type == DType.name():
                type_or_not = True
                value = self.type_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
            elif curr_node_type == DVarAccess.name():
                var_or_not = True
                value = self.var_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
                type_helper_val = self.type_vocab.conditional_add_or_get_node_val(type_helper, self.infer)
                expr_type_val = self.type_vocab.conditional_add_or_get_node_val(expr_type, self.infer)
                ret_type_val = self.type_vocab.conditional_add_or_get_node_val(return_type, self.infer)
            elif curr_node_type == DAPICall.name():
                api_or_not = True
                value = self.api_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)

                ## Even though the expr_type, ret_type are not part of data extracted now
                ## they can be when random apicalls are invoked during testing
                _, expr_type, ret_type = DAPIInvoke.split_api_call(curr_node_val)
                arg_list = DAPICallMulti.get_formal_types_from_data(curr_node_val)
                _ = self.type_vocab.conditional_add_or_get_node_val(expr_type, self.infer)
                _ = self.type_vocab.conditional_add_or_get_node_val(ret_type, self.infer)
                for arg in arg_list:
                    _ = self.type_vocab.conditional_add_or_get_node_val(arg, self.infer)

            elif curr_node_type == DSymtabMod.name():
                symtabmod_or_not = True
                value = 0
                type_helper_val = self.type_vocab.conditional_add_or_get_node_val(type_helper, self.infer)
            elif curr_node_type == DOp.name():
                op_or_not = True
                value = self.op_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
            else:
                value = self.concept_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)

            # now parent id is already evaluated since this is top-down breadth_first_search
            parent_id = path[parent_node_id][0]
            parent_type = path[parent_node_id][1]
            if parent_type not in [DType.name(), DAPICall.name(), DVarAccess.name(), DSymtabMod.name(), DOp.name()]:
                parent_call_val = self.concept_vocab.get_node_val(parent_id)

            if value is not None and i > 0:
                parsed_ast_array.append((parent_call_val, edge_type, value,
                                         curr_node_var_decl_ids,
                                         var_or_not, type_or_not, api_or_not, symtabmod_or_not, op_or_not,
                                         type_helper_val, expr_type_val, ret_type_val))

        return parsed_ast_array

    # sz is total number of data points, Wrangle Program ASTs into numpy arrays
    def wrangle(self, ast_programs, min_num_data=None):
        if min_num_data is None:
            sz = len(ast_programs)
        else:
            sz = max(min_num_data, len(ast_programs))

        self.nodes = np.zeros((sz, self.max_depth), dtype=np.int32)
        self.edges = np.zeros((sz, self.max_depth), dtype=np.bool)
        self.targets = np.zeros((sz, self.max_depth), dtype=np.int32)

        self.var_decl_ids = np.zeros((sz, self.max_depth), dtype=np.int32)

        self.varOrNot = np.zeros((sz, self.max_depth), dtype=np.bool)
        self.typeOrNot = np.zeros((sz, self.max_depth), dtype=np.bool)
        self.apiOrNot = np.zeros((sz, self.max_depth), dtype=np.bool)
        self.symtabmod_or_not = np.zeros((sz, self.max_depth), dtype=np.bool)
        self.op_or_not = np.zeros((sz, self.max_depth), dtype=np.bool)

        self.type_helper_val = np.zeros((sz, self.max_depth), dtype=np.int32)
        self.expr_type_val = np.zeros((sz, self.max_depth), dtype=np.int32)
        self.ret_type_val = np.zeros((sz, self.max_depth), dtype=np.int32)

        for i, api_path in enumerate(ast_programs):
            len_path = min(len(api_path), self.max_depth)
            mod_path = api_path[:len_path]
            self.nodes[i, :len_path] = [p[0] for p in mod_path]
            self.edges[i, :len_path] = [p[1] for p in mod_path]
            self.targets[i, :len_path] = [p[2] for p in mod_path]

            self.var_decl_ids[i, :len_path] = [p[3] for p in mod_path]

            self.varOrNot[i, :len_path] = [p[4] for p in mod_path]
            self.typeOrNot[i, :len_path] = [p[5] for p in mod_path]
            self.apiOrNot[i, :len_path] = [p[6] for p in mod_path]
            self.symtabmod_or_not[i, :len_path] = [p[7] for p in mod_path]
            self.op_or_not[i, :len_path] = [p[8] for p in mod_path]

            self.type_helper_val[i, :len_path] = [p[9] for p in mod_path]
            self.expr_type_val[i, :len_path] = [p[10] for p in mod_path]
            self.ret_type_val[i, :len_path] = [p[11] for p in mod_path]

        return

    def save(self, path):
        with open(path + '/ast_apis.pickle', 'wb') as f:
            pickle.dump([self.nodes, self.edges, self.targets,
                         self.var_decl_ids,
                         self.varOrNot, self.typeOrNot, self.apiOrNot, self.symtabmod_or_not, self.op_or_not,
                         self.type_helper_val, self.expr_type_val, self.ret_type_val
                         ], f)

    def load_data(self, path):
        with open(path + '/ast_apis.pickle', 'rb') as f:
            [self.nodes, self.edges, self.targets, self.var_decl_ids,
             self.varOrNot, self.typeOrNot, self.apiOrNot, self.symtabmod_or_not, self.op_or_not,
             self.type_helper_val, self.expr_type_val, self.ret_type_val
             ] = pickle.load(f)
        return

    def truncate(self, sz):
        self.nodes = self.nodes[:sz, :self.max_depth]
        self.edges = self.edges[:sz, :self.max_depth]
        self.targets = self.targets[:sz, :self.max_depth]

        self.var_decl_ids = self.var_decl_ids[:sz, :self.max_depth]

        self.varOrNot = self.varOrNot[:sz, :self.max_depth]
        self.typeOrNot = self.typeOrNot[:sz, :self.max_depth]
        self.apiOrNot = self.apiOrNot[:sz, :self.max_depth]
        self.symtabmod_or_not = self.symtabmod_or_not[:sz, :self.max_depth]
        self.op_or_not = self.op_or_not[:sz, :self.max_depth]

        self.type_helper_val = self.type_helper_val[:sz, :self.max_depth]
        self.expr_type_val = self.expr_type_val[:sz, :self.max_depth]
        self.ret_type_val = self.ret_type_val[:sz, :self.max_depth]

        return

    def split(self, num_batches):
        # split into batches
        self.nodes = np.split(self.nodes, num_batches, axis=0)
        self.edges = np.split(self.edges, num_batches, axis=0)
        self.targets = np.split(self.targets, num_batches, axis=0)

        self.var_decl_ids = np.split(self.var_decl_ids, num_batches, axis=0)

        self.varOrNot = np.split(self.varOrNot, num_batches, axis=0)
        self.typeOrNot = np.split(self.typeOrNot, num_batches, axis=0)
        self.apiOrNot = np.split(self.apiOrNot, num_batches, axis=0)
        self.symtabmod_or_not = np.split(self.symtabmod_or_not, num_batches, axis=0)
        self.op_or_not = np.split(self.op_or_not, num_batches, axis=0)

        self.type_helper_val = np.split(self.type_helper_val, num_batches, axis=0)
        self.expr_type_val = np.split(self.expr_type_val, num_batches, axis=0)
        self.ret_type_val = np.split(self.ret_type_val, num_batches, axis=0)

        return

    def get(self):
        return self.nodes, self.edges, self.targets, \
               self.var_decl_ids, \
               self.varOrNot, self.typeOrNot, self.apiOrNot, self.symtabmod_or_not, self.op_or_not, \
               self.type_helper_val, self.expr_type_val, self.ret_type_val

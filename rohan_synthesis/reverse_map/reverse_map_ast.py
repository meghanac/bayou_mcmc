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
from program_helper.ast.ops import DAPIInvoke


class AstReverseMapper:
    def __init__(self, vocab):
        self.vocab = vocab

        self.nodes, self.edges, self.targets = [], [], []
        self.var_decl_ids = []
        self.varOrNot, self.typeOrNot, self.apiOrNot, self.symtabmod_or_not, self.op_or_not = [], [], [], [], []
        self.type_helper_val, self.expr_type_val, self.ret_type_val = [], [], []
        self.num_data = 0
        return

    def add_data(self, nodes, edges, targets,
                 var_decl_ids,
                 varOrNot, typeOrNot, apiOrNot, symtabmod_or_not,op_or_not,
                 type_helper_val, expr_type_val, ret_type_val):


        self.nodes.extend(nodes)
        self.edges.extend(edges)
        self.targets.extend(targets)
        self.var_decl_ids.extend(var_decl_ids)
        self.varOrNot.extend(varOrNot)
        self.typeOrNot.extend(typeOrNot)
        self.apiOrNot.extend(apiOrNot)
        self.symtabmod_or_not.extend(symtabmod_or_not)
        self.op_or_not.extend(op_or_not)
        self.type_helper_val.extend(type_helper_val)
        self.expr_type_val.extend(expr_type_val)
        self.ret_type_val.extend(ret_type_val)
        self.num_data += len(nodes)

    def get_element(self, id):
        return self.nodes[id], self.edges[id], self.targets[id], \
               self.var_decl_ids[id], \
               self.varOrNot[id], self.typeOrNot[id], self.apiOrNot[id], self.symtabmod_or_not[id], self.op_or_not[id], \
               self.type_helper_val[id], self.expr_type_val[id], self.ret_type_val[id]

    def decode_ast_paths(self, ast_element, partial=True):

        nodes, edges, targets, \
        var_decl_ids, \
        var_or_nots, type_or_nots, api_or_nots, symtab_or_nots, op_or_nots, \
        type_helper_vals, expr_type_vals, ret_type_vals = ast_element

        for node in nodes:
            print(self.vocab.chars_concept[node], end=',')
        print()
        #
        for edge in edges:
            print(edge, end=',')
        print()

        for _, _, target, \
            var_decl_id, \
            var_or_not, type_or_not, api_or_not, symtab_or_not, op_or_not, \
            type_helper_val, expr_type_val, ret_type_val in zip(*ast_element):
            if symtab_or_not:
                print('--symtab--', end=',')
            elif var_or_not:
                print(self.vocab.chars_var[target], end=',')
            elif type_or_not:
                print(self.vocab.chars_type[target], end=',')
            elif api_or_not:
                api = self.vocab.chars_api[target]
                api = api.split(DAPIInvoke.delimiter())[0]
                print(api, end=',')
            elif op_or_not:
                op = self.vocab.chars_op[target]
                print(op, end=',')
            else:
                print(self.vocab.chars_concept[target], end=',')
        print()

        if not partial:
            for var_decl_id in var_decl_ids:
                print(var_decl_id, end=',')
            print()

            for type_helper_val in type_helper_vals:
                print(self.vocab.chars_type[type_helper_val], end=',')
            print()

            for expr_type_val in expr_type_vals:
                print(self.vocab.chars_type[expr_type_val], end=',')
            print()

            for ret_type_val in ret_type_vals:
                print(self.vocab.chars_type[ret_type_val], end=',')
            print()

            for v in var_or_nots:
                print(int(v), end=',')
            print()

            for t in type_or_nots:
                print(int(t), end=',')
            print()

            for a in api_or_nots:
                print(int(a), end=',')
            print()

            for s in symtab_or_nots:
                print(int(s), end=',')
            print()

    def reset(self):
        self.nodes, self.edges, self.targets = [], [], []
        self.var_decl_ids = []
        self.varOrNot, self.typeOrNot, self.apiOrNot, self.symtabmod_or_not, self.op_or_not = [], [], [], [], []
        self.type_helper_val, self.expr_type_val, self.ret_type_val = [], [], []
        self.num_data = 0
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
from copy import deepcopy

from program_helper.ast.ops import DSubTree, DStop, DBranch, DExcept, DLoop, \
    DAPIInvoke, DVarDecl, DFieldDecl, DVarAccess, DClsInit, DVarAssign, DSymtabMod, CHILD_EDGE, SIBLING_EDGE, Node
from program_helper.ast.ops.concepts.DExceptionVarDecl import DExceptionVarDecl
from program_helper.ast.ops.concepts.DInfix import DInfix
from program_helper.ast.ops.concepts.DReturnVar import DReturnVar
from program_helper.ast.parser.ast_exceptions import UnknownVarAccessException, IgnoredForNowException, \
    NestedAPIParsingException


class AstParser:
    def __init__(self):
        self.head = None
        self.global_var_id = 0
        return

    def get_ast_with_memory(self, js, symtab, fp_type_head=None, field_head=None, repair_mode=True):
        '''

        :param js: actual json
        :param symtab: dictionary of AST Nodes from ret type, field and fps
        :param fp_type_head: Formal param head
        :param field_head: Field Type Head
        :param repair_mode: repair mode ensures you delete unused variables
        :return:
        '''
        # new_js = self.preprocess(js)
        self.head = self.form_ast(js, symtab=symtab)
        if repair_mode:
            self.skip_invalid(self.head)
            self.skip_invalid_vars(fp_type_head)
            self.skip_invalid_vars(field_head)
        defaults = {'system_package': 'system_package', 'LITERAL': 'LITERAL',
                    'no_return_var': 'no_return_var'}
        fp_dict = self.handle_fp_vars(fp_type_head)
        field_dict = self.handle_field_vars(field_head)
        defaults.update(fp_dict)
        defaults.update(field_dict)
        self.handle_ast_vars(self.head, input_dict=defaults,
                             repair_mode=repair_mode)
        return self.head

    def form_ast(self, js, idx=0, symtab=None):
        if symtab is None:
            symtab = dict()

        i = idx
        head = curr_node = DSubTree()
        while i < len(js):
            new_node = self.create_new_nodes(js[i], symtab)
            # new nodes are only returned if it is not a last_DAPICall node
            # if it is a last_DAPICall node, no new nodes are returned, but
            # already attached to last api node
            curr_node = curr_node.add_and_progress_sibling_node(new_node)
            i += 1

        curr_node.add_sibling_node(DStop())

        head.child = head.sibling
        head.sibling = None

        return head

    def create_new_nodes(self, node_js, symtab):
        node_type = node_js['node']

        if node_type == 'DAPIInvoke':
            new_node = DAPIInvoke(node_js, symtab)
        elif node_type == 'DVarDecl':
            new_node = DVarDecl(node_js, symtab)
        elif node_type == 'DExceptionVar':
            new_node = DExceptionVarDecl(node_js, symtab)
        elif node_type == 'DClsInit':
            new_node = DClsInit(node_js, symtab)
        elif node_type == 'DFieldCall':
            new_node = DFieldDecl(node_js, symtab)
        elif node_type == 'DReturnVar':
            new_node = DReturnVar(node_js, symtab)
        elif node_type == 'DAssign':
            raise IgnoredForNowException
            node_js_varcall = {'_id': node_js['_from'], '_returns': node_js['_type']}
            new_node = DVarDecl(node_js_varcall, symtab)
            new_nodes.append(new_node)
            new_node = DVarAssign(node_js, symtab)
        # Split operation node types
        elif node_type == 'DBranch':
            new_node, symtab = self.read_DBranch(node_js, symtab)
        elif node_type == 'DExcept':
            new_node, symtab = self.read_DExcept(node_js, symtab)
        elif node_type == 'DLoop':
            new_node, symtab = self.read_DLoop(node_js, symtab)
        elif node_type == 'DInfix':
            new_node = self.read_DInfix(node_js, symtab)
        # Else throw exception
        else:
            print(node_type)
            raise Exception("Node type unknown")

        return new_node

    def read_DInfix(self, js_infix, symtab):
        nodeIn = DInfix(js_infix)
        nodeL = self.form_ast(js_infix['_left'], symtab=symtab)  # will have at most 1 "path"
        nodeL.val = 'DLeft'
        nodeR = self.form_ast(js_infix['_right'], symtab=symtab)
        nodeR.val = 'DRight'
        nodeL.add_sibling_node(nodeR)
        nodeIn.child.add_sibling_node(nodeL)
        return nodeIn

    def read_DLoop(self, js_branch, symtab):
        old_symtab = deepcopy(symtab)
        # assert len(pC) <= 1
        nodeC = self.form_ast(js_branch['_cond'], symtab=symtab)  # will have at most 1 "path"
        nodeC.val = 'DCond'
        nodeB = self.form_ast(js_branch['_body'], symtab=symtab)
        nodeB.val = 'DBody'
        nodeC.add_sibling_node(nodeB)
        return DLoop(child=nodeC), old_symtab

    def read_DExcept(self, js_branch, symtab):
        old_symtab = deepcopy(symtab)
        nodeT = self.form_ast(js_branch['_try'], symtab=symtab)
        nodeT.val = 'DTry'
        nodeC = self.form_ast(js_branch['_catch'], symtab=symtab)
        nodeC.val = 'DCatch'

        nodeT.add_sibling_node(nodeC)
        return DExcept(child=nodeT), old_symtab

    def read_DBranch(self, js_branch, symtab):
        old_symtab = deepcopy(symtab)
        nodeC = self.form_ast(js_branch['_cond'], symtab=symtab)  # will have at most 1 "path"
        nodeC.val = 'DCond'
        freeze_symtab = deepcopy(symtab)
        # assert len(pC) <= 1
        nodeT = self.form_ast(js_branch['_then'], symtab=symtab)
        nodeT.val = 'DThen'
        nodeE = self.form_ast(js_branch['_else'], symtab=freeze_symtab)
        nodeE.val = 'DElse'
        nodeT.add_sibling_node(nodeE)
        nodeC.add_sibling_node(nodeT)
        return DBranch(child=nodeC), old_symtab

    def handle_fp_vars(self, head, input_dict=None, var_count=0):

        ## If a variable was declared and was valid, it must not have been re-declared, unless in a different
        ## context, for example inside branching
        ## symtab mod can only occur with var declaration, hence its dependent var_decl_id is covered by this
        ## # since system package is 0 by default, length is the value
        if input_dict is None:
            input_dict = dict()

        node = head.child
        while node is not None:
            if isinstance(node, DVarDecl) and node.valid is True:
                input_dict[node.var_id] = 'fp_' + str(var_count)
                var_count = var_count + 1
            node.var_decl_id = var_count
            node = node.sibling
        return input_dict

    def handle_field_vars(self, head, input_dict=None, var_count=0):

        ## If a variable was declared and was valid, it must not have been re-declared, unless in a different
        ## context, for example inside branching
        ## symtab mod can only occur with var declaration, hence its dependent var_decl_id is covered by this
        ## # since system package is 0 by default, length is the value
        if input_dict is None:
            input_dict = dict()

        node = head.child
        while node is not None:
            if isinstance(node, DVarDecl) and node.valid is True:
                input_dict[node.var_id] = 'field_' + str(var_count)
                var_count = var_count + 1
            node.var_decl_id = var_count
            node = node.sibling
        return input_dict


    def handle_ast_vars(self, node, input_dict=None, var_count=0,
                        repair_mode=True):

        if node is None:
            return

        ## If a variable was declared and was valid, it must not have been re-declared, unless in a different
        ## context, for example inside branching
        ## symtab mod can only occur with var declaration, hence its dependent var_decl_id is covered by this
        ## # since system package is 0 by default, length is the value
        if isinstance(node, DVarDecl) and (node.valid is True or repair_mode is False):
            input_dict[node.var_id] = 'var_' + str(var_count)
            var_count = var_count + 1
            if repair_mode is False:
                node.make_valid()

        node.var_decl_id = var_count

        if isinstance(node, DVarAccess):
            if node.val not in input_dict:
                raise UnknownVarAccessException
            node.val = input_dict[node.val]

        ## child can declare new variables without bothering the sibling
        self.handle_ast_vars(node.child,
                             input_dict=deepcopy(input_dict),
                             var_count=var_count,
                             repair_mode=repair_mode
                             )

        ## sibling shares the same dictionary as current node
        self.handle_ast_vars(node.sibling,
                             input_dict=deepcopy(input_dict),
                             var_count=var_count,
                             repair_mode=repair_mode
                             )

    def skip_invalid_vars(self, head):
        node = head.child
        last_node = head
        last_edge = CHILD_EDGE
        while node is not None:
            assert node.val in [DVarDecl.name(),
                                DStop.name()]
            ## If invalid make connection with last node
            if node.valid is False:
                if last_edge is CHILD_EDGE:
                    last_node.child = node.sibling
                else:
                    last_node.sibling = node.sibling
            else:
                ## Update last node
                last_node = node
                last_edge = SIBLING_EDGE

            ## anyway node has to progress
            node = node.sibling

        return

    def skip_invalid(self, head):
        node = head.child
        last_node = head
        last_edge = CHILD_EDGE
        while node is not None:
            assert node.val in [DAPIInvoke.name(), DClsInit.name(), DVarDecl.name(), DExceptionVarDecl.name(),
                                DStop.name(),
                                DVarAssign.name(),
                                DBranch.name(), DLoop.name(), DExcept.name(),
                                DInfix.name(), DReturnVar.name()]
            ## If invalid make connection with last node
            if node.valid is False:
                if last_edge is CHILD_EDGE:
                    last_node.child = node.sibling
                else:
                    last_node.sibling = node.sibling
            else:
                ## Update last node
                last_node = node
                last_edge = SIBLING_EDGE

            if node.val == DBranch.name():
                self.skip_invalid(node.child)
                self.skip_invalid(node.child.sibling)
                self.skip_invalid(node.child.sibling.sibling)

            if node.val == DLoop.name():
                self.skip_invalid(node.child)
                self.skip_invalid(node.child.sibling)

            if node.val == DExcept.name():
                self.skip_invalid(node.child)
                self.skip_invalid(node.child.sibling)

            if node.val == DInfix.name():
                self.skip_invalid(node.child.sibling)
                self.skip_invalid(node.child.sibling.sibling)

            ## anyway node has to progress
            node = node.sibling

        return

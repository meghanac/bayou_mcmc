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

from program_helper.ast.ops import DAPICall, DBranch, DLoop, DVarAccess, DSymtabMod, SINGLE_PARENTS, DVarDecl, \
    DAPIInvoke, DExcept, CONTROL_FLOW_NAMES, DClsInit, DVarAssign, DStop, DType, DAPICallMulti, DAPICallSingle
from program_helper.ast.parser.ast_checker import AstChecker
from program_helper.ast.parser.ast_exceptions import \
    VoidProgramException, UndeclaredVarException, TypeMismatchException, \
    ConceptMismatchException
from program_helper.ast.parser.ast_traverser import AstTraverser


class AstGenChecker(AstChecker):

    def __init__(self, vocab,
                 max_loop_num=None,
                 max_branching_num=None,
                 max_variables=None,
                 compiler=None,
                 logger=None):

        super().__init__(max_loop_num, max_branching_num, max_variables)
        self.ast_traverser = AstTraverser()
        self.vocab = vocab
        self.logger = logger
        self.java_compiler = compiler
        self.reset_stats()
        return

    def reset_stats(self):
        self.total, self.passed_count, \
        self.void_count, \
        self.undeclared_var_count, self.type_mismatch_count, self.concept_mismatch_count = 0, 0, 0, 0, 0, 0

    def check_generated_progs(self, ast_head, init_symtab=None,
                              update_mode=False):
        if init_symtab is None:
            init_symtab = dict()

        if not update_mode:
            self.check_void_programs(ast_head)

        self.variable_count = 0
        init_symtab.update({'system_package': 0, 'LITERAL': 0, 'no_return_var': 0})

        self.dynamic_type_var_checker(ast_head, symtab=init_symtab, update_mode=update_mode)
        return

    def get_fp_symtabs(self, fp_head, symtab=None):
        if symtab is None:
            symtab = dict()

        fp_path = self.ast_traverser.depth_first_search(fp_head)

        j = 0
        for item in fp_path:
            curr_node_val, curr_node_type, curr_node_validity, curr_node_var_decl_ids, \
            parent_node_id, \
            edge_type, expr_type, type_helper, return_type = item
            if curr_node_type == DType.name():
                symtab['fp_' + str(j)] = curr_node_val  # TODO self.vocab[curr_node_val]
                j += 1
        return symtab

    def check_existence_in_symtab(self, id, symtab, name=None, log_location=None):

        if self.java_compiler.var_violation_check(id, symtab):
            if name is not None:
                self.failure_spot += 'Location :: {}'.format(log_location)
                self.failure_spot += '\nWhile synthesizing API {}'.format(name)
                self.failure_spot += '\nId {} does not exist in symtab'.format(id)
            raise UndeclaredVarException
        return

    def check_type_in_symtab(self, id, symtab,
                             ret_type=None,
                             expr_type=None,
                             type_helper=None,
                             log_api_name=None,
                             log_location=None,
                             update_mode=False
                             ):

        assert not all([ret_type, expr_type, type_helper]) is None

        if ret_type is not None and self.java_compiler.type_violation_check(id, symtab, ret_type,
                                                                            var_type='ret_type',
                                                                            update_mode=update_mode):
            if log_api_name is not None:
                self.failure_spot += 'Location :: {}'.format(log_location)
                self.failure_spot += '\nWhile synthesizing API {}'.format(log_api_name)
                self.failure_spot += '\nRet Type {} does not match symtab id with type {}:{}'.format(ret_type, id,
                                                                                                     symtab[id])
            raise TypeMismatchException

        if expr_type is not None and self.java_compiler.type_violation_check(id, symtab, expr_type,
                                                                             var_type='expr_type',
                                                                             update_mode=update_mode):
            if log_api_name is not None:
                self.failure_spot += 'Location :: {}'.format(log_location)
                self.failure_spot += '\nWhile synthesizing API {}'.format(log_api_name)
                self.failure_spot += '\nExp Type {} does not match symtab id with type {}:{}'.format(expr_type, id,
                                                                                                     symtab[id])
            raise TypeMismatchException

        if type_helper is not None and self.java_compiler.type_violation_check(id, symtab, type_helper,
                                                                               var_type='type_helper',
                                                                               update_mode=update_mode):
            if log_api_name is not None:
                self.failure_spot += 'Location :: {}'.format(log_location)
                self.failure_spot += '\nWhile synthesizing API {}'.format(log_api_name)
                self.failure_spot += '\nType Helper {} does not match symtab id with type {}:{}'.format(type_helper, id,
                                                                                                        symtab[id])
            raise TypeMismatchException

        return

    def dynamic_type_var_checker(self, head, symtab=None, update_mode=False):

        if head is None:
            return

        if head.type in SINGLE_PARENTS:
            self.dynamic_type_var_checker(head.child,
                                          symtab=symtab,
                                          update_mode=update_mode
                                          )
            return

        if symtab is None:
            symtab = {}

        node = head
        while node is not None:
            if isinstance(node, DVarDecl):
                num_elems = 0
                for item in symtab.keys():
                    if 'var_' in item:
                        num_elems += 1
                self.variable_count = num_elems
                id = "var_" + str(self.variable_count)
                node.set_var_id(id)

                symtab[node.get_var_id()] = node.get_return_type()

            elif isinstance(node, DClsInit):
                ret_id = node.get_return_id()

                self.check_existence_in_symtab(ret_id, symtab)
                # self.check_type_in_symtab(ret_id, symtab)
                # if call_val != node.get_return_type():
                #     raise TypeMismatchException

            elif isinstance(node, DAPIInvoke):
                # First check for used vars
                expr_id = node.get_expr_id()
                ret_id = node.get_return_id()
                # ret_type, expr_type = None, None
                if not isinstance(node.child, DAPICallMulti):
                    self.failure_spot += 'DAPIInvoke->DAPICallMulti production rule was violated'
                    raise ConceptMismatchException
                else:
                    singlecallnode = node.child.child
                    if not isinstance(singlecallnode, DAPICallSingle):
                        self.failure_spot += 'DAPICallMulti->DAPICallSingle production rule was violated'
                        raise ConceptMismatchException

                    api_node = singlecallnode.child
                    api_call = api_node.val
                    _, expr_type, ret_type = DAPIInvoke.split_api_call(api_call)

                    while not isinstance(singlecallnode, DStop):
                        if not isinstance(singlecallnode, DAPICallSingle):
                            self.failure_spot += 'DAPICallMulti->DAPICallSingle production rule was violated'
                            raise ConceptMismatchException
                        api_node = singlecallnode.child
                        api_call = api_node.val
                        _, _, ret_type = DAPIInvoke.split_api_call(api_call)

                        if '(' not in api_call:
                            self.failure_spot += 'Bracket missing in API Call'
                            raise ConceptMismatchException

                        arg_list = DAPICallMulti.get_formal_types_from_data(api_call)
                        arg_id = 0

                        start = singlecallnode.child.sibling
                        while start is not None:
                            fp_id = start.val
                            self.check_existence_in_symtab(fp_id, symtab, name=singlecallnode.get_api_name(),
                                                           log_location='formal param id : ' + str(arg_id))

                            self.check_type_in_symtab(fp_id, symtab, type_helper=arg_list[arg_id],
                                                      log_api_name=singlecallnode.get_api_name(),
                                                      log_location='formal param id : ' + str(arg_id),
                                                      update_mode=update_mode)
                            start = start.sibling
                            arg_id += 1

                        singlecallnode = singlecallnode.sibling

                self.check_existence_in_symtab(expr_id, symtab, name=node.get_api_name(), log_location='expr_id')
                self.check_type_in_symtab(expr_id, symtab, expr_type=expr_type,
                                          log_api_name=node.get_api_name(),
                                          log_location='expr id',
                                          update_mode=update_mode
                                          )

                self.check_existence_in_symtab(ret_id, symtab, name=node.get_api_name(), log_location='ret_id')
                self.check_type_in_symtab(ret_id, symtab, ret_type=ret_type,
                                          log_api_name=node.get_api_name(),
                                          log_location='ret id',
                                          update_mode=update_mode
                                          )

            elif node.name() in CONTROL_FLOW_NAMES:
                self.handle_control_flow(node, symtab, update_mode=update_mode)


            elif isinstance(node, DStop):
                pass

            node = node.sibling

        return

    def handle_control_flow(self, node, symtab, update_mode=False):
        # control flow does not impact the symtab

        if isinstance(node, DBranch):
            temp = deepcopy(symtab)
            self.dynamic_type_var_checker(node.child, symtab=temp, update_mode=update_mode)
            self.dynamic_type_var_checker(node.child.sibling, symtab=symtab, update_mode=update_mode)
            self.dynamic_type_var_checker(node.child.sibling.sibling, symtab, update_mode=update_mode)

        elif isinstance(node, DLoop):
            self.dynamic_type_var_checker(node.child, symtab=symtab, update_mode=update_mode)
            self.dynamic_type_var_checker(node.child.sibling, symtab=symtab, update_mode=update_mode)

        elif isinstance(node, DExcept):
            self.dynamic_type_var_checker(node.child, symtab=symtab, update_mode=update_mode)
            self.dynamic_type_var_checker(node.child.sibling, symtab=symtab, update_mode=update_mode)

        return

    def run_viability_check(self, ast_candies,
                            fp_types=None,
                            ret_type=None,
                            field_vals=None,
                            debug_print=True):

        outcome_strings = []
        for ast_candy in ast_candies:
            ast_node = ast_candy.head
            outcome_string = self.check_single_program(ast_node, fp_types=fp_types,
                                                       ret_type=ret_type,
                                                       field_vals=field_vals)

            if debug_print:
                self.logger.info(outcome_string)

            outcome_strings.append(outcome_string)

        return outcome_strings

    def check_single_program(self, ast_node,
                             fp_types=None,
                             ret_type=None,
                             field_vals=None,
                             update_mode=False):

        passed = False
        self.failure_spot = ''

        init_symtab = dict()
        for j, val in enumerate(field_vals):
            key = 'field_' + str(j)
            init_symtab.update({key: val})

        for j, val in enumerate(fp_types):
            key = 'fp_' + str(j)
            init_symtab.update({key: val})


        try:
            self.check_generated_progs(ast_node, init_symtab=init_symtab,
                                       update_mode=update_mode)
            passed, fail_reason = True, ''
        except VoidProgramException:
            self.void_count += 1 if not update_mode else 0
            fail_reason = 'is void'
        except UndeclaredVarException:
            self.undeclared_var_count += 1 if not update_mode else 0
            fail_reason = 'has undeclared var'
        except TypeMismatchException:
            self.type_mismatch_count += 1 if not update_mode else 0
            fail_reason = 'has mismatched type'
        except ConceptMismatchException:
            self.concept_mismatch_count += 1 if not update_mode else 0
            fail_reason = 'has mismatched concept'

        outcome_string = 'This program passed' if passed else 'This program failed because it ' + fail_reason
        outcome_string += '\n' + self.failure_spot

        if not update_mode:
            self.passed_count += 1 if 'passed' in outcome_string else 0
            self.total += 1

        return outcome_string

    def print_stats(self, logger=None):
        self.logger.info('')
        self.logger.info('{:8d} programs/asts in total'.format(self.total))
        self.logger.info('{:8d} programs/asts missed for being void'.format(self.void_count))
        self.logger.info('{:8d} programs/asts missed for illegal var access'.format(self.undeclared_var_count))
        self.logger.info('{:8d} programs/asts missed for type mismatch'.format(self.type_mismatch_count))
        self.logger.info('{:8d} programs/asts missed for concept mismatch'.format(self.concept_mismatch_count))
        self.logger.info('{:8d} programs/asts passed'.format(self.passed_count))

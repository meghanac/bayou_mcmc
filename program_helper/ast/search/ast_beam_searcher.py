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
import numpy as np
from copy import deepcopy

from program_helper.ast.ops.concepts.DExceptionVarDecl import DExceptionVarDecl
from program_helper.ast.ops.concepts.DInfix import DInfix, DLeft, DRight
from program_helper.ast.ops.concepts.DReturnVar import DReturnVar
from synthesis.ops.candidate_ast import Candidate_AST, CONCEPT_NODE, VAR_NODE, API_NODE, TYPE_NODE, SYMTAB_MOD, OP_NODE

from program_helper.ast.ops import CHILD_EDGE, SIBLING_EDGE, \
    DBranch, DCond, DThen, DElse, DExcept, DTry, DLoop, DBody, \
    DSubTree, DStop, DAPIInvoke, DVarDecl, DClsInit, \
    DVarAssign, DCatch, DAPICallMulti, DAPICallSingle, SINGLE_PARENTS
from synthesis.ops.candidate_element import CandidateElement

MAX_LENGTH = 40


class TreeBeamSearcher:

    def __init__(self, infer_model):
        self.infer_model = infer_model
        self.beam_width = infer_model.config.batch_size

        return

    def resolve_curr_node(self, candies):
        inputs = []
        for candy in candies:
            val = self.infer_model.config.vocab.concept_dict[candy.curr_node_val]
            inputs.append([val])
        return inputs

    def resolve_next_node(self, candies, all_beam_ids):
        # states = states[0]
        next_nodes = []
        for k, candy in enumerate(candies):
            next_node_type = candy.next_node_type
            beam = all_beam_ids[next_node_type][k]
            if next_node_type == CONCEPT_NODE:
                possible_next_nodes = [self.infer_model.config.vocab.chars_concept[idx] for idx in beam]
            elif next_node_type == API_NODE:
                possible_next_nodes = [self.infer_model.config.vocab.chars_api[idx] for idx in beam]
            elif next_node_type == TYPE_NODE:
                possible_next_nodes = [self.infer_model.config.vocab.chars_type[idx] for idx in beam]
            elif next_node_type == VAR_NODE:
                possible_next_nodes = [self.infer_model.config.vocab.chars_var[idx] for idx in beam]
            elif next_node_type == OP_NODE:
                possible_next_nodes = [self.infer_model.config.vocab.chars_op[idx] for idx in beam]
            elif next_node_type == SYMTAB_MOD:
                possible_next_nodes = ['-symtab-' for _ in beam]
            else:
                raise Exception
            next_nodes.append(possible_next_nodes)
        return next_nodes

    def resolve_beam_probs(self, candies, all_beam_ln_probs):
        # if a candidate has ended operation, it will not be expanding further
        beam_ln_probs = [[0.0 if j == 0 else -np.inf for j in range(len(candies))]
                         if candy.is_not_rolling() or candy.next_node_type == SYMTAB_MOD
                         else all_beam_ln_probs[candy.next_node_type][k] for
                         k, candy in enumerate(candies)]

        NORMALIZING_FACTOR = 0.7
        # Length normalization
        length = np.array([candy.length for candy in candies])

        # Old probs
        old_probs = np.array([candy.log_probability for candy in candies])[:, None] \
                    * np.power(length[:, None] , NORMALIZING_FACTOR)

        # add to the existing candidate probs
        len_norm_probs = (beam_ln_probs + old_probs) / np.power(length[:, None] + 1, NORMALIZING_FACTOR)

        return len_norm_probs

    def beam_search(self, initial_state=None,
                    ret_type=None,
                    formal_params=None,
                    field_types=None,
                    probs=None):

        symtab = self.infer_model.get_initial_symtab()
        candies = [Candidate_AST(initial_state[k], symtab[k], ret_type, formal_params,
                                 field_types, probs[k])
                   for k in range(self.beam_width)]

        while not self.check_for_all_stop(candies):
            candies = self.get_next_output_with_fan_out(candies)

        # [candy.print_symtab() for candy in candies]
        [candy.force_finish() for candy in candies]
        candies.sort(key=lambda x: x.log_probability, reverse=True)
        # for candy in candies:
        #     candy.debug_print(self.infer_model.config.vocab.type_dict)
        return candies

    def check_for_all_stop(self, candies):
        return all([not candy.is_rolling() for candy in candies])

    def get_next_output_with_fan_out(self, candies):

        curr_node = self.resolve_curr_node(candies)
        curr_edge = [[candy.curr_edge] for candy in candies]
        states = np.transpose(np.array([candy.state for candy in candies]), [1, 0, 2])

        # states is still top_k * LSTM_Decoder_state_size
        # next_node is top_k * top_k
        # node_probs in  top_k * top_k
        # log_probability is top_k
        # [candy.add_to_storage() for candy in candies]
        states, symtab, all_beam_ids, all_beam_ln_probs = self.infer_model. \
            get_next_ast_state(curr_node, curr_edge, states,
                               candies)

        # Update next nodes, next nodes are real values like 'java.lang.String'
        next_nodes = self.resolve_next_node(candies, all_beam_ids)
        # Update next probs
        beam_ln_probs = self.resolve_beam_probs(candies, all_beam_ln_probs)

        top_k = len(candies)
        rows, cols = np.unravel_index(np.argsort(beam_ln_probs, axis=None)[::-1], (top_k, top_k))
        rows, cols = rows[:top_k], cols[:top_k]

        # rows mean which of the original candidate was finally selected
        new_candies = []
        for row, col in zip(rows, cols):
            new_candy = deepcopy(candies[row])  # candies[row].copy()
            if new_candy.is_rolling():
                new_candy.state = [states[k][row] for k in range(len(states))]
                new_candy.symtab = symtab[row]
                new_candy.log_probability = beam_ln_probs[row][col]

                # Like next_nodes, value2add is also real values, like 'Float'
                value2add = next_nodes[row][col]
                # CHECK LENGTH, DOES IMPACT ROLLING
                new_candy.length_mod_and_check(value2add, MAX_LENGTH)
                if new_candy.is_rolling():
                    '''
                        Now lets make sure if a concept node is being predicted 
                        and fixed node val is there we do not end up not choosing it
                    '''
                    # TODO: the probabilites here are however from wrong concept choices
                    if new_candy.next_node_type == CONCEPT_NODE and new_candy.fixed_next_node is not None:
                        value2add = new_candy.fixed_next_node
                        new_candy.fixed_next_node = None

                    # ADD THE NODE
                    new_candy = new_candy.add_node(value2add)



                    # This can only occur inside a DVarDecl, since type node is never accessed otherwise
                    # Ret type is changed fot the next symtab update
                    # Also update your internal symtab
                    if new_candy.next_node_type == TYPE_NODE:
                        type_val = self.infer_model.config.vocab.type_dict[value2add]
                        assert len(new_candy.type_helper_val_queue) == 0
                        new_candy.type_helper_val_queue.insert(0, type_val)

                    # This can only occur inside a DAPIInvoke or DClsInit, since API_NODE is never accessed otherwise
                    # That will trigger generation of updated ret_type and expr_type and FPs
                    if new_candy.next_node_type == API_NODE:
                        fps = DAPICallMulti.get_formal_types_from_data(value2add)
                        self.handle_DAPICallSingle_dynamic(new_candy, fps)

                        value2add, expr_type, ret_type = DAPIInvoke.split_api_call(value2add)
                        expr_type_val = self.infer_model.config.vocab.type_dict[expr_type]
                        ret_type_val = self.infer_model.config.vocab.type_dict[ret_type]

                        # Nested API Call, Normal API Call, ClsInit, always change ret_type_val
                        new_candy.ret_type_val_queue = [ret_type_val]

                        # For Nested API Call, no change,
                        # for ClsInit expr type val is DELIM hence no change
                        if len(new_candy.expr_type_val_queue) == 0 and expr_type_val != 0:
                            new_candy.expr_type_val_queue = [expr_type_val]

                    if new_candy.next_node_type == CONCEPT_NODE and value2add == DReturnVar.name():
                        new_candy.ret_type_val_queue = [new_candy.return_type]

                    if new_candy.next_node_type == CONCEPT_NODE:
                        new_candy.curr_node_val = value2add

                    new_candy = self.handle_control_flow(new_candy)
                else:
                    new_candy = new_candy.add_node(DStop.name())

            new_candies.append(new_candy)

        return new_candies

    def handle_control_flow(self, new_candy):
        node_value = new_candy.tree_currNode.val
        # Modify the control flows as appropriate
        if node_value == DAPIInvoke.name():
            new_candy.single_api_count = 0
            new_candy = self.handle_DAPIInvoke(new_candy)
        elif node_value in [DVarDecl.name(), DExceptionVarDecl.name()]:
            new_candy = self.handle_DVarDecl(new_candy, node_value=node_value)
        elif node_value == DClsInit.name():
            new_candy = self.handle_DClsInit(new_candy)
        elif node_value == DVarAssign.name():
            new_candy = self.handle_DVarAssign(new_candy)
        # Multi API Invoke
        elif node_value == DAPICallSingle.name():
            new_candy = self.handle_DAPICallSingle(new_candy)
        # split ops
        elif node_value == DBranch.name():
            new_candy = self.handle_DBranch(new_candy)
        elif node_value == DExcept.name():
            new_candy = self.handle_DExcept(new_candy)
        elif node_value == DLoop.name():
            new_candy = self.handle_DLoop(new_candy)
        # Delims
        elif node_value == DInfix.name():
            new_candy = self.handle_DInfix(new_candy)
        elif node_value == DReturnVar.name():
            new_candy = self.handle_DReturnVar(new_candy)
        elif node_value == DAPICallMulti.name():
            new_candy = self.handle_DAPICallMulti(new_candy)
        elif node_value in SINGLE_PARENTS:
            new_candy = self.handle_single_child(new_candy)
        '''
        Finally evaluate the next node
        '''
        new_candy = self.handle_next_node(new_candy)
        return new_candy

    def handle_DAPICallSingle_dynamic(self, new_candy, fps):
        edge_path = [SIBLING_EDGE for _ in range(len(fps))]
        # Note that new_candy.tree_currNode still points to DAPICall Node

        for j, fp in enumerate(fps[::-1]):
            fp_ = self.infer_model.config.vocab.type_dict[fp]
            new_candy.type_helper_val_queue.append(fp_)

            new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                                 curr_node_val=DAPICallSingle.name(),
                                                                 edge_path=deepcopy(edge_path),
                                                                 next_node_type=VAR_NODE,
                                                                 )
                                                )
            edge_path.pop()

        return

    def handle_DAPICallSingle(self, new_candy):
        if not new_candy.stop_apisingle and new_candy.single_api_count <= 5:
            # last we will expand the elements after the first call for last_DAPICall invocations
            new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                                 curr_node_val=DAPICallSingle.name(),
                                                                 edge_path=[SIBLING_EDGE],
                                                                 next_node_type=CONCEPT_NODE
                                                                 )
                                                )
            new_candy.single_api_count += 1
        else:
            new_candy.stop_apisingle = False

        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DAPICallSingle.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=API_NODE
                                                             )
                                            )
        return new_candy

    def handle_DInfix(self, new_candy):
        # last we will expand the elements after the first call for last_DAPICall invocations
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DInfix.name(),
                                                             edge_path=[SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )

        # For the right branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DLeft.name(),
                                                             fixed_node_val=DRight.name(),
                                                             edge_path=[CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )

        # For the left branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DInfix.name(),
                                                             fixed_node_val=DLeft.name(),
                                                             edge_path=[CHILD_EDGE, SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )

        # Right now we will evaluate the op node
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DInfix.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=OP_NODE
                                                             )
                                            )
        return new_candy

    def handle_DBranch(self, new_candy):
        # last we will expand the elements after branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DBranch.name(),
                                                             edge_path=[SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE,
                                                             var_decl_id_val=new_candy.var_decl_id
                                                             )
                                            )

        # This is for the else branch
        # TODO: Else branch needs to get the var decl id from output of If
        new_candy.control_flow_stack.append(
            CandidateElement(curr_node=new_candy.tree_currNode,
                             curr_node_val=DThen.name(),
                             fixed_node_val=DElse.name(),
                             edge_path=[CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE],
                             next_node_type=CONCEPT_NODE,
                             # var_decl_id_val=new_candy.var_decl_id
                             )
        )
        # This is for the then branch
        new_candy.control_flow_stack.append(
            CandidateElement(curr_node=new_candy.tree_currNode,
                             curr_node_val=DCond.name(),
                             fixed_node_val=DThen.name(),
                             edge_path=[CHILD_EDGE, SIBLING_EDGE],
                             next_node_type=CONCEPT_NODE)
        )

        # Right now we will evaluate the op node
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DBranch.name(),
                                                             fixed_node_val=DCond.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )
        return new_candy

    def handle_DExcept(self, new_candy):
        # last we will expand the elements after except
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DExcept.name(),
                                                             edge_path=[SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE,
                                                             var_decl_id_val=new_candy.var_decl_id
                                                             )
                                            )

        # This is for the catch branch
        new_candy.control_flow_stack.append(CandidateElement(
            curr_node=new_candy.tree_currNode,
            curr_node_val=DTry.name(),
            fixed_node_val=DCatch.name(),
            edge_path=[CHILD_EDGE, SIBLING_EDGE],
            next_node_type=CONCEPT_NODE,
            var_decl_id_val=new_candy.var_decl_id
        )
        )

        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DExcept.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             fixed_node_val=DTry.name(),
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )

        return new_candy

    def handle_DLoop(self, new_candy):
        # last we will expand the elements after loop
        new_candy.control_flow_stack.append(CandidateElement(
            curr_node=new_candy.tree_currNode,
            curr_node_val=DLoop.name(),
            edge_path=[SIBLING_EDGE],
            next_node_type=CONCEPT_NODE,
            var_decl_id_val=new_candy.var_decl_id
        )
        )
        # This is for the body branch
        new_candy.control_flow_stack.append(
            CandidateElement(curr_node=new_candy.tree_currNode,
                             curr_node_val=DCond.name(),
                             fixed_node_val=DBody.name(),
                             edge_path=[CHILD_EDGE, SIBLING_EDGE],
                             next_node_type=CONCEPT_NODE
                             )
        )
        # Right now we will evaluate the cond branch, which starts as a DSubTree child call
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DLoop.name(),
                                                             fixed_node_val=DCond.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )
        return new_candy

    # DAPIInvoke is a concept
    def handle_DAPIInvoke(self, new_candy):
        # last we will expand the elements after invoke
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DAPIInvoke.name(),
                                                             edge_path=[SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )

        # This is for the Expression VarAccess branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=None, # None since we want to reuse last concept
                                                             edge_path=[CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE],
                                                             next_node_type=VAR_NODE,
                                                             )
                                            )
        # This is for the Return VarAccess branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=None, # None since we want to reuse last concept
                                                             edge_path=[CHILD_EDGE, SIBLING_EDGE],
                                                             next_node_type=VAR_NODE,
                                                             )
                                            )
        # Right now we will evaluate the next DAPICallMulti branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DAPIInvoke.name(),
                                                             fixed_node_val=DAPICallMulti.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )
        return new_candy

    def handle_DVarDecl(self, new_candy, node_value=DVarDecl.name()):
        # last we will expand the elements after invoke
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=node_value,
                                                             edge_path=[SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE,
                                                             )
                                            )

        # This is for the DSymtabMod branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=None, # None since we want to reuse last concept
                                                             edge_path=[CHILD_EDGE, SIBLING_EDGE],
                                                             next_node_type=SYMTAB_MOD
                                                             )
                                            )

        # Right now we will evaluate the Dtype branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=node_value,
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=TYPE_NODE
                                                             )
                                            )

        new_candy.var_decl_id += 1
        return new_candy


    def handle_DReturnVar(self, new_candy):
        # Right now we will evaluate the DVarAccess node
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DReturnVar.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=VAR_NODE
                                                             )
                                            )
        return new_candy

    def handle_DAPICallMulti(self, new_candy):
        # Right now we will evaluate the DVarAccess node
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DAPICallMulti.name(),
                                                             fixed_node_val=DAPICallSingle.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )
        return new_candy


    def handle_DClsInit(self, new_candy):
        # last we will expand the elements after invoke
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DClsInit.name(),
                                                             edge_path=[SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )

        # This is for the DVarAccess branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=None, # None since we want to reuse last concept
                                                             edge_path=[CHILD_EDGE, SIBLING_EDGE],
                                                             next_node_type=VAR_NODE
                                                             )
                                            )
        # Right now we will evaluate the DAPICallSingle branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DClsInit.name(),
                                                             fixed_node_val=DAPICallSingle.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )
        new_candy.stop_apisingle = True
        return new_candy

    def handle_DVarAssign(self, new_candy):
        # last we will expand the elements after invoke
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             edge_path=[SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )

        # This is for the second DVarAccess branch
        new_candy.control_flow_stack.append(
            CandidateElement(curr_node=new_candy.tree_currNode,
                             edge_path=[CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE, SIBLING_EDGE],
                             next_node_type=VAR_NODE
                             )
        )

        # This is for the DSymtabMod branch
        new_candy.control_flow_stack.append(
            CandidateElement(curr_node=new_candy.tree_currNode,
                             edge_path=[CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE],
                             next_node_type=VAR_NODE,
                             )
        )

        # This is for the first DVarAccess branch
        new_candy.control_flow_stack.append(
            CandidateElement(curr_node=new_candy.tree_currNode,
                             edge_path=[CHILD_EDGE, SIBLING_EDGE],
                             next_node_type=VAR_NODE,
                             )
        )
        # Right now we will evaluate the Dtype branch
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=DVarAssign.name(),
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=TYPE_NODE
                                                             )
                                            )
        return new_candy

    def handle_single_child(self, new_candy):
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             curr_node_val=new_candy.tree_currNode.val,
                                                             edge_path=[CHILD_EDGE],
                                                             next_node_type=CONCEPT_NODE
                                                             )
                                            )
        return new_candy


    def handle_next_node(self, new_candy):
        if len(new_candy.control_flow_stack) == 0:
            new_candy.stop_rolling()
        else:
            element = new_candy.control_flow_stack.pop()
            new_candy.tree_currNode = element.get_current_node()
            for edge in element.get_edge_path()[:-1]:
                new_candy.tree_currNode = new_candy.tree_currNode.progress_node(edge=edge is CHILD_EDGE)

            ## Update the new candy accordingly
            new_candy.curr_edge = element.get_edge_path()[-1]
            '''
                Note that you have to pass the correct curr_node_val to the candidate to deduce the 
                next ast node correctly. Some of the times get_curr_node_val can return None, this is deliberate
                because then curr_node_val will continue from the last concept node. For example, DClsInit has children
                DSingleAPICall followed by a sibling access to DVarAccess. DVarAccess needs to have DSingleAPICall as
                parent, which is what DFS search gives during training.
            '''
            if element.get_curr_node_val() is not None:
                new_candy.curr_node_val = element.get_curr_node_val()
            new_candy.next_node_type = element.get_next_node_type()
            temp = element.get_next_var_decl_id_val()
            if temp is not None:
                new_candy.var_decl_id = temp
            new_candy.fixed_next_node = element.get_fixed_node_val()
        return new_candy

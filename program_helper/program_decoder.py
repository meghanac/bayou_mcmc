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
import tensorflow as tf
from program_helper.ast.tf_model.tree_decoder import TreeDecoder
from program_helper.sequence.sequence_decoder import SequenceDecoder
from program_helper.element.dense_decoder import DenseDecoder


class ProgramDecoder:

    def __init__(self, config,
                 nodes, edges, var_decl_ids,
                 type_helper_val, expr_type_val, ret_type_val,
                 varOrNot, typeOrNot, apiOrNot, symtabmod_or_not, op_or_not,
                 fp_inputs, field_inputs,
                 ret_type, initial_state):
        self.type_emb = tf.get_variable('emb_type', [config.vocab.type_dict_size, config.decoder.units])
        self.concept_emb = tf.get_variable('emb_concept', [config.vocab.concept_dict_size, config.decoder.units])

        with tf.variable_scope("ast_tree"):
            self.ast_tree = TreeDecoder(nodes, edges, var_decl_ids,
                                        type_helper_val, expr_type_val, ret_type_val,
                                        varOrNot, typeOrNot, apiOrNot,
                                        symtabmod_or_not, op_or_not,
                                        ret_type, fp_inputs, field_inputs,
                                        initial_state,
                                        config.decoder.num_layers, config.decoder.units, config.batch_size,
                                        config.vocab.api_dict_size,
                                        config.vocab.type_dict_size,
                                        config.vocab.var_dict_size,
                                        config.vocab.concept_dict_size,
                                        config.vocab.op_dict_size,
                                        self.type_emb,
                                        self.concept_emb,
                                        max_variables=config.max_variables,
                                        )

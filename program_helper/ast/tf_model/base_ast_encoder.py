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

from program_helper.ast.tf_model.top_down_lstm import TopDownLSTM
from program_helper.ast.tf_model.symtab_encoder import SymTabEncoder
from program_helper.sequence.simple_lstm import BaseLSTMClass, SimpleLSTM


class BaseTreeEncoding(BaseLSTMClass):
    def __init__(self, units, num_layers, output_units, batch_size,
                 api_vocab_size, type_vocab_size, var_vocab_size, concept_vocab_size, op_vocab_size,
                 type_emb, concept_emb,
                 max_variables=None,
                 drop_prob=None):
        super().__init__(units, num_layers, output_units, drop_prob)
        self.projection_w, self.projection_b = self.create_projections()

        self.type_emb = type_emb
        self.concept_emb = concept_emb

        self.max_variables = max_variables

        self.batch_size = batch_size
        self.initialize_lstms(units, num_layers, concept_vocab_size, type_vocab_size, api_vocab_size, var_vocab_size, op_vocab_size)

    def initialize_lstms(self, units, num_layers, concept_vocab_size, type_vocab_size, api_vocab_size, var_vocab_size, op_vocab_size):
        with tf.variable_scope('concept_prediction'):
            self.concept_encoder = TopDownLSTM(units, num_layers,
                                               output_units=concept_vocab_size)

        with tf.variable_scope('api_prediction'):
            self.api_encoder = SimpleLSTM(units, num_layers,
                                          output_units=api_vocab_size)

        with tf.variable_scope('type_prediction'):
            self.type_encoder = SimpleLSTM(units, num_layers,
                                           output_units=type_vocab_size)

        with tf.variable_scope('var_access_prediction_type'):
            self.var_encoder1 = SimpleLSTM(units, num_layers,
                                           output_units=var_vocab_size)

        with tf.variable_scope('var_access_prediction_exp'):
            self.var_encoder2 = SimpleLSTM(units, num_layers,
                                           output_units=var_vocab_size)

        with tf.variable_scope('var_access_prediction_ret'):
            self.var_encoder3 = SimpleLSTM(units, num_layers,
                                           output_units=var_vocab_size)

        with tf.variable_scope('op_prediction'):
            self.op_encoder = SimpleLSTM(units, num_layers,
                                           output_units=op_vocab_size)


        with tf.variable_scope("symtab_updater"):
            self.symtab_encoder = SymTabEncoder(units, num_layers,
                                                num_vars=self.max_variables,
                                                batch_size=self.batch_size)

        return

    def get_next_output(self, node, edge,
                        var_decl_id,
                        type_helper_val, expr_type_val, ret_type_val,
                        var_or_not, type_or_not, api_or_not, symtabmod_or_not, op_or_not,
                        symtab_in, method_ret_type_helper, method_fp_type_emb,
                        method_field_type_emb,
                        state_in):

        # Var declaration ID is decremented by 1. This is because when input var decl id is 1
        # or when we see the first variable, we tag it in symtab as 0-th var
        # When there are no variable, var_decl_id becomes -1 and does not update the symtab
        var_decl_id = var_decl_id - 1

        with tf.variable_scope('symtab_in'):
            symtab_all = tf.concat([symtab_in, method_fp_type_emb, method_field_type_emb, method_ret_type_helper], axis=1)
            symtab_all = tf.layers.dense(symtab_all, self.units, activation=tf.nn.tanh)
            symtab_all = tf.layers.dense(symtab_all, self.units)
            flat_symtab = tf.reshape(symtab_all, (self.batch_size, -1))

        with tf.variable_scope('concept_prediction'):
            input = tf.concat([flat_symtab,
                               tf.nn.embedding_lookup(self.concept_emb, node)
                               ], axis=1)
            concept_output, concept_state = self.concept_encoder.get_next_output_with_symtab(input, edge,
                                                                                             state_in)
            concept_logit = self.concept_encoder.get_projection(concept_output)

        with tf.variable_scope('api_prediction'):
            input = tf.concat([
                flat_symtab,
                tf.nn.embedding_lookup(self.concept_emb, node)
            ], axis=1)

            api_output, api_state = self.api_encoder.get_next_output_with_symtab(input, state_in)
            api_logit = self.api_encoder.get_projection(api_output)

        with tf.variable_scope('type_prediction'):
            input = tf.concat([flat_symtab,
                               tf.nn.embedding_lookup(self.concept_emb, node)
                               ], axis=1)
            type_output, type_state = self.type_encoder.get_next_output_with_symtab(input, state_in)
            type_logit = self.type_encoder.get_projection(type_output)

        with tf.variable_scope('op_prediction'):
            input = tf.concat([flat_symtab
                               ], axis=1)
            op_output, op_state = self.op_encoder.get_next_output_with_symtab(input, state_in)
            op_logit = self.op_encoder.get_projection(type_output)

        with tf.variable_scope('var_access_prediction_type'):
            input1 = tf.concat([flat_symtab,
                                tf.nn.embedding_lookup(self.type_emb, type_helper_val),
                                ], axis=1)
            input1 = tf.layers.dense(input1, self.units, activation=tf.nn.tanh)
            input1 = tf.layers.dense(input1, self.units, activation=tf.nn.tanh)
            var_output1, var_state1 = self.var_encoder1.get_next_output_with_symtab(input1, state_in)

        with tf.variable_scope('var_access_prediction_exp'):
            input2 = tf.concat([flat_symtab,
                                tf.nn.embedding_lookup(self.type_emb, expr_type_val),
                                ], axis=1)
            input2 = tf.layers.dense(input2, self.units, activation=tf.nn.tanh)
            input2 = tf.layers.dense(input2, self.units, activation=tf.nn.tanh)
            var_output2, var_state2 = self.var_encoder2.get_next_output_with_symtab(input2, state_in)

        with tf.variable_scope('var_access_prediction_ret'):
            input3 = tf.concat([flat_symtab,
                                tf.nn.embedding_lookup(self.type_emb, ret_type_val),
                                ], axis=1)
            input3 = tf.layers.dense(input3, self.units, activation=tf.nn.tanh)
            input3 = tf.layers.dense(input3, self.units, activation=tf.nn.tanh)
            var_output3, var_state3 = self.var_encoder3.get_next_output_with_symtab(input3, state_in)

        var_output = tf.where(tf.not_equal(type_helper_val, 0),
                              var_output1,
                              tf.where(tf.not_equal(expr_type_val, 0),
                                       var_output2,
                                       var_output3
                                       )
                              )

        var_state = [tf.where(tf.not_equal(type_helper_val, 0),
                              var_state1[j],
                              tf.where(tf.not_equal(expr_type_val, 0),
                                       var_state2[j],
                                       var_state3[j]
                                       )
                              )
                     for j in range(self.num_layers)]

        var_logit = tf.where(tf.not_equal(type_helper_val, 0),
                             self.var_encoder1.get_projection(var_output1),
                             tf.where(tf.not_equal(expr_type_val, 0),
                                      self.var_encoder2.get_projection(var_output2),
                                      self.var_encoder3.get_projection(var_output3)
                                      )
                             )

        with tf.variable_scope("symtab_updater"):
            input = tf.nn.embedding_lookup(self.type_emb, type_helper_val)
            new_symtab = self.symtab_encoder.update_symtab(input, var_decl_id, symtab_in)

        # Update symtab
        new_symtab = tf.where(symtabmod_or_not, new_symtab, symtab_in)
        stripped_symtab = self.symtab_encoder.strip_symtab(var_decl_id, new_symtab)

        # Update output
        output = tf.where(symtabmod_or_not, tf.zeros_like(var_output),
                          tf.where(var_or_not,
                                   var_output,
                                   tf.where(type_or_not,
                                            type_output,
                                            tf.where(api_or_not,
                                                     api_output,
                                                     tf.where(op_or_not,
                                                              op_output,
                                                              concept_output
                                                              )
                                                     )
                                            )
                                   )
                          )

        # Update state
        state = [tf.where(symtabmod_or_not, state_in[j], tf.where(var_or_not,
                                                                  var_state[j],
                                                                  tf.where(type_or_not,
                                                                           type_state[j],
                                                                           tf.where(
                                                                               api_or_not,
                                                                               api_state[j],
                                                                               tf.where(op_or_not,
                                                                                        op_state[j],
                                                                                        concept_state[j]
                                                                                        )
                                                                                )
                                                                           )
                                                                  )
                          )
                 for j in range(self.num_layers)]

        logits = [api_logit, type_logit, var_logit, concept_logit, op_logit]

        return state, output, stripped_symtab, logits


    def get_projection(self, input):
        return tf.nn.xw_plus_b(input, self.projection_w, self.projection_b)


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

from program_helper.ast.tf_model.base_ast_encoder import BaseTreeEncoding


class TreeDecoder(BaseTreeEncoding):
    def __init__(self, nodes, edges, var_decl_ids,
                 type_helper_val, expr_type_val, ret_type_val,
                 var_or_not, type_or_not, api_or_not, symtabmod_or_not, op_or_not,
                 ret_type, fps,  field_inputs,
                 initial_state, num_layers, units, batch_size,
                 api_vocab_size, type_vocab_size,
                 var_vocab_size, concept_vocab_size, op_vocab_size,
                 type_emb, concept_emb,
                 drop_prob=None,
                 max_variables=None,
                 ):

        self.drop_prob = drop_prob

        super().__init__(units, num_layers, units, batch_size,
                         api_vocab_size, type_vocab_size, var_vocab_size, concept_vocab_size, op_vocab_size,
                         type_emb, concept_emb,
                         max_variables=max_variables,
                         drop_prob=drop_prob)

        self.init_symtab = self.symtab_encoder.create_symtab(batch_size, units)
        method_ret_type_emb = tf.expand_dims(tf.nn.embedding_lookup(type_emb, ret_type), axis=1)
        method_fp_type_emb = tf.stack([tf.nn.embedding_lookup(type_emb, fp_type) for fp_type in fps], axis=1)
        method_field_type_emb = tf.stack([tf.nn.embedding_lookup(type_emb, fp_type) for fp_type in field_inputs], axis=1)

        with tf.variable_scope('tree_decoder'):
            self.state = initial_state
            self.symtab = self.init_symtab
            api_output_logits, type_output_logits, \
            var_output_logits, concept_output_logits,\
                op_output_logits = [], [], [], [], []
            for i in range(len(nodes)):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()

                self.state, output, self.symtab, logits = self.get_next_output(nodes[i], edges[i],
                                                                               var_decl_ids[i],
                                                                               type_helper_val[i], expr_type_val[i],
                                                                               ret_type_val[i],
                                                                               var_or_not[i], type_or_not[i],
                                                                               api_or_not[i], symtabmod_or_not[i],
                                                                               op_or_not[i],
                                                                               self.symtab,
                                                                               method_ret_type_emb, method_fp_type_emb,
                                                                               method_field_type_emb,
                                                                               self.state)

                api_logit, type_logit, var_logit, concept_logit, op_logit = logits

                api_output_logits.append(api_logit)
                type_output_logits.append(type_logit)
                var_output_logits.append(var_logit)
                concept_output_logits.append(concept_logit)
                op_output_logits.append(op_logit)

        self.output_logits = [
            tf.stack(concept_output_logits, 1),
            tf.stack(api_output_logits, 1),
            tf.stack(type_output_logits, 1),
            tf.stack(var_output_logits, 1),
            tf.stack(op_output_logits, 1),
            tf.ones((batch_size, len(nodes), batch_size), dtype=tf.float32)
        ]

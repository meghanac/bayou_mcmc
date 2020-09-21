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

import os
import json
import argparse

from data_extraction.data_reader.utils import read_vocab, dump_vocab

from program_helper.ast import AstReader
from program_helper.ast.parser.ast_gen_checker import AstGenChecker
from program_helper.ast.parser.java_compiler import JavaCompiler
from program_helper.sequence.field_reader import FieldReader
from program_helper.sequence.sequence_reader import SequenceReader
from program_helper.set.set_reader import SetReader
from program_helper.element.element_reader import ElementReader
from program_helper.surrounding.surrounding_reader import SurroundingReader
from synthesis.json_synthesis import JSON_Synthesis
from synthesis.ops.candidate_ast import Candidate_AST
from utilities import VocabBuildingDictionary
from utilities.basics import dump_java, dump_json
from trainer_vae.utils import read_config


class ProgramReader:

    def __init__(self, max_ast_depth=32, max_loop_num=3,
                 max_branching_num=3, max_trys=3,
                 max_fp_depth=8,
                 max_fields=8,
                 max_keywords=10,
                 max_camel=3,
                 max_variables=10,
                 data_path=None,
                 infer=False,
                 infer_vocab_path=None,
                 logger=None,
                 repair_mode=True
                 ):

        self.vocab = argparse.Namespace()
        self.data_path = data_path
        self.infer = infer
        self.repair_mode = repair_mode

        self.json_synthesizer = JSON_Synthesis()
        self.java_compiler = JavaCompiler()

        if not self.infer:
            self.concept_vocab = VocabBuildingDictionary()
            self.api_vocab = VocabBuildingDictionary()
            self.apiname_vocab = VocabBuildingDictionary()
            self.type_vocab = VocabBuildingDictionary()
            self.typename_vocab = VocabBuildingDictionary()
            self.var_vocab = VocabBuildingDictionary()
            self.kw_vocab = VocabBuildingDictionary()
            self.op_vocab = VocabBuildingDictionary()
        else:
            self.load_dictionary_from_config(infer_vocab_path)
            compiler_path = infer_vocab_path.replace('/config.json', '')
            self.java_compiler.load(compiler_path)

            self.concept_vocab = VocabBuildingDictionary(self.vocab.concept_dict)
            self.api_vocab = VocabBuildingDictionary(self.vocab.api_dict)
            self.apiname_vocab = VocabBuildingDictionary(self.vocab.apiname_dict)
            self.type_vocab = VocabBuildingDictionary(self.vocab.type_dict)
            self.typename_vocab = VocabBuildingDictionary(self.vocab.typename_dict)
            self.var_vocab = VocabBuildingDictionary(self.vocab.var_dict)
            self.kw_vocab = VocabBuildingDictionary(self.vocab.kw_dict)
            self.op_vocab = VocabBuildingDictionary(self.vocab.op_dict)

        self.ast_reader = AstReader(
            max_depth=max_ast_depth,
            max_loop_num=max_loop_num,
            max_trys=max_trys,
            max_branching_num=max_branching_num,
            max_variables=max_variables,
            concept_vocab=self.concept_vocab,
            api_vocab=self.api_vocab,
            type_vocab=self.type_vocab,
            var_vocab=self.var_vocab,
            op_vocab=self.op_vocab,
            infer=self.infer
        )

        self.ast_checker = AstGenChecker(self.type_vocab, compiler=self.java_compiler, logger=logger)

        self.formal_param_reader = SequenceReader(
            max_depth=max_fp_depth,
            type_vocab=self.type_vocab,
            var_vocab=self.var_vocab,
            concept_vocab=self.concept_vocab,
            infer=self.infer
        )

        self.field_reader = FieldReader(
            max_depth=max_fields,
            type_vocab=self.type_vocab,
            var_vocab=self.var_vocab,
            concept_vocab=self.concept_vocab,
            infer=self.infer
        )

        self.return_type_reader = ElementReader(
            vocab=self.type_vocab,
            infer=self.infer
        )

        self.keyword_reader = SetReader(
            max_elements=max_keywords,
            max_camel=max_camel,
            vocab=[self.apiname_vocab, self.typename_vocab, self.kw_vocab],
            infer=self.infer
        )

        self.surrounding_reader = SurroundingReader(
            max_elements=max_keywords,
            max_camel=max_camel,
            vocab=[self.apiname_vocab, self.type_vocab, self.kw_vocab],
            infer=self.infer
        )

        self.json_programs = None
        self.symtab = dict()

        self.ast_storage_jsons = []
        return

    def get_size(self):
        return len(self.ast_reader.nodes)

    def read_json(self, program_js):

        self.symtab = dict()

        ## Add the return type as part of symbol table, though it shud b added as a dict
        self.symtab['return_type'] = program_js['return_type']
        return_type_id = self.return_type_reader.read_while_vocabing(program_js['return_type'])

        fp_type_head = self.formal_param_reader.form_fp_ast(program_js['formal_params'], self.symtab)
        field_head = self.field_reader.form_ast(program_js['field_ast']['_nodes'], self.symtab)

        parsed_api_array = self.ast_reader.read_while_vocabing(program_js['ast'],
                                                               symtab=self.symtab,
                                                               fp_type_head=fp_type_head,
                                                               field_head=field_head,
                                                               repair_mode=self.repair_mode
                                                               )

        if not self.repair_mode:
            for key in self.symtab.keys():
                if key == "return_type":
                    continue
                node = self.symtab[key]
                if node.type in ["DVarDecl", "DFieldCall"]:
                   node.make_valid()

        parsed_fp_array, valid_fp_vals = self.formal_param_reader.read_while_vocabing(fp_type_head)

        ## Form the AST Node
        ast_node = Candidate_AST(None, None, None, None, None, None)
        ast_node.head = self.ast_reader.ast_node_graph
        ast_node.return_type = program_js['return_type']
        ast_node.formal_param_inputs = valid_fp_vals
        new_program_js = self.json_synthesizer.full_json_extract_no_vocab(ast_node)

        ## Parse the evidences
        parsed_field_array, valid_field_vals = self.field_reader.read_while_vocabing(field_head)

        apicalls, types, keywords = self.keyword_reader.read_while_vocabing(program_js)
        method = self.keyword_reader.read_methodname(program_js['method'])
        classname = self.keyword_reader.read_classname(program_js['className'])
        javadoc_kws = self.keyword_reader.read_natural_language(program_js['javaDoc'])

        surr_ret, surr_fp, surr_method = self.surrounding_reader.read_surrounding(program_js["Surrounding_Methods"])

        ## Run AST checker
        if not self.infer:
            self.ast_checker.check_single_program(ast_node.head, valid_fp_vals, ast_node.return_type,
                                                  field_vals=valid_field_vals,
                                                  update_mode=True)

        checker_outcome_string = self.ast_checker.check_single_program(ast_node.head, valid_fp_vals,
                                                                       ast_node.return_type,
                                                                       field_vals=valid_field_vals)


        self.ast_storage_jsons.append(program_js)

        return parsed_api_array, return_type_id, parsed_fp_array, \
               parsed_field_array, apicalls, types, keywords, \
               method, classname, javadoc_kws, \
               surr_ret, surr_fp, surr_method, \
               new_program_js, checker_outcome_string

    def wrangle(self, ast_programs, return_types, formal_params, field_array,
                apicalls, types, keywords,
                method, classname, javadoc_kws,
                surr_ret, surr_fp, surr_method,
                min_num_data=None):

        self.ast_reader.wrangle(ast_programs, min_num_data=min_num_data)
        self.formal_param_reader.wrangle(formal_params, min_num_data=min_num_data)
        self.field_reader.wrangle(field_array, min_num_data=min_num_data)
        self.return_type_reader.wrangle(return_types, min_num_data=min_num_data)
        self.keyword_reader.wrangle(apicalls, types, keywords, method, classname, javadoc_kws, min_num_data=min_num_data)
        self.surrounding_reader.wrangle(surr_ret, surr_fp, surr_method, min_num_data=min_num_data)

    def save_data(self, effective_javas=None,
                  real_javas=None,
                  program_jsons=None):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.ast_reader.save(self.data_path)
        self.formal_param_reader.save(self.data_path)
        self.field_reader.save(self.data_path)
        self.keyword_reader.save(self.data_path)
        self.return_type_reader.save(self.data_path)
        self.surrounding_reader.save(self.data_path)

        if effective_javas is not None and real_javas is not None:
            output_file = os.path.join(self.data_path, 'passed_programs.java')
            dump_java([[r, e] for e, r in zip(effective_javas, real_javas)], output_file)

        if program_jsons is not None:
            output_file = os.path.join(self.data_path, 'passed_programs.json')
            dump_json({'programs': real_javas}, output_file)

        self.save_dictionary(self.data_path)
        self.java_compiler.save(self.data_path)

        return

    def load_data(self):
        self.ast_reader.load_data(self.data_path)
        self.formal_param_reader.load_data(self.data_path)
        self.field_reader.load_data(self.data_path)
        self.keyword_reader.load_data(self.data_path)
        self.return_type_reader.load_data(self.data_path)
        self.surrounding_reader.load_data(self.data_path)

        self.load_dictionary(os.path.join(self.data_path, 'vocab.json'))
        output_file = os.path.join(self.data_path, 'passed_programs.json')
        with open(output_file, 'r') as f:
            self.json_programs = json.load(f)

        self.java_compiler.load(self.data_path)
        return

    def truncate(self, sz):
        self.ast_reader.truncate(sz)
        self.formal_param_reader.truncate(sz)
        self.field_reader.truncate(sz)
        self.keyword_reader.truncate(sz)
        self.return_type_reader.truncate(sz)
        self.surrounding_reader.truncate(sz)

    def split(self, num_batches):

        self.ast_reader.split(num_batches)
        self.formal_param_reader.split(num_batches)
        self.field_reader.split(num_batches)
        self.keyword_reader.split(num_batches)
        self.return_type_reader.split(num_batches)
        self.surrounding_reader.split(num_batches)

        return

    def load_dictionary(self, vocab_path):
        with open(vocab_path) as f:
            self.vocab = read_vocab(json.load(f))
        return

    def load_dictionary_from_config(self, vocab_path):
        with open(vocab_path) as f:
            config = read_config(json.load(f), infer=True)
            self.vocab = config.vocab

        return

    def save_dictionary(self, data_path):
        self.vocab.concept_dict, self.vocab.concept_dict_size = self.concept_vocab.get_dictionary()
        self.vocab.api_dict, self.vocab.api_dict_size = self.api_vocab.get_dictionary()
        self.vocab.apiname_dict, self.vocab.apiname_dict_size = self.apiname_vocab.get_dictionary()
        self.vocab.var_dict, self.vocab.var_dict_size = self.var_vocab.get_dictionary()
        self.vocab.type_dict, self.vocab.type_dict_size = self.type_vocab.get_dictionary()
        self.vocab.typename_dict, self.vocab.typename_dict_size = self.typename_vocab.get_dictionary()
        self.vocab.op_dict, self.vocab.op_dict_size = self.op_vocab.get_dictionary()
        self.vocab.kw_dict, self.vocab.kw_dict_size = self.kw_vocab.get_dictionary()
        dump_json(dump_vocab(self.vocab), os.path.join(data_path + '/vocab.json'))
        return

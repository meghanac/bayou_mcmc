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
import os

from data_extraction.data_reader.data_reader import Reader
from program_helper.ast.parser.ast_gen_checker import AstGenChecker
from program_helper.ast.parser.ast_similarity_checker import AstSimilarityChecker
from program_helper.ast.parser.ast_traverser import AstTraverser
from program_helper.program_beam_searcher import ProgramBeamSearcher
from synthesis.json_synthesis import JSON_Synthesis
from synthesis.write_java import Write_Java
from trainer_vae.infer import BayesianPredictor
from data_extraction.data_reader.data_loader import Loader
from program_helper.program_reverse_map import ProgramRevMapper
from utilities.basics import dump_json, dump_java
from utilities.logging import create_logger


class InferModelHelper:
    '''
        @filepath controls whether you are reading from a new file or not.
        if unused (i.e. set to None) it reduces to a memory test from the training data
    '''

    def __init__(self, model_path=None,
                 seed=0,
                 beam_width=5,
                 max_num_data=1000,
                 ):

        self.model_path = model_path
        self.infer_model = BayesianPredictor(model_path, batch_size=beam_width, seed=seed, depth='change')
        self.prog_mapper = ProgramRevMapper(self.infer_model.config.vocab)

        self.max_num_data = max_num_data
        self.max_batches = max_num_data // beam_width

        self.reader = None
        self.loader = None

        self.program_beam_searcher = ProgramBeamSearcher(self.infer_model)
        self.logger = create_logger(os.path.join(model_path, 'test_ast_checking.log'))
        self.ast_checker = AstGenChecker(self.program_beam_searcher.infer_model.config.vocab,
                                         compiler=None,
                                         logger=self.logger)

        self.ast_sim_checker = AstSimilarityChecker(logger=self.logger)

        self.json_synthesis = JSON_Synthesis()
        self.java_synthesis = Write_Java()

    def read_and_dump_data(self, filepath=None,
                           data_path=None,
                           min_num_data=0,
                           repair_mode=True):
        '''
        reader will dump the new data into the data path
        '''
        self.reader = Reader(
            dump_data_path=data_path,
            infer=True,
            infer_vocab_path=os.path.join(self.model_path, 'config.json'),
            repair_mode=repair_mode
            )
        self.reader.read_file(filename=filepath,
                           max_num_data=self.max_num_data)
        self.reader.wrangle(min_num_data=min_num_data)
        self.reader.log_info()
        self.reader.dump()

    def synthesize_programs(self, data_path=None,
                            debug_print=False,
                            viability_check=True,
                            dump_result_path=None,
                            dump_jsons=False,
                            dump_psis=False,
                            max_programs=None,
                            real_ast_jsons=None
                            ):

        self.loader = Loader(data_path, self.infer_model.config)
        ## TODO: need to remove
        self.ast_checker.java_compiler = self.loader.program_reader.java_compiler

        psis = self.encode_inputs()

        jsons_synthesized, javas_synthesized = self.decode_programs(
            initial_state=psis,
            debug_print=debug_print,
            viability_check=viability_check,
            max_programs=max_programs,
            real_ast_jsons=real_ast_jsons
        )
        real_codes = self.extract_real_codes(self.loader.program_reader.json_programs)

        if dump_result_path is not None:
            dump_java(javas_synthesized, os.path.join(dump_result_path, 'beam_search_programs.java'),
                      real_codes=real_codes)

            if dump_jsons is True:
                dump_json(jsons_synthesized, os.path.join(dump_result_path, 'beam_search_programs.json'))

            if dump_psis is True:
                dump_json({'embeddings': [psi.tolist() for psi in psis]},
                          os.path.join(dump_result_path + '/EmbeddedProgramList.json'))

        return

    def extract_real_codes(self, json_program):
        # real_codes = []
        # for js in json_program['programs']:
        #     real_codes.append(js['body'])
        return json_program['programs']

    def encode_inputs(self,
                      skip_batches=0
                      ):

        psis = []
        batch_num = 0
        while True:
            try:
                batch = self.loader.next_batch()
                skip_batches -= 1
                if skip_batches >= 0:
                    continue
            except StopIteration:
                break
            psi = self.infer_model.get_initial_state_from_next_batch(batch)
            psi_ = np.transpose(np.array(psi), [1, 0, 2])  # batch_first
            psis.extend(psi_)
            self.prog_mapper.add_batched_data(batch)
            batch_num += 1
            if batch_num >= self.max_batches:
                break

        return psis

    def decode_programs(self, initial_state=None,
                        debug_print=False,
                        viability_check=True,
                        max_programs=None,
                        real_ast_jsons=None,
                        ):
        jsons_synthesized = list()
        javas_synthesized = list()
        outcome_strings = ['' for _ in range(self.infer_model.config.batch_size)]
        sz = min(max_programs, len(initial_state)) if max_programs is not None else len(initial_state)
        for i in range(sz):
            temp = [initial_state[i] for _ in range(self.infer_model.config.batch_size)]
            ast_nodes = self.program_beam_searcher.beam_search_memory(
                initial_state=temp,
                ret_type=self.prog_mapper.get_return_type(i),
                fp_types=self.prog_mapper.get_fp_type_inputs(i),
                field_types=self.prog_mapper.get_field_types(i)
            )
            method_name = self.prog_mapper.get_reconstructed_method_name(i,
                                                                         vocab=self.infer_model.config.vocab.chars_kw)
            beam_js, beam_javas = self.get_json_and_java(
                ast_nodes,
                outcome_strings,
                type_vocab=self.infer_model.config.vocab.chars_type,
                name=method_name
            )

            jsons_synthesized.append({'programs': beam_js})
            javas_synthesized.append(beam_javas)

            if viability_check:
                fp_type_names, ret_typ_name, field_typ_names = self.prog_mapper.get_fp_ret_and_field_names(i,
                                                                                                           vocab=self.infer_model.config.vocab.chars_type)
                outcome_strings = self.ast_checker.run_viability_check(ast_nodes,
                                                                       ret_type=ret_typ_name,
                                                                       fp_types=fp_type_names,
                                                                       field_vals=field_typ_names,
                                                                       debug_print=False
                                                                       )
            if real_ast_jsons is not None:
                real_ast_json = real_ast_jsons[i]
                self.ast_sim_checker.check_similarity_for_all_beams(real_ast_json, beam_js)

            if debug_print:
                self.debug_print_fn(i, ast_nodes, prog_mapper=self.prog_mapper)

        if viability_check:
            self.ast_checker.print_stats()
        if real_ast_jsons is not None:
            self.ast_sim_checker.print_stats()

        return jsons_synthesized, javas_synthesized

    def get_json_and_java(self, ast_nodes, outcome_strings, name='foo', type_vocab=None):

        beam_js, beam_javas = list(), list()
        for j, (outcome, ast_node) in enumerate(zip(outcome_strings, ast_nodes)):
            # path = os.path.join(saver_path, 'program-ast-' + str(i) + 'beam-' + str(j) + '.gv')
            # ast_visualizer.visualize_from_ast_head(ast_node.head, ast_node.log_probability, save_path=path)

            ast_dict = self.json_synthesis.full_json_extract(ast_node, type_vocab, name=name)
            java_program = self.java_synthesis.program_synthesize_from_json(ast_dict,
                                                                            beam_id=j,
                                                                            comment=outcome,
                                                                            prob=ast_node.log_probability)
            beam_js.append(ast_dict)
            beam_javas.append(java_program)
        return beam_js, beam_javas

    def debug_print_fn(self, i, ast_candies, prog_mapper=None):
        ast_traverser = AstTraverser()

        ast_paths = [ast_traverser.depth_first_search(candy.head) for candy in ast_candies]
        print(i)
        if prog_mapper is not None:
            prog_mapper.decode_paths(i)
        print("----------------AST-------------------")
        for ast_path in ast_paths:
            print([item[0] for item in ast_path])

    def close(self):
        self.infer_model.close()

    def reset(self):
        self.prog_mapper.reset()
        self.ast_checker.java_compiler = None
        self.loader = None
        self.reader = None
        self.ast_checker.reset_stats()

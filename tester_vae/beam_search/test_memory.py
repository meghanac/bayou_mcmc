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
import argparse
import sys
import textwrap
import os

from trainer_vae.infer import BayesianPredictor
from ast_helper.beam_searcher.program_beam_searcher import ProgramBeamSearcher
from data_extractor.data_loader import Loader
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

HELP = """"""


def test_memory(_clargs):

    encoder = BayesianPredictor(_clargs.continue_from)
    loader = Loader(clargs, encoder.config)
    psis, apis = [], []
    for i in range(20):
        nodes, edges, targets, \
                ret_type, fp_type, fp_type_targets, _ = loader.next_batch()
        print(ret_type)
        print(fp_type)
        psi = encoder.get_initial_state(nodes, edges, ret_type, fp_type)
        psi_ = np.transpose(np.array(psi), [1, 0, 2])  # batch_first
        psis.extend(psi_)
        apis.extend(nodes)
    encoder.close()

    # print(psis)
    print(apis)

    beam_width = 20
    decoder = BayesianPredictor(_clargs.continue_from, depth='change', batch_size=beam_width)
    program_beam_searcher = ProgramBeamSearcher(decoder)

    for i in range(10):
        for node in apis[i]:
            print(decoder.config.vocab.chars_api[node], end=',')
        print('')
        temp = [psis[i] for _ in range(decoder.config.batch_size)]
        ast_paths, fp_paths, ret_types = program_beam_searcher.beam_search(initial_state=temp)

        print(' ========== AST ==========')
        for ast_path in ast_paths:
            print(ast_path)

        print(' ========== Fp ==========')
        for fp_path in fp_paths:
            print(fp_path)

        print(' ========== Return Type ==========')
        print(ret_types)

        print('\n\n\n\n')
    decoder.close()

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--continue_from', type=str, default='../../trainer_vae/save/',
                        help='ignore config options and continue training model checkpointed here')
    parser.add_argument('--data', default='../../data_extractor/data')
    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    test_memory(clargs)

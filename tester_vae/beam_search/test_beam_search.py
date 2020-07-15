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

from ast_helper.beam_searcher.program_beam_searcher import ProgramBeamSearcher
from ast_helper.ast_visualizor import visualize_from_ast_path
from trainer_vae.infer import BayesianPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

HELP = """"""


def test(clargs):
    beam_width = 20
    predictor = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)
    searcher = ProgramBeamSearcher(predictor)
    ast_paths, fp_paths, ret_types = searcher.beam_search()
    print(' ========== AST ==========')
    for i, ast_path in enumerate(ast_paths):
        print(ast_path)
        path = os.path.join( clargs.saver , 'program-ast-' + str(i) + '.gv')
        visualize_from_ast_path(ast_path, 1.0, save_path=path)

    print(' ========== Fp ==========')
    for i, fp_path in enumerate(fp_paths):
        print(fp_path)
        path = os.path.join(clargs.saver, 'program-fp-' + str(i) + '.gv')
        visualize_from_ast_path(fp_path, 1.0, save_path=path)

    print(' ========== Return Type ==========')
    print(ret_types)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--continue_from', type=str, default='../../trainer_vae/save/1k_vocab_constraint_min_3-600000/',
                        help='ignore config options and continue training model checkpointed here')
    parser.add_argument('--saver', type=str, default='plots/beam_search/')

    clargs = parser.parse_args()
    if not os.path.exists(clargs.saver):
        os.mkdir(clargs.saver)

    sys.setrecursionlimit(clargs.python_recursion_limit)

    test(clargs)

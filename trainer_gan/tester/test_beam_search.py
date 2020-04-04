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

from ast_helper.ast_traverser import AstTraverser
from trainer_gan.infer import BayesianPredictor
from ast_helper.beam_searcher.ast_beam_searcher import TreeBeamSearcher

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

HELP = """"""


def test(_clargs):
    decoder = BayesianPredictor(_clargs.continue_from, beam_width=20, depth_mod=True)
    beam_searcher = TreeBeamSearcher(decoder)
    init_state = decoder.get_random_initial_state()

    ast_traverser = AstTraverser()
    candies = beam_searcher.beam_search(initial_state=init_state)
    for candy in candies:
        path = ast_traverser.depth_first_search(candy.head)
        print(path)
    print('\n\n\n\n')
    decoder.close()


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--continue_from', type=str, default='save',
                        help='ignore config options and continue training model checkpointed here')
    parser.add_argument('--data', default='../data_extractor/data')
    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    test(clargs)

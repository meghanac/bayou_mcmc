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

import os
import argparse
import sys
import textwrap

from data_extractor.data_loader import Loader
from trainer_vae.infer import BayesianPredictor
from tester_vae.tSNE_visualizor.get_labels import get_api
from tester_vae.tSNE_visualizor.tSNE import fitTSNEandplot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


HELP = """{}"""


def plot(clargs):

    predictor = BayesianPredictor(clargs.continue_from, batch_size=5)
    loader = Loader(clargs, predictor.config)

    states, labels = [], []
    for i in range(5):
        nodes, edges, targets, \
            ret_type, fp_type, fp_type_targets, _ = loader.next_batch()
        state = predictor.get_latent_state(nodes, edges, ret_type, fp_type)
        states.extend(state)
        for node in nodes:
            label = get_api(predictor.config, node)
            print(label)
            labels.append(label)
    predictor.close()

    new_states, new_labels = [], []
    for state, label in zip(states, labels):
        if label != 'N/A':
            new_states.append(state)
            new_labels.append(label)
    print('Fitting tSNE')
    fitTSNEandplot(new_states, new_labels, clargs)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--continue_from', type=str, default='save',
                        help='ignore config options and continue training model checkpointed here')
    parser.add_argument('--topK', type=int, default=10,
                        help='plot only the top-k labels')
    parser.add_argument('--data', type=str, default='../data_extraction/data_reader/data',
                        help='load data from here')
    clargs = parser.parse_args()
    clargs.folder = 'plots/test_visualize/'
    clargs.filename = clargs.folder + 'plot_' + clargs.continue_from + '.png'
    if not os.path.exists(clargs.folder):
        os.makedirs(clargs.folder)
    sys.setrecursionlimit(clargs.python_recursion_limit)

    plot(clargs)

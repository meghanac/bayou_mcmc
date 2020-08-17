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

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import numpy as np

from trainer_vae.infer import BayesianPredictor
from data_extractor.data_loader import Loader
from tester_vae.jaccard_metric.get_jaccard_metrics import helper
from tester_vae.jaccard_metric.utils import plotter
from tester_vae.tSNE_visualizor.get_labels import LABELS



def main(clargs):
    num_centroids = 10

    predictor = BayesianPredictor(clargs.continue_from)
    loader = Loader(clargs, predictor.config)
    psis, apis = [], []
    for i in range(1000):
        nodes, edges, targets, \
                ret_type, fp_type, fp_type_targets = loader.next_batch()
        # print("nodes", nodes)
        # print("edges", edges)
        psi = predictor.get_latent_state(nodes, edges, ret_type, fp_type)
        psis.extend(psi)

        labels = get_apis(nodes, predictor.config.vocab.chars_api)
        apis.extend(labels)
    predictor.close()

    print('API Call Jaccard Calculations')
    jac_api_matrix, jac_api_vector = helper(psis, apis, num_centroids=num_centroids)
    plotter(jac_api_matrix, jac_api_vector, name='api_jaccard')

    return


def get_apis(nodes, vocab):
    list_of_labels = []
    for node_arr in nodes:
        labels = []
        for node_val in node_arr:
            label = vocab[node_val]
            if label not in ['DStop', 'DSubTree', 'DBranch', 'DLoop', 'DExcept', '__delim__']:
                # labels.extend(from_call(label))
                label = label.split('.')[1]
                labels.append(label)

        list_of_labels.append(labels)
    return list_of_labels


def from_call(callnode):
    call = re.sub('^\$.*\$', '', callnode)  # get rid of predicates
    name = call.split('(')[0].split('.')[-1]
    name = name.split('<')[0]  # remove generics from call name
    return [name] if name[0].islower() else []  # Java convention




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--continue_from', type=str, default='../../trainer_vae/save/1k_vocab_constraint_min_3-600000',
                        help='directory to load model from')
    parser.add_argument('--top', type=int, default=10,
                        help='plot only the top-k labels')
    parser.add_argument('--data', default='../../data_extractor/data/1k_vocab_constraint_min_3-600000')
    clargs = parser.parse_args()
    main(clargs)

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
import pickle
import json

import numpy as np
from data_extractor.utils import read_vocab


class Loader:
    def __init__(self, clargs, config):
        self.config = config
        print('Loading Data')
        with open(clargs.data + '/ast_apis.pickle', 'rb') as f:
            [self.nodes, self.edges, self.targets] = pickle.load(f)

        with open(clargs.data + '/return_types.pickle', 'rb') as f:
            self.return_types = pickle.load(f)

        with open(clargs.data + '/formal_params.pickle', 'rb') as f:
            [self.fp_types, self.fp_type_targets] = pickle.load(f)

        with open(clargs.data + '/keywords.pickle', 'rb') as f:
            self.keywords = pickle.load(f)

        with open(os.path.join(clargs.data, 'vocab.json')) as f:
            self.config.vocab = read_vocab(json.load(f))

        self.truncate()
        self.split()

        self.reset_batches()
        print('Done')

    def truncate(self):
        config = self.config
        if config.trunct_num_batch is None:
            self.config.num_batches = int(len(self.nodes) / config.batch_size)
        else:
            self.config.num_batches = min(config.trunct_num_batch, int(len(self.nodes) / config.batch_size))

        assert self.config.num_batches > 0, 'Not enough data'
        sz = self.config.num_batches * self.config.batch_size

        self.nodes = self.nodes[:sz, :config.max_ast_depth]
        self.edges = self.edges[:sz, :config.max_ast_depth]
        self.targets = self.targets[:sz, :config.max_ast_depth]

        self.return_types = self.return_types[:sz]

        self.fp_types = self.fp_types[:sz, :config.max_fp_depth]
        self.fp_type_targets = self.fp_type_targets[:sz, :config.max_fp_depth]

        self.keywords = self.keywords[:sz, :config.max_keywords]
        return

    def split(self):
        # split into batches
        self.nodes = np.split(self.nodes, self.config.num_batches, axis=0)
        self.edges = np.split(self.edges, self.config.num_batches, axis=0)
        self.targets = np.split(self.targets, self.config.num_batches, axis=0)

        self.return_types = np.split(self.return_types, self.config.num_batches, axis=0)

        self.fp_types = np.split(self.fp_types, self.config.num_batches, axis=0)
        self.fp_type_targets = np.split(self.fp_type_targets, self.config.num_batches, axis=0)

        self.keywords = np.split(self.keywords, self.config.num_batches, axis=0)
        return

    def reset_batches(self):
        self.batches = iter(
            zip(self.nodes, self.edges, self.targets,
                self.return_types, self.fp_types, self.fp_type_targets,
                self.keywords))
        return

    def next_batch(self):
        n, e, t, r, fp, fpt, kw = next(self.batches)
        return n, e, t, r, fp, fpt, kw

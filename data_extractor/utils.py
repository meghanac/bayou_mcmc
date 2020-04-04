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

CONFIG_VOCAB = ['api_dict', 'api_dict_size', 'ret_dict', 'ret_dict_size',
                'fp_dict', 'fp_dict_size', 'keyword_dict', 'keyword_dict_size']


# convert vocab to JSON
def dump_vocab(vocab):
    js = {}
    for attr in CONFIG_VOCAB:
        js[attr] = vocab.__getattribute__(attr)
    return js


def read_vocab(js):
    vocab = argparse.Namespace()
    for attr in CONFIG_VOCAB:
        vocab.__setattr__(attr, js[attr])

    chars_dict_api = dict()
    for item, value in vocab.api_dict.items():
        chars_dict_api[value] = item
    vocab.__setattr__('chars_api', chars_dict_api)

    chars_dict_fp = dict()
    for item, value in vocab.fp_dict.items():
        chars_dict_fp[value] = item
    vocab.__setattr__('chars_fp', chars_dict_fp)

    chars_dict_ret = dict()
    for item, value in vocab.ret_dict.items():
        chars_dict_ret[value] = item
    vocab.__setattr__('chars_ret', chars_dict_ret)

    chars_dict_kw = dict()
    for item, value in vocab.keyword_dict.items():
        chars_dict_kw[value] = item
    vocab.__setattr__('chars_kw', chars_dict_kw)

    return vocab

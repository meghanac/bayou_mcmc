import pickle

import ijson
import networkx
import json
import os
from networkx.readwrite import json_graph
import argparse
import sys

from data_reader import Reader
from data_extractor.utils import read_vocab
from json_data_extractor import build_graph_from_json_file

STR_BUF = 'java.lang.StringBuffer.StringBuffer()'
STR_APP = 'java.lang.StringBuffer.append(java.lang.String)'
READ_LINE = 'java.io.BufferedReader.readLine()'
CLOSE = 'java.io.InputStream.close()'
STR_LEN = 'java.lang.String.length()'
STR_BUILD = 'java.lang.StringBuilder.StringBuilder(int)'
STR_BUILD_APP = 'java.lang.StringBuilder.append(java.lang.String)'
LOWERCASE_LOCALE = "java.lang.String.toLowerCase(java.util.Locale)"

ALL_DATA_1K_VOCAB = 'all_data_1k_vocab'
TESTING = 'testing-600'
NEW_VOCAB = 'new_1k_vocab_min_3-600000'


class GraphAnalyzer:

    def __init__(self, folder_name):
        self.dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data/" + folder_name + "/"
        self.folder_name = folder_name

        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description="")
        # 'data/' + folder_name + "/"
        parser.add_argument('--data', type=str, default=self.dir_path,
                            help='load data from here')
        # parser.add_argument('--config', type=str, default='config.json',
        #                     help='config file (see description above for help)')
        # parser.add_argument('--continue_from', type=str, default=None,
        #                     help='ignore config options and continue training model checkpointed here')
        # parser.add_argument('--filename', type=str, help='name of data file and dir name')
        self.clargs = parser.parse_args()
        self.clargs.folder_name = folder_name

        # if self.clargs.config and self.clargs.continue_from:
        #     parser.error('Do not provide --config if you are continuing from checkpointed model')
        # if not self.clargs.config and not self.clargs.continue_from:
        #     parser.error('Provide at least one option: --config or --continue_from')
        # if self.clargs.continue_from is None:
        #     self.clargs.continue_from = self.clargs.save

        # Build graph
        if not os.path.exists(os.path.join(self.dir_path, folder_name + "_api_graph.json")):
            vocab_freq_filename = folder_name + "_vocab_freq.json"
            vocab_freq_saved = os.path.exists(os.path.join(self.dir_path, vocab_freq_filename))
            print("vocab_freq_saved:", vocab_freq_saved)
            data_filename = folder_name + ".json"
            self.g = build_graph_from_json_file(self.dir_path, data_filename, vocab_freq_saved=vocab_freq_saved)
        else:
            d = json.load(open(os.path.join(self.dir_path, folder_name + "_api_graph.json")))
            self.g = json_graph.adjacency_graph(d)

        print("Built graph\n")

        # Build database
        if not os.path.exists(self.dir_path + "/vocab.json"):
            reader = Reader(self.clargs, create_database=True)
            reader.save_data(self.clargs.data)
            # Save vocab dictionaries
            with open(os.path.join(self.clargs.data, 'vocab.json')) as f:
                self.vocab = read_vocab(json.load(f))

        elif os.path.exists(self.dir_path + "/vocab.json") and not os.path.exists(self.dir_path + "/program_database.pickle"):
            # Save vocab dictionaries
            with open(os.path.join(self.clargs.data, 'vocab.json')) as f:
                self.vocab = read_vocab(json.load(f))
                print(type(self.vocab))

            reader = Reader(self.clargs, infer=True, create_database=True, vocab=self.vocab)
            reader.save_database(self.clargs.data)
        else:
            # Save vocab dictionaries
            with open(os.path.join(self.clargs.data, 'vocab.json')) as f:
                self.vocab = read_vocab(json.load(f))

        self.vocab2node = self.vocab.api_dict
        self.node2vocab = dict(zip(self.vocab2node.values(), self.vocab2node.keys()))
        self.rettype2num = self.vocab.ret_dict
        self.num2rettype = dict(zip(self.rettype2num.values(), self.rettype2num.keys()))
        self.fp2num = self.vocab.fp_dict
        self.num2fp = dict(zip(self.fp2num.values(), self.fp2num.keys()))

        print("Built database\n")

        # Open database
        with open(self.clargs.data + '/program_database.pickle', 'rb') as f:
            self.database = pickle.load(f)
        self.program_ids = self.database['program_ids']
        self.api_to_prog_ids = self.database['api_to_prog_ids']
        # self.programs_to_ids = self.database['programs_to_ids']



    # def build_prog_database(self):
    #     data_path = os.path.join(self.dir_path, self.folder_name + ".json")
    #     data_f = open(data_path, 'rb')
    #
    #     for program in ijson.items(data_f, 'programs.item'):

    def get_connected_nodes(self, node):
        print("Node:", node)
        print("Number of programs it appears in:", self.g.nodes[node]['frequency'])
        print("Number of APIs it appears in programs with:", len(self.g.edges(node)))
        print("")
        edges = self.g.edges(node, data=True)
        edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
        for i in edges:
            print(i[1], i[2]['weight'])

    def get_program_ids_for_api(self, api):
        return self.api_to_prog_ids[self.vocab2node[api]]

    def get_programs_for_api(self, api):
        prog_ids = self.get_program_ids_for_api(api)
        programs = []

        for id in list(prog_ids):
            programs.append(self.get_formatted_program(id))

        return programs

    def get_formatted_program(self, prog_id):
        i = self.program_ids[prog_id]
        nodes = i[0]
        nodes = [self.node2vocab[n] for n in nodes if n != 0]
        # nodes = [self.node2vocab[n] for n in nodes]
        edges = i[1]
        edges = edges[:len(nodes)]
        return_type = self.num2rettype[i[2]]
        formal_params = i[3]
        formal_params = [self.num2fp[fp] for fp in formal_params if fp != 0]
        return nodes, edges, return_type, formal_params

    def get_program_ids_with_multiple_apis(self, apis):
        common_programs = self.get_program_ids_for_api(apis[0])
        for api in apis:
            progs = self.get_program_ids_for_api(api)
            common_programs.intersection_update(progs)

        return common_programs

    def get_programs_with_multiple_apis(self, apis):
        prog_ids = self.get_program_ids_with_multiple_apis(apis)
        programs = []
        for id in prog_ids:
            programs.append(self.get_formatted_program(id))
        return programs

    def print_lists(self, given_list):
        for i in given_list:
            print(i)


    def testing(self):
        self.get_connected_nodes(LOWERCASE_LOCALE)
        # prog_id =
        print("\n\n")
        self.print_lists(self.get_programs_for_api('java.lang.String.length()'))
        print("\n\n")
        self.print_lists(self.get_programs_with_multiple_apis(['java.lang.String.length()', 'DBranch']))

graph_analyzer = GraphAnalyzer(NEW_VOCAB)
graph_analyzer.testing()



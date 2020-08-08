import itertools
import pickle
import math
import ijson
import networkx as nx
import json
import os
from networkx.readwrite import json_graph
import argparse
import sys
import numpy as np
from graphviz import Digraph

from data_extractor.data_reader import Reader, MAX_AST_DEPTH

from ast_helper.ast_traverser import AstTraverser
from data_extractor.utils import read_vocab
from data_extractor.json_data_extractor import build_graph_from_json_file
from ast_helper.ast_reader import AstReader

from test_suite import MOST_COMMON_APIS, MID_COMMON_APIS, UNCOMMON_APIS, MID_COMMON_DISJOINT_PAIRS, \
    MOST_COMMON_DISJOINT_PAIRS, UNCOMMON_DISJOINT_PAIRS

STR_BUF = 'java.lang.StringBuffer.StringBuffer()'
STR_APP = 'java.lang.StringBuffer.append(java.lang.String)'
READ_LINE = 'java.io.BufferedReader.readLine()'
CLOSE = 'java.io.InputStream.close()'
STR_LEN = 'java.lang.String.length()'
STR_BUILD = 'java.lang.StringBuilder.StringBuilder(int)'
STR_BUILD_APP = 'java.lang.StringBuilder.append(java.lang.String)'
LOWERCASE_LOCALE = "java.lang.String.toLowerCase(java.util.Locale)"

DATA_DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/data/"
ALL_DATA_1K_VOCAB = 'all_data_1k_vocab'
ALL_DATA_1K_VOCAB_NO_DUP = 'all_data_1k_vocab_no_duplicates'
TESTING = 'testing-600'
NEW_VOCAB = 'new_1k_vocab_min_3-600000'
ALL_DATA = 'all_data'
ALL_DATA_NO_DUP = 'all_data_no_duplicates'

APIS = 'apis'
RT = 'return_types'
FP = 'fp'

TOP = 'top'
MID = 'mid'
LOW = 'low'

MIN = 'min'
MIN_EQ = 'min_eq'
MAX = 'max'
MAX_EQ = 'max_eq'
EQ = 'eq'

class GraphAnalyzer:

    def __init__(self, folder_name, test=False, save_reader=False, load_reader=False, shuffle_data=True,
                 remove_duplicates=False, load_g_without_control_structs=True):
        if test:
            self.dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data/" + folder_name + "/test_set/"
        else:
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

        if test:
            self.clargs.data_filename = folder_name + "_test"
        else:
            self.clargs.data_filename = folder_name

        # if self.clargs.config and self.clargs.continue_from:
        #     parser.error('Do not provide --config if you are continuing from checkpointed model')
        # if not self.clargs.config and not self.clargs.continue_from:
        #     parser.error('Provide at least one option: --config or --continue_from')
        # if self.clargs.continue_from is None:
        #     self.clargs.continue_from = self.clargs.save

        data_filename = self.clargs.data_filename + ".json"

        # Build graph
        if not os.path.exists(os.path.join(self.dir_path, folder_name + "_api_graph.json")):
            vocab_freq_filename = folder_name + "_vocab_freq.json"
            vocab_freq_saved = os.path.exists(os.path.join(self.dir_path, vocab_freq_filename))
            print("vocab_freq_saved:", vocab_freq_saved)
            self.g = build_graph_from_json_file(self.dir_path, data_filename, vocab_freq_saved=vocab_freq_saved,
                                                return_g_without_control_structs=load_g_without_control_structs)
        else:
            if load_g_without_control_structs:
                self.g = json_graph.adjacency_graph(
                    json.load(open(os.path.join(self.dir_path, folder_name + "_api_graph.json"))))
            else:
                self.g = json_graph.adjacency_graph(
                    json.load(open(os.path.join(self.dir_path, folder_name + "_graph.json"))))

        print("Built graph\n")

        # Build database
        if not os.path.exists(self.dir_path + "/vocab.json"):
            self.reader = Reader(self.clargs, create_database=True, shuffle=shuffle_data,
                                 remove_duplicates=remove_duplicates)
            self.reader.save_data(self.clargs.data)
            # Save vocab dictionaries
            with open(os.path.join(self.clargs.data, 'vocab.json')) as f:
                self.vocab = read_vocab(json.load(f))

        elif os.path.exists(self.dir_path + "/vocab.json") and not os.path.exists(
                self.dir_path + "/program_database.pickle"):
            # Save vocab dictionaries
            with open(os.path.join(self.clargs.data, 'vocab.json')) as f:
                self.vocab = read_vocab(json.load(f))

            self.reader = Reader(self.clargs, infer=True, create_database=True, vocab=self.vocab)
            self.reader.save_database(self.clargs.data)
        else:
            with open(os.path.join(self.clargs.data, 'vocab.json')) as f:
                self.vocab = read_vocab(json.load(f))

            # Save vocab dictionaries
            with open(os.path.join(self.clargs.data, 'vocab.json')) as f:
                self.vocab = read_vocab(json.load(f))

            if load_reader:
                with open(self.clargs.data + '/reader.pickle', 'rb') as f:
                    self.reader = pickle.load(f)
            else:
                self.reader = Reader(self.clargs, infer=True, vocab=self.vocab)

        self.vocab2node = self.vocab.api_dict
        self.node2vocab = dict(zip(self.vocab2node.values(), self.vocab2node.keys()))
        self.rettype2num = self.vocab.ret_dict
        self.num2rettype = dict(zip(self.rettype2num.values(), self.rettype2num.keys()))
        self.fp2num = self.vocab.fp_dict
        self.num2fp = dict(zip(self.fp2num.values(), self.fp2num.keys()))

        print("Loaded Reader\n")

        # Open database
        with open(self.clargs.data + '/program_database.pickle', 'rb') as f:
            self.database = pickle.load(f)
        with open(self.clargs.data + '/ast_apis.pickle', 'rb') as f:
            [self.nodes, self.edges, self.targets] = pickle.load(f)
        with open(self.clargs.data + '/return_types.pickle', 'rb') as f:
            self.return_types = pickle.load(f)
        with open(self.clargs.data + '/formal_params.pickle', 'rb') as f:
            [self.fp_types, self.fp_type_targets] = pickle.load(f)

        # self.program_ids = self.database['program_ids']
        self.api_to_prog_ids = self.database['api_to_prog_ids']
        self.rt_to_prog_ids = self.database['rt_to_prog_ids']
        self.fp_to_prog_ids = self.database['fp_to_prog_ids']
        self.num_programs = len(self.nodes)

        print("Built Database\n")

        data_f = open(os.path.join(self.dir_path, data_filename))
        self.json_asts = ijson.items(data_f, 'programs.item')
        # data_f.close()
        # self.json_asts = json.load(data_f)

        if save_reader:
            with open(self.dir_path + '/reader.pickle', 'wb') as f:
                pickle.dump(self.reader, f)
                f.close()
            print("Saved Reader\n")

        self.ast_reader = AstReader()
        self.ast_traverser = AstTraverser()

    # def build_prog_database(self):
    #     data_path = os.path.join(self.dir_path, self.folder_name + ".json")
    #     data_f = open(data_path, 'rb')
    #
    #     for program in ijson.items(data_f, 'programs.item'):

    def fetch_data(self, prog_id):
        return self.nodes[prog_id], self.edges[prog_id], self.return_types[prog_id], self.fp_types[prog_id]

    def fetch_data_with_targets(self, prog_id):
        return self.nodes[prog_id], self.edges[prog_id], self.return_types[prog_id], \
               self.fp_types[prog_id], self.targets[prog_id], self.fp_type_targets[prog_id]

    def fetch_hashable_data_with_targets(self, prog_id):
        return tuple(self.nodes[prog_id].tolist()), tuple(self.edges[prog_id].tolist()), self.return_types[prog_id], \
               tuple(self.fp_types[prog_id].tolist()), tuple(self.targets[prog_id].tolist()), tuple(
            self.fp_type_targets[prog_id].tolist())

    def fetch_nodes_as_list(self, prog_id):
        return self.nodes[prog_id].tolist()

    def get_apis_in_prog_set(self, prog_id):
        nodes, _, _, _, targets, _ = self.fetch_data_with_targets(prog_id)
        nodes = set(nodes)
        nodes.update(set(targets))
        nodes.discard(0)
        return list(nodes)

    def get_connected_nodes(self, node):
        print("Node:", node)
        print("Number of programs it appears in:", self.g.nodes[node]['frequency'])
        print("Number of APIs it appears in programs with:", len(self.g.edges(node)))
        print("")
        edges = self.g.edges(node, data=True)
        edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
        for i in edges:
            print(i[1], i[2]['weight'])

    def get_program_ids_for_api(self, api, limit=None):
        try:
            prog_ids = self.api_to_prog_ids[self.vocab2node[api]].copy()
        except KeyError:
            # print("api key error:", api)
            return set([])
        if limit is not None and len(prog_ids) > limit:
            return set(itertools.islice(prog_ids, limit))
        # print(prog_ids)
        # print(type(prog_ids))
        return set(prog_ids)

    def get_program_ids_for_api_length_k(self, api, min_max_eq, k, limit=None):
        prog_ids = self.get_program_ids_for_api(api)
        prog_ids = list(prog_ids)

        if min_max_eq == MIN:
            prog_ids = filter(lambda x: self.fetch_nodes_as_list(x).index(0) > k, prog_ids)
        elif min_max_eq == MIN_EQ:
            prog_ids = filter(lambda x: self.fetch_nodes_as_list(x).index(0) >= k, prog_ids)
        elif min_max_eq == MAX:
            prog_ids = filter(lambda x: self.fetch_nodes_as_list(x).index(0) < k, prog_ids)
        elif min_max_eq == MAX_EQ:
            prog_ids = filter(lambda x: self.fetch_nodes_as_list(x).index(0) <= k, prog_ids)
        elif min_max_eq == EQ:
            prog_ids = filter(lambda x: self.fetch_nodes_as_list(x).index(0) == k, prog_ids)
        else:
            raise ValueError("min_max_eq must be min, min_eq, max, max_eq, or eq")

        if limit is not None and len(list(prog_ids)) > limit:
            return set(itertools.islice(prog_ids, limit))

        return set(prog_ids)

    def get_programs_for_api(self, api, input_prog_ids=None, limit=None, get_targets=True, get_jsons=False):
        if input_prog_ids is None:
            prog_ids = self.get_program_ids_for_api(api, limit=limit)
        else:
            prog_ids = input_prog_ids
        programs = []

        for id in list(prog_ids):
            programs.append(self.get_formatted_program(id, get_targets=get_targets, get_jsons=get_jsons))

        return programs

    def get_json_ast(self, prog_id):
        return None

    def get_formatted_program(self, prog_id, get_targets=True, get_jsons=False):
        i = self.fetch_data_with_targets(prog_id)
        nodes = i[0]
        nodes = [self.node2vocab[n] for n in nodes if n != 0]
        # nodes = [self.node2vocab[n] for n in nodes]
        edges = i[1]
        edges = edges[:len(nodes)]
        return_type = self.num2rettype[i[2]]
        formal_params = i[3]
        formal_params = [self.num2fp[fp] for fp in formal_params if fp != 0]
        if get_targets:
            targets = [self.node2vocab[t] for t in i[4] if t != 0]
            fp_targets = [self.num2fp[fp] for fp in i[5] if fp != 0]
            if get_jsons:
                return ("nodes:", nodes), ("edges:", edges), ("targets:", targets), ("return type:", return_type), (
                    "formal params:", formal_params), ("fp targets:", fp_targets), ("json:", self.get_json_ast(prog_id))
            else:
                return ("nodes:", nodes), ("edges:", edges), ("targets:", targets), ("return type:", return_type), (
                    "formal params:", formal_params), ("fp targets:", fp_targets)

        if get_jsons:
            return ("nodes:", nodes), ("edges:", edges), ("return type:", return_type), (
                "formal params:", formal_params), ("json:", self.get_json_ast(prog_id))
        else:
            return ("nodes:", nodes), ("edges:", edges), ("return type:", return_type), (
                "formal params:", formal_params)

    def get_program_ids_with_multiple_apis(self, apis, limit=None, exclude=None):
        common_programs = self.get_program_ids_for_api(apis[0])
        if type(common_programs) == dict:
            print(apis[0], "\n\n\n")

        for api in apis:
            progs = self.get_program_ids_for_api(api)
            if type(progs) == dict:
                print(api, "\n\n\n")
            if type(common_programs) == dict:
                print(apis[0], "\n\n\n")
            common_programs.intersection_update(progs)

        if exclude is not None:
            for e in exclude:
                apis_copy = apis.copy()
                apis_copy.append(e)
                e_progs = self.get_program_ids_with_multiple_apis(apis_copy)
                common_programs -= e_progs

        if limit is not None and len(common_programs) > limit:
            print("Total number of results:", len(common_programs))
            return set(itertools.islice(common_programs, limit))

        return common_programs

    def get_programs_with_multiple_apis(self, apis, limit=None, get_targets=True, get_jsons=False, exclude=None):
        prog_ids = self.get_program_ids_with_multiple_apis(apis, limit=limit, exclude=exclude)
        programs = []
        for id in prog_ids:
            programs.append(self.get_formatted_program(id, get_targets=get_targets, get_jsons=get_jsons))
        print("Total number of outputted results:", len(programs), "\n")
        return programs

    def print_programs_from_ids(self, prog_ids, limit=None):
        prog_ids = list(prog_ids)
        prog_ids = [self.get_formatted_program(i) for i in prog_ids]
        if limit is not None:
            prog_ids = prog_ids[:limit]
        print("\n")
        self.print_lists(prog_ids)

    def print_lists(self, given_list):
        for i in given_list:
            for i1 in i:
                print(i1)
            print("")

    # TODO: this doesn't work because the nodes/edges/etc. are shuffled at the end of Reader
    # def get_json_ast(self, prog_id):
    #     json_ast = next(itertools.islice(self.json_asts, prog_id, None))
    #     # print([i for i in json_ast])
    #     return json_ast

    def plot_ast(self, nodes, edges, targets, filename='temporary'):
        dot = Digraph(comment='Program AST', format='eps')
        # dot.node(nodes[0], )
        for i in range(len(nodes)):
            dot.node(str(i), label=nodes[i])
            dot.node(str(i+1), label=targets[i])
            label = 'child' if edges[i] else 'sibling'
            dot.edge(str(i), str(i+1), label=label, constraint='true', direction='LR')
            # dfs_id += 1

        dot.render("graph_analysis_outputs/" + filename)
        return dot

    def get_vectors_and_plot(self, js, filename='temporary'):
        # print(self.get_json_ast(600000))
        print(js)
        ast_node_graph = self.ast_reader.get_ast_from_json(js['ast']['_nodes'])
        path = self.ast_traverser.depth_first_search(ast_node_graph)
        print(path)
        print(self.reader.read_ast(js['ast']))
        output = self.reader.read_ast(js['ast'])
        self.plot_path(output, filename=filename)
        print([i[0] for i in output])
        print([i[1] for i in output])
        print([i[2] for i in output])
        print("")

    def plot_path(self, path, filename='temporary'):
        dot = Digraph(comment='Program AST', format='eps')

        for i in path:
            label = 'child' if i[1] else 'sibling'
            dot.edge(str(i[0]), str(i[2]), label=label, constraint='true', direction='LR')
            # dfs_id += 1

        dot.render("graph_analysis_outputs/" + filename)
        return dot

    def get_cooccurrence_stats(self, prog_ids):
        APIS = 'apis'
        RT = 'return_types'
        FP = 'fp'
        prog_ids = list(prog_ids)
        stats = {APIS: {}, RT: {}, FP: {}}

        # compile data
        for i in prog_ids:
            apis = list(set(self.nodes[i]).union(set(self.targets[i])))
            for api in apis:
                if api in stats[APIS]:
                    stats[APIS][api] += 1
                else:
                    stats[APIS][api] = 1
            if self.return_types[i] in stats[RT]:
                stats[RT][self.return_types[i]] += 1
            else:
                stats[RT][self.return_types[i]] = 1
            for fp in self.fp_types[i]:
                if fp in stats[FP]:
                    stats[FP][fp] += 1
                else:
                    stats[FP][fp] = 1

        # remove __delim__
        try:
            stats[APIS].pop(0)
            stats[FP].pop(0)
            stats[RT].pop(0)
        except KeyError:
            pass

        return stats

    def get_sorted_stats(self, stats):
        def take_count(e):
            return e[1]
        sorted_apis = sorted(stats[APIS].items(), key=take_count, reverse=True)
        sorted_rt = sorted(stats[RT].items(), key=take_count, reverse=True)
        sorted_fp = sorted(stats[FP].items(), key=take_count, reverse=True)

        return sorted_apis, sorted_rt, sorted_fp

    def print_summary_stats(self, prog_ids):

        sorted_apis, sorted_rt, sorted_fp = self.get_sorted_stats(self.get_cooccurrence_stats(prog_ids))

        # print stats
        print("\n-----------------\n", APIS, ":")
        for i in sorted_apis:
            print(self.node2vocab[i[0]], "\t", str(i[1]))
        print("\n-----------------\n", RT, ":")
        for i in sorted_rt:
            print(self.num2rettype[i[0]], "\t", str(i[1]))
        print("\n-----------------\n", FP, ":")
        for i in sorted_fp:
            print(self.num2fp[i[0]], "\t", str(i[1]))

    def get_k_cooccurring_apis_rt_fp(self, api, level, k=1):
        if level not in {'top', 'mid', 'low'}:
            raise ValueError("level must be 'top' 'mid' or 'low'")

        if type(api) == list:
            stats = self.get_cooccurrence_stats(self.get_program_ids_with_multiple_apis(api))
        elif type(api) == str:
            stats = self.get_cooccurrence_stats(self.get_program_ids_for_api(api))
        else:
            raise ValueError("api type must be list or string")

        sorted_apis, sorted_rt, sorted_fp = self.get_sorted_stats(stats)

        api_k = min(len(sorted_apis), k)
        rt_k = min(len(sorted_rt), k)
        fp_k = min(len(sorted_fp), k)

        if level == 'top':
            apis = [i[0] for i in sorted_apis[:api_k]]
            rt = [i[0] for i in sorted_rt[:rt_k]]
            fp = [i[0] for i in sorted_fp[:fp_k]]

        elif level == 'mid':
            api_diff = math.floor((len(sorted_apis) - api_k)/2)
            apis = [i[0] for i in sorted_apis[api_diff:api_diff+k]]
            rt_diff = math.floor((len(sorted_rt) - rt_k)/2)
            rt = [i[0] for i in sorted_rt[rt_diff:rt_diff+k]]
            fp_diff = math.floor((len(sorted_fp) - fp_k)/2)
            fp = [i[0] for i in sorted_fp[fp_diff:fp_diff+k]]

        else:
            apis = [i[0] for i in sorted_apis[-api_k:]]
            rt = [i[0] for i in sorted_rt[-rt_k:]]
            fp = [i[0] for i in sorted_fp[-fp_k:]]

        assert len(apis) == api_k, "apis: " + str(apis) + " api_k: " + str(api_k)
        assert len(rt) == rt_k, "rt: " + str(rt) + " rt_k: " + str(rt_k)
        assert len(fp) == fp_k, "fp: " + str(fp) + " fp_k: " + str(fp_k)

        return apis, rt, fp

    def get_top_k_rt_fp(self, apis, k=np.inf):
        rt = []
        fp = []
        for api in apis:
            prog_ids = self.get_program_ids_for_api(api)
            stats = self.get_cooccurrence_stats(prog_ids)
            # self.print_summary_stats(prog_ids)
            _, sorted_rt, sorted_fp = self.get_sorted_stats(stats)
            total_rt = sum([i[1] for i in sorted_rt])
            total_fp = sum([i[1] for i in sorted_fp])
            sorted_rt = [(i[0], 1.0 * i[1] / total_rt) for i in sorted_rt]
            sorted_fp = [(i[0], 1.0 * i[1] / total_fp) for i in sorted_fp]
            rt.append(sorted_rt)
            fp.append(sorted_fp)

        # flatten and reduce lists
        rt = [item for elem in rt for item in elem]
        fp = [item for elem in fp for item in elem]

        def reduce(items):
            reduced = []
            keys = list(set([i[0] for i in items]))
            for key in keys:
                item_value = 0.0
                for item in items:
                    if key == item[0]:
                        item_value += item[1]
                reduced.append((key, item_value))
            return reduced

        rt = reduce(rt)
        fp = reduce(fp)

        # sort lists
        rt = sorted(rt, key=lambda x: x[1], reverse=True)
        fp = sorted(fp, key=lambda x: x[1], reverse=True)

        rt_k = min(k, len(rt))
        fp_k = min(k, len(fp))

        return rt[:rt_k], fp[:fp_k]

    def get_disjoint_api(self, api, level, k=1):
        if level not in {'top', 'mid', 'low'}:
            raise ValueError("level must be 'top' 'mid' or 'low'")
        orig_k = k
        disjoint_nodes = self.g.nodes() - self.g.neighbors(api) - {api}
        node_attr = nx.get_node_attributes(self.g, 'frequency')

        def sort_by_freq(node):
            return node_attr[node]

        sorted_dj_nodes = sorted(disjoint_nodes, key=sort_by_freq, reverse=True)

        k = min(k, len(sorted_dj_nodes))
        if level == TOP:
            selected = sorted_dj_nodes[:k]
        elif level == MID:
            idx = max(math.floor(len(sorted_dj_nodes)/2 - k/2), 0)
            selected = sorted_dj_nodes[idx:idx+k]
        elif level == LOW:
            selected = sorted_dj_nodes[-k:]
        else:
            raise ValueError("level must be 'top', 'mid', or 'low'")

        assert len(selected) == orig_k

        return [(api, node_attr[api]) for api in selected]
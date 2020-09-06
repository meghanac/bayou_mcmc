import json
import math
import os
import pickle
import random

import numpy as np

from data_extractor.graph_analyzer import GraphAnalyzer, MIN_EQ, MAX_EQ
from test_utils import get_str_posterior_distribution
from data_extractor.dataset_creator import IN_CS, IN_API, EX_API, EX_CS, SEEN, NEW

from mcmc import MCMCProgram

DBRANCH = 'DBranch'
DLOOP = 'DLoop'
DEXCEPT = 'DExcept'
DSUBTREE = 'DSubTree'
DSTOP = 'DStop'

NODES_IDX = 0
EDGES_IDX = 1
TARGETS_IDX = 2
RET_TYPE_IDX = 3
FP_IDX = 4
FP_TARGETS_IDX = 5

JACC_API = "jaccard_api"
JACC_SEQ = "jaccard_seq"
AST_EQ = "ast_equivalence"
IN_SET = "in_set"
MIN_DIFF_ST = 'min_diff_statements'
MIN_DIFF_CS = "min_diff_control_structs"

RAND = 'random'
ASC = 'ascending'
PLUS_0 = 0
PLUS_1 = 1
PLUS_2 = 2
PLUS_3 = 3


class Metrics:
    def __init__(self, num_iterations, top_k=10):
        self.num_iter = num_iterations
        self.top_k = top_k

    def get_top_k_gen_progs(self, posterior_dist):
        posterior = posterior_dist.items
        posterior = sorted([(i[0], i[1][0] / self.num_iter * i[1][1]) for i in posterior], reverse=True,
                           key=lambda x: x[1])  # TODO: CHECK IF THIS METRIC MAKES SENSE
        top_k = min(self.top_k, len(posterior))
        posterior = posterior[:top_k]
        posterior = [i[0] for i in posterior]
        return posterior

    def jaccard_api(self, test_data_point, posterior_dist):
        test_apis_set = self.get_apis_set(test_data_point)
        posterior = self.get_top_k_gen_progs(posterior_dist)
        posterior = set([self.get_apis_set(i) for i in posterior])
        return self.get_jaccard_distance(test_apis_set, posterior)

    def jaccard_sequence(self, test_data_point, posterior_dist):
        def get_api_seq_from_data_point(data_point):
            sequence = []
            for i in range(len(data_point[NODES_IDX])):
                for idx in [NODES_IDX, TARGETS_IDX]:
                    api = data_point[idx][i]
                    if api not in {DLOOP, DEXCEPT, DBRANCH, DSUBTREE, DSTOP} and api not in sequence:
                        sequence.append(api)
            return tuple(sequence)

        test_sequence_set = {get_api_seq_from_data_point(test_data_point)}
        posterior = self.get_top_k_gen_progs(posterior_dist)
        posterior = set([get_api_seq_from_data_point(i) for i in posterior])
        return self.get_jaccard_distance(test_sequence_set, posterior)

    def ast_equivalence_top_k(self, test_data_point, posterior_dist):
        return test_data_point in self.get_top_k_gen_progs(posterior_dist)

    def abs_min_diff_statements_top_k(self, test_data_point, posterior_dist):
        top_k_diff = self.get_top_k_gen_progs(posterior_dist)
        top_k_diff = [abs(len(gen_data_point[NODES_IDX]) - len(test_data_point[NODES_IDX])) for gen_data_point in
                      top_k_diff]
        return 1.0 * min(top_k_diff)

    def abs_min_diff_control_structs_top_k(self, test_data_point, posterior_dist):
        def get_num_control_structures(data_point):
            return sum([int(api in {DBRANCH, DLOOP, DEXCEPT}) for api in data_point[NODES_IDX]])

        top_k_diff = self.get_top_k_gen_progs(posterior_dist)
        top_k_diff = [abs(get_num_control_structures(gen_data_point) - get_num_control_structures(test_data_point)) for
                      gen_data_point in top_k_diff]
        return 1.0 * min(top_k_diff)

    def get_apis_set(self, data_point):
        apis = set(data_point[NODES_IDX])
        apis.update(set(data_point[TARGETS_IDX]))
        return apis

    def appears_in_set(self, posterior_dist, ret_type, fp, ga):
        posterior_progs = self.get_top_k_gen_progs(posterior_dist)

        in_set = False
        for gen_data_point in posterior_progs:
            apis = self.get_apis_set(gen_data_point)

            prog_ids = ga.get_program_ids_with_multiple_apis(list(apis))
            for prog_id in prog_ids:
                prog = ga.fetch_all_list_data_without_delim(prog_id)
                in_set = in_set or (prog[:5] == (
                    gen_data_point[NODES_IDX], gen_data_point[EDGES_IDX], gen_data_point[TARGETS_IDX], ret_type, fp))

        return 1.0 * int(in_set)

    def get_jaccard_distance(self, set_a, set_b):
        try:
            1.0 * len(set_a & set_b) / len(set_a | set_b)
        except ZeroDivisionError:
            print("get_jaccard_distance: ZeroDivisionError but shouldn't be here!")
            print("len set_a:", len(set_a), "len set_b:", len(set_b), "len(set_a & set_b):", len(set_a & set_b),
                  "len(set_a | set_b):", len(set_a | set_b))
            return 0.0

    def get_all_metrics(self, test_data_point, posterior_dist, ret_type, fp, ga):
        jaccard_api = self.jaccard_api(test_data_point, posterior_dist)
        jaccard_seq = self.jaccard_sequence(test_data_point, posterior_dist)
        ast_eq = self.ast_equivalence_top_k(test_data_point, posterior_dist)
        min_diff_cs = self.abs_min_diff_control_structs_top_k(test_data_point, posterior_dist)
        min_diff_statements = self.abs_min_diff_statements_top_k(test_data_point, posterior_dist)
        in_dataset = self.appears_in_set(posterior_dist, ret_type, fp, ga)

        return jaccard_api, jaccard_seq, ast_eq, min_diff_cs, min_diff_statements, in_dataset

    def get_all_averaged_metrics(self, test_data_point, posterior_dist, ret_type, fp, ga):
        jaccard_api, jaccard_seq, ast_eq, min_diff_cs, min_diff_statements, in_dataset = self.get_all_metrics(
            test_data_point, posterior_dist, ret_type, fp, ga)

        return jaccard_api / self.num_iter, jaccard_seq / self.num_iter, ast_eq / self.num_iter, \
            min_diff_cs / self.num_iter, min_diff_statements / self.num_iter, in_dataset / self.num_iter


class Experiments:
    def __init__(self, data_dir_name, model_dir_path, experiment_dir_name, num_iterations, save_mcmc_progs=True):
        self.metrics = Metrics(num_iterations * 1.0)
        self.num_iter = num_iterations * 1.0
        self.model_dir_path = model_dir_path
        self.data_dir_name = data_dir_name

        experiments_path = os.path.dirname(os.path.realpath(__file__)) + "/experiments"

        if not os.path.exists(experiments_path):
            os.mkdir(experiments_path)

        if not os.path.exists(experiments_path + "/" + experiment_dir_name):
            os.mkdir(experiments_path + "/" + experiment_dir_name)

        self.exp_dir_path = experiments_path + "/" + experiment_dir_name
        self.analysis_f = open(self.exp_dir_path + "/analysis.txt", "w+")

        self.analysis_f.write("data dir name: " + data_dir_name + "\n")
        self.analysis_f.write("model dir path: " + model_dir_path + "\n")
        self.analysis_f.write("num iterations: " + str(num_iterations) + "\n")

        self.train_ga = GraphAnalyzer(data_dir_name, train_test_split='train', filename='all_training_data', load_reader=True)
        self.all_test_ga = GraphAnalyzer(data_dir_name, train_test_split='test', filename='test_set', load_reader=True)  # TODO: fix
        self.test_ga = GraphAnalyzer(data_dir_name, train_test_split='small_test', filename='small_test_set', load_reader=True)

        print(self.train_ga.g.nodes['java.lang.StringBuffer.deleteCharAt(int)'])

        data_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data_extractor/data/" + data_dir_name)
        testing_path = os.path.join(data_dir_path, "train_test_sets/test")

        with open(testing_path + "/test_set_new_prog_ids.pickle", "rb") as f:
            self.all_test_prog_ids_to_idx = pickle.load(f)
            self.all_test_idx_to_prog_ids = dict(
                zip(self.all_test_prog_ids_to_idx.values(), self.all_test_prog_ids_to_idx.keys()))

        with open(testing_path + "/small/small_test_set_new_prog_ids.pickle", "rb") as f:
            self.test_prog_ids_to_idx = pickle.load(f)
            self.test_idx_to_prog_ids = dict(zip(self.test_prog_ids_to_idx.values(), self.test_prog_ids_to_idx.keys()))

        with open(testing_path + "/small/small_curated_test_sets.pickle", "rb") as f:
            self.curated_test_sets = pickle.load(f)
            print(self.curated_test_sets.keys())

        with open(data_dir_path + "/train_test_sets/curated_test_sets.pickle", "rb") as f:
            self.all_curated_test_sets = pickle.load(f)
            print(self.all_curated_test_sets.keys())

        with open(data_dir_path + "/train_test_sets/dataset_creator.pickle", "rb") as f:
            self.dataset_creator = pickle.load(f)

        self.save_mcmc_progs = save_mcmc_progs
        self.posterior_dists = {}
        self.avg_metrics = {}

        self.category_names = self.curated_test_sets.keys()

        self.curated_test_sets_idxs = {}
        for category in self.curated_test_sets:
            self.curated_test_sets_idxs[category] = {}
            self.posterior_dists[category] = {}
            self.avg_metrics[category] = {}
            for label in self.curated_test_sets[category].keys():
                self.curated_test_sets_idxs[category][label] = set([])
                self.posterior_dists[category][label] = {}
                self.avg_metrics[category][label] = {}
                for data_point in self.curated_test_sets[category][label]:
                    dp0 = self.test_prog_ids_to_idx[data_point[0]]
                    dp1 = self.dataset_creator.ga.node2vocab[data_point[1]]
                    dp2 = data_point[2]
                    if category != MIN_EQ and category != MAX_EQ and category != RAND:
                        dp2 = self.dataset_creator.ga.node2vocab[data_point[2]]
                    self.curated_test_sets_idxs[category][label].add((dp0, dp1, dp2))

        self.metric_labels = [JACC_API, JACC_SEQ, AST_EQ, IN_SET, MIN_DIFF_CS, MIN_DIFF_ST]

    def get_mcmc_prog_and_ast(self, data_point, category, in_random_order, num_apis_to_add_to_constraint):
        prog_id = data_point[0]
        constraints = [data_point[1]]
        dp2 = data_point[2]
        exclude = []
        min_length = 0
        max_length = np.inf

        if category == IN_API or category == IN_CS:
            constraints.append(dp2)
        elif category == EX_CS or category == EX_API:
            exclude.append(dp2)
        elif category == MIN_EQ:
            min_length = dp2
        elif category == MAX_EQ:
            max_length = dp2
        else:
            print("ERROR: CATEGORY NOT ALLOWED")

        # print(data_point)

        nodes, edges, targets, return_type, fp, fp_targets = self.test_ga.fetch_all_list_data_without_delim(prog_id)
        ast = (nodes, edges, targets)

        ordered_apis = self.get_nonconstraint_apis_in_prog(constraints, ast, in_random_order)
        constraints += ordered_apis[:num_apis_to_add_to_constraint]
        print(constraints)
        # init MCMCProgram
        mcmc_prog = MCMCProgram(self.model_dir_path)
        mcmc_prog.init_program(constraints, return_type, fp, exclude=exclude, min_length=min_length,
                               max_length=max_length, ordered=False)

        return mcmc_prog, ast, return_type, fp

    def dump_metrics(self, category, label):
        with open(self.exp_dir_path + "/metrics_" + category + "_" + label + ".json", "w+") as f:
            f.write(json.dumps(self.avg_metrics[category][label]))
            f.close()

    def dump_posterior_distributions(self, category, label):
        with open(self.exp_dir_path + "/post_dist_" + category + "_" + label + ".json", "w+") as f:
            f.write(json.dumps(self.posterior_dists[category][label]))
            f.close()

    def run_mcmc(self, category, label, in_random_order=True, num_apis_to_add_to_constraint=PLUS_0):
        post_dist_dict = self.posterior_dists[category][label]
        test_progs = self.curated_test_sets_idxs[category][label]
        print(len(test_progs))

        for data_point in test_progs:
            mcmc_prog, ast, ret_type, fp = self.get_mcmc_prog_and_ast(data_point, category, in_random_order,
                                                                      num_apis_to_add_to_constraint)

            for _ in self.num_iter:
                mcmc_prog.mcmc()

            post_dist_dict = self.add_to_post_dist(post_dist_dict, get_str_posterior_distribution(mcmc_prog),
                                                   data_point, ast, ret_type, fp)

        self.posterior_dists[category][label] = post_dist_dict
        print("here")
        self.calculate_metrics(category, label)
        self.dump_metrics(category, label)
        self.dump_posterior_distributions(category, label)

    def add_to_post_dist(self, post_dist_dict, posterior_dist, data_point, ast, ret_type, fp):
        post_dist_dict[data_point] = (posterior_dist, ast, ret_type, fp)
        return post_dist_dict

    def calculate_metrics(self, category, label):
        post_dist_dict = self.posterior_dists[category][label]

        metrics = {}
        for metric in self.metric_labels:
            metrics[metric] = 0.0

        # jaccard_api_metric = 0.0
        # jaccard_seq_metric = 0.0
        # ast_eq_metric = 0.0
        # in_dataset_metric = 0.0
        # min_diff_statements_metric = 0.0
        # min_diff_cs_metric = 0.0

        if label == SEEN:
            ga = self.train_ga
        else:
            ga = self.all_test_ga

        for posterior_dist, ast, ret_type, fp in post_dist_dict.values():
            jaccard_api, jaccard_seq, ast_eq, min_diff_cs, min_diff_statements, in_dataset = \
                self.metrics.get_all_averaged_metrics(posterior_dist, ast, ret_type, fp, ga)

            metrics[JACC_API] += jaccard_api
            metrics[JACC_SEQ] += jaccard_seq
            metrics[AST_EQ] += ast_eq
            metrics[IN_SET] += in_dataset
            metrics[MIN_DIFF_ST] += min_diff_statements
            metrics[MIN_DIFF_CS] += min_diff_cs

        self.avg_metrics[category][label] = metrics

    def get_nonconstraint_apis_in_prog(self, constraints, ast, in_random_order):
        apis = set(ast[NODES_IDX])
        apis.update(set(ast[TARGETS_IDX]))
        apis.difference_update(constraints)

        if in_random_order:
            random.shuffle(list(apis))
        else:  # ascending order in terms of api frequency in training set
            apis = list(apis)
            apis = sorted(apis, key=lambda x: self.train_ga.g.nodes[x]['frequency'])
        return list(apis)

    





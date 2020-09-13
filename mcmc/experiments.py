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
MIN_DIFF_ST = 'min_diff_statements'
MIN_DIFF_CS = "min_diff_control_structs"

IN_SET = "in_set"
MEET_CONST = "meets_constraints"
HAS_MORE_APIS = "has_more_apis"
JACC_TEST_SET = "jaccard_test_set"
REL_ADD = "relevant_additions"

RAND = 'random'
ASC = 'ascending'
PLUS_0 = 0
PLUS_1 = 1
PLUS_2 = 2
PLUS_3 = 3

INCLUDE = 'include'
EXCLUDE = 'exclude'
MIN_LENGTH = 'min_length'
MAX_LENGTH = 'max_length'


class Metrics:
    def __init__(self, num_iterations, top_k=10):
        self.num_iter = num_iterations
        self.top_k = top_k

    def get_top_k_gen_progs(self, posterior_dist):
        posterior = posterior_dist.items()
        posterior = sorted([(i[0], i[1][1]) for i in posterior], reverse=True, key=lambda x: x[1])  # TODO: CHECK IF THIS METRIC MAKES SENSE
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
        # print("data point:", data_point)
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
            return 1.0 * len(set_a.intersection(set_b)) / len(set_a.union(set_b))
        except ZeroDivisionError:
            print("get_jaccard_distance: ZeroDivisionError but shouldn't be here!")
            print("len set_a:", len(set_a), "len set_b:", len(set_b), "len(set_a & set_b):", len(set_a & set_b),
                  "len(set_a | set_b):", len(set_a | set_b))
            return 0.0

    def get_all_metrics(self, posterior_dist, ret_type, fp, constraint_dict, test_ga, train_ga):
        # jaccard_api = self.jaccard_api(test_data_point, posterior_dist)
        # jaccard_seq = self.jaccard_sequence(test_data_point, posterior_dist)
        # ast_eq = self.ast_equivalence_top_k(test_data_point, posterior_dist)
        # min_diff_cs = self.abs_min_diff_control_structs_top_k(test_data_point, posterior_dist)
        # min_diff_statements = self.abs_min_diff_statements_top_k(test_data_point, posterior_dist)

        metrics = {}

        metrics[IN_SET] = self.appears_in_set(posterior_dist, ret_type, fp, test_ga)
        metrics[MEET_CONST] = self.meets_constraints(posterior_dist, constraint_dict)
        metrics[REL_ADD] = self.relevant_additions(posterior_dist, constraint_dict, train_ga)
        metrics[HAS_MORE_APIS] = self.has_more_apis(posterior_dist, constraint_dict)
        metrics[JACC_TEST_SET] = self.jaccard_distance_test_set(posterior_dist, constraint_dict, test_ga)

        return metrics

    def get_all_averaged_metrics(self, posterior_dist, ret_type, fp, constraint_dict, test_ga, train_ga):
        # jaccard_api, jaccard_seq, ast_eq, min_diff_cs, min_diff_statements, in_dataset = self.get_all_metrics(
        #     test_data_point, posterior_dist, ret_type, fp, constraint_dict, ga)
        #
        # # TODO: CHANGE!!!!
        # return jaccard_api / self.num_iter, jaccard_seq / self.num_iter, ast_eq / self.num_iter, \
        #     min_diff_cs / self.num_iter, min_diff_statements / self.num_iter, in_dataset / self.num_iter

        all_metrics = self.get_all_metrics(posterior_dist, ret_type, fp, constraint_dict, test_ga, train_ga)
        print(all_metrics)
        for metric in all_metrics:
            all_metrics[metric] /= self.num_iter

        return all_metrics

    def meets_constraints(self, posterior_dist, constraint_dict):
        posterior = self.get_top_k_gen_progs(posterior_dist)

        def get_score(prog):
            score = True
            apis = self.get_apis_set(prog)
            score = score and set(constraint_dict[INCLUDE]).issubset(apis)
            score = score and not set(constraint_dict[EXCLUDE]).issubset(apis)
            score = score and constraint_dict[MIN_LENGTH] <= len(prog[0])
            score = score and len(prog[0]) <= constraint_dict[MAX_LENGTH]
            return int(score)

        return sum([get_score(i) for i in posterior]) * 1.0 / self.top_k

    def has_more_apis(self, posterior_dist, constraint_dict):
        posterior = self.get_top_k_gen_progs(posterior_dist)

        def get_score(prog):
            return int(len(prog[0]) > len(constraint_dict[INCLUDE]))
        # TODO: FIGURE OUT HOW TO ACCOUNT FOR CFS
        return sum([get_score(i) for i in posterior]) * 1.0 / self.top_k

    def get_post_dict_from_str(self, posterior_dist):
        post_dist = {}
        print(type(posterior_dist))
        for str_prog in posterior_dist:
            prog = json.loads(str_prog)
            post_dist[prog] = posterior_dist[str_prog]
        return post_dist

    def jaccard_distance_test_set(self, posterior_dist, constraint_dict, ga):
        test_progs_meet_constraints = ga.get_programs_with_multiple_apis(constraint_dict[INCLUDE],
                                                                         exclude=constraint_dict[EXCLUDE],
                                                                         min_length=constraint_dict[MIN_LENGTH],
                                                                         max_length=constraint_dict[MAX_LENGTH])
        test_progs_meet_constraints = set([tuple(self.get_apis_set((i[0][1], i[1][1], i[2][1]))) for i in test_progs_meet_constraints])
        print(test_progs_meet_constraints)
        # posterior_dist = self.get_post_dict_from_str(posterior_dist)
        posterior = self.get_top_k_gen_progs(posterior_dist)
        api_sets = set([tuple(self.get_apis_set(i)) for i in posterior])
        return self.get_jaccard_distance(api_sets, test_progs_meet_constraints)

    def relevant_additions(self, posterior_dist, constraint_dict, ga):
        # posterior_dist = self.get_post_dict_from_str(posterior_dist)
        posterior = self.get_top_k_gen_progs(posterior_dist)
        api_sets = set([tuple(self.get_apis_set(i)) for i in posterior])
        num_progs = len(api_sets)

        avg_score = 0.0
        for prog in api_sets:
            added_apis = list(set(prog).difference(set(constraint_dict[INCLUDE])))
            num_added_apis = len(added_apis)
            score = 0.0
            for added_api in added_apis:
                api_appears_with_constraint = False
                for constraint in constraint_dict[INCLUDE]:
                    api_appears_with_constraint = api_appears_with_constraint or len(
                        ga.get_program_ids_with_multiple_apis([added_api, constraint])) > 0
                score += int(api_appears_with_constraint)
            avg_score += 1.0 * score / num_added_apis
        return 1.0 * avg_score / num_progs


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

        # self.metric_labels = [JACC_API, JACC_SEQ, AST_EQ, IN_SET, MIN_DIFF_CS, MIN_DIFF_ST]

        self.metric_labels = [JACC_TEST_SET, HAS_MORE_APIS, REL_ADD, MEET_CONST, IN_SET]

    def get_mcmc_prog_and_ast(self, data_point, category, in_random_order, num_apis_to_add_to_constraint, verbose=False):
        prog_id = data_point[0]
        constraints = [data_point[1]]
        dp2 = data_point[2]
        exclude = []
        min_length = 1
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
        mcmc_prog = MCMCProgram(self.model_dir_path, verbose=verbose)
        mcmc_prog.init_program(constraints, return_type, fp, exclude=exclude, min_length=min_length,
                               max_length=max_length, ordered=False)

        constraint_dict = {INCLUDE: constraints, EXCLUDE: exclude, MIN_LENGTH: min_length, MAX_LENGTH: max_length}

        return mcmc_prog, ast, return_type, fp, constraint_dict

    def dump_metrics(self, category, label):
        with open(self.exp_dir_path + "/metrics_" + category + "_" + label + ".json", "w+") as f:
            f.write(json.dumps(self.avg_metrics[category][label]))
            f.close()

    def dump_posterior_dist(self, category, label):
        # print(self.posterior_dists[category][label])
        # str_post_dist = {}
        # for key in self.posterior_dists[category][label]:
        #     str_key = []
        #     for i in key:
        #         str_key.append(str(key))
        #     str_key = str(tuple(str_key))
        #     str_post_dist[str_key] = self.posterior_dists[category][label][key]
        with open(self.exp_dir_path + "/post_dist_" + category + "_" + label + ".pickle", "wb") as f:
            # f.write(json.dumps(str_post_dist))
            # f.close()
            pickle.dump(self.posterior_dists[category][label], f)
            f.close()

    def run_mcmc(self, category, label, in_random_order=True, num_apis_to_add_to_constraint=PLUS_0, verbose=False):
        post_dist_dict = self.posterior_dists[category][label]
        test_progs = self.curated_test_sets_idxs[category][label]
        print(len(test_progs))


        counter = 0
        for data_point in test_progs:
            mcmc_prog, ast, ret_type, fp, constraint_dict = self.get_mcmc_prog_and_ast(data_point, category, in_random_order,
                                                                      num_apis_to_add_to_constraint, verbose=verbose)

            for _ in range(int(self.num_iter)):
                mcmc_prog.mcmc()
            print(get_str_posterior_distribution(mcmc_prog))
            post_dist_dict = self.add_to_post_dist(post_dist_dict, get_str_posterior_distribution(mcmc_prog),
                                                   data_point, ast, ret_type, fp, constraint_dict)

            counter += 1
            if counter == 2:
                break

        self.posterior_dists[category][label] = post_dist_dict
        print("here")
        self.calculate_metrics(category, label)
        self.dump_metrics(category, label)
        self.dump_posterior_dist(category, label)

    def add_to_post_dist(self, post_dist_dict, posterior_dist, data_point, ast, ret_type, fp, constraint_dict):
        post_dist_dict[data_point] = (posterior_dist, ast, ret_type, fp, constraint_dict)
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

        train_ga = self.train_ga
        test_ga = self.all_test_ga

        for posterior_dist, ast, ret_type, fp, constraint_dict in post_dist_dict.values():
            print(posterior_dist)
            print(ast)
            print(ret_type)
            print(fp)
            print(constraint_dict)
            prog_metrics = \
                self.metrics.get_all_averaged_metrics(posterior_dist, ret_type, fp, constraint_dict, test_ga, train_ga)

            print(prog_metrics)

            # metrics[JACC_API] += jaccard_api
            # metrics[JACC_SEQ] += jaccard_seq
            # metrics[AST_EQ] += ast_eq
            # metrics[IN_SET] += in_dataset
            # metrics[MIN_DIFF_ST] += min_diff_statements
            # metrics[MIN_DIFF_CS] += min_diff_cs

            metrics[IN_SET] += prog_metrics[IN_SET]
            metrics[JACC_TEST_SET] += prog_metrics[JACC_TEST_SET]
            metrics[HAS_MORE_APIS] += prog_metrics[HAS_MORE_APIS]
            metrics[REL_ADD] += metrics[REL_ADD]
            metrics[MEET_CONST] += metrics[MEET_CONST]

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

    






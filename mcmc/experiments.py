import argparse
import json
import math
import os
import pickle
import random
import textwrap

import ijson
import tensorflow as tf

import numpy as np

from data_extractor.graph_analyzer import GraphAnalyzer, MIN_EQ, MAX_EQ
from test_utils import get_str_posterior_distribution
from data_extractor.dataset_creator import IN_CS, IN_API, EX_API, EX_CS, SEEN, NEW

from mcmc import MCMCProgram
from infer import BayesianPredictor

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
TEST_REL_ADD = "test_set_relevant_additions"

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


def get_apis_set(data_point):
    # print("data point:", data_point)
    apis = set(data_point[NODES_IDX])
    apis.update(set(data_point[TARGETS_IDX]))
    return apis


class Metrics:
    def __init__(self, num_iterations, stop_num, top_k=5, verbose=False):
        self.num_iter = num_iterations
        self.top_k = int(min(top_k, num_iterations))
        self.stop_num = stop_num
        self.verbose = verbose
        if verbose:
            print("METRICS: Top k:", self.top_k, "Num iterations:", num_iterations)

    def get_top_k_gen_progs(self, posterior_dist, constraint_dict):
        if self.verbose:
            print("\n------ GET TOP K GEN PROGS --------\n")
            print("constraint dict:", constraint_dict)
        def get_non_starting_programs(x):
            is_non_starting = len(x[0][NODES_IDX]) > len(constraint_dict[INCLUDE]) + 1
            if len(x[0][TARGETS_IDX]) == len(constraint_dict[INCLUDE]) + 2:
                is_non_starting = is_non_starting and x[0][TARGETS_IDX][-1] != self.stop_num
            return is_non_starting

        posterior = list(posterior_dist.items())
        filtered_posterior = list(filter(lambda x: get_non_starting_programs(x), posterior))
        if self.verbose:
            print("posterior:", posterior)
            print("filtered posterior:", filtered_posterior)
        if len(filtered_posterior) != 0:
            posterior = filtered_posterior

        # print(posterior)
        if type(posterior[0][1]) == list or type(posterior[0][1]) == tuple:
            posterior = sorted([(i[0], i[1][1]) for i in posterior], reverse=True,
                               key=lambda x: x[1])  # TODO: CHECK IF THIS METRIC MAKES SENSE
        else:
            posterior = sorted(posterior, reverse=True, key=lambda x: x[1])
        # else:
        #     raise ValueError("posterior item value should have only 1 or 2 elements in it")

        top_k = min(self.top_k, len(posterior))
        posterior = posterior[:top_k]
        posterior = [i[0] for i in posterior]
        return posterior

    # def jaccard_api(self, test_data_point, posterior_dist):
    #     test_apis_set = self.get_apis_set(test_data_point)
    #     posterior = self.get_top_k_gen_progs(posterior_dist)
    #     posterior = set([self.get_apis_set(i) for i in posterior])
    #     return self.get_jaccard_distance(test_apis_set, posterior)
    #
    # def jaccard_sequence(self, test_data_point, posterior_dist):
    #     def get_api_seq_from_data_point(data_point):
    #         sequence = []
    #         for i in range(len(data_point[NODES_IDX])):
    #             for idx in [NODES_IDX, TARGETS_IDX]:
    #                 api = data_point[idx][i]
    #                 if api not in {DLOOP, DEXCEPT, DBRANCH, DSUBTREE, DSTOP} and api not in sequence:
    #                     sequence.append(api)
    #         return tuple(sequence)
    #
    #     test_sequence_set = {get_api_seq_from_data_point(test_data_point)}
    #     posterior = self.get_top_k_gen_progs(posterior_dist)
    #     posterior = set([get_api_seq_from_data_point(i) for i in posterior])
    #     return self.get_jaccard_distance(test_sequence_set, posterior)
    #
    # def ast_equivalence_top_k(self, test_data_point, posterior_dist):
    #     return test_data_point in self.get_top_k_gen_progs(posterior_dist)
    #
    # def abs_min_diff_statements_top_k(self, test_data_point, posterior_dist):
    #     top_k_diff = self.get_top_k_gen_progs(posterior_dist)
    #     top_k_diff = [abs(len(gen_data_point[NODES_IDX]) - len(test_data_point[NODES_IDX])) for gen_data_point in
    #                   top_k_diff]
    #     return 1.0 * min(top_k_diff)
    #
    # def abs_min_diff_control_structs_top_k(self, test_data_point, posterior_dist):
    #     def get_num_control_structures(data_point):
    #         return sum([int(api in {DBRANCH, DLOOP, DEXCEPT}) for api in data_point[NODES_IDX]])
    #
    #     top_k_diff = self.get_top_k_gen_progs(posterior_dist)
    #     top_k_diff = [abs(get_num_control_structures(gen_data_point) - get_num_control_structures(test_data_point)) for
    #                   gen_data_point in top_k_diff]
    #     return 1.0 * min(top_k_diff)

    def get_jaccard_distance(self, set_a, set_b):
        try:
            return 1 - (1.0 * len(set_a.intersection(set_b)) / len(set_a.union(set_b)))
        except ZeroDivisionError:
            print("get_jaccard_distance: ZeroDivisionError but shouldn't be here!")
            print("len set_a:", len(set_a), "len set_b:", len(set_b), "len(set_a & set_b):", len(set_a & set_b),
                  "len(set_a | set_b):", len(set_a | set_b))
            return 0.0

    def get_all_metrics(self, posterior_dist, ret_type, fp, constraint_dict, test_ga, train_ga):
        metrics = {}

        metrics[IN_SET] = self.appears_in_set(posterior_dist, constraint_dict, ret_type, fp, test_ga)
        metrics[MEET_CONST] = self.meets_constraints(posterior_dist, constraint_dict)
        metrics[REL_ADD] = self.relevant_additions(posterior_dist, constraint_dict, train_ga)
        metrics[TEST_REL_ADD] = self.relevant_additions(posterior_dist, constraint_dict, test_ga, is_test_ga=True)
        metrics[HAS_MORE_APIS] = self.has_more_apis(posterior_dist, constraint_dict)
        metrics[JACC_TEST_SET] = self.jaccard_distance_test_set(posterior_dist, constraint_dict, test_ga)

        return metrics

    def get_all_averaged_metrics(self, posterior_dist, ret_type, fp, constraint_dict, test_ga, train_ga,
                                 num_test_progs):

        all_metrics = self.get_all_metrics(posterior_dist, ret_type, fp, constraint_dict, test_ga, train_ga)
        if self.verbose:
            print("all metrics before avg:", all_metrics)
        for metric in all_metrics:
            all_metrics[metric] /= num_test_progs

        return all_metrics

    def appears_in_set(self, posterior_dist, constraint_dict, ret_type, fp, ga):
        posterior_progs = self.get_top_k_gen_progs(posterior_dist, constraint_dict)

        if self.verbose:
            print("\n\n\n--------APPEARS IN SET----------")
            print("posterior top k:", posterior_progs)
            print("num in posterior top k:", len(posterior_progs))
            print("ret type:", ret_type)
            print("fp:", fp)

        in_set = False
        for gen_data_point in posterior_progs:
            apis = get_apis_set(gen_data_point)

            prog_ids = ga.get_program_ids_with_multiple_apis(list(apis))
            for prog_id in prog_ids:
                prog = ga.fetch_all_list_data_without_delim(prog_id)
                in_set = in_set or (prog[:5] == (
                    list(gen_data_point[NODES_IDX]), list(gen_data_point[EDGES_IDX]), list(gen_data_point[TARGETS_IDX]),
                    ret_type, list(fp)))

        if self.verbose:
            print("return:", 1.0 * int(in_set))

        return 1.0 * int(in_set)

    def meets_constraints(self, posterior_dist, constraint_dict):
        posterior = self.get_top_k_gen_progs(posterior_dist, constraint_dict)

        if self.verbose:
            print("\n\n\n--------MEETS CONSTRAINTS----------")
            print("posterior top k:", posterior)
            print("num in posterior top k:", len(posterior))

        def get_score(prog):
            # print(prog)
            score = True
            apis = get_apis_set(prog)

            if self.verbose:
                print("")
                print("apis in prog", apis)
                print("constraint apis", constraint_dict[INCLUDE])
                print("include:", set(constraint_dict[INCLUDE]).issubset(apis))
                print("")
            score = score and set(constraint_dict[INCLUDE]).issubset(apis)
            if len(constraint_dict[EXCLUDE]) != 0:
                if self.verbose:
                    print("exclude:", not set(constraint_dict[EXCLUDE]).issubset(apis))
                score = score and not set(constraint_dict[EXCLUDE]).issubset(apis)
            # if self.verbose:
            # print(constraint_dict[MIN_LENGTH], len(prog[0]))
            # print("min length:", constraint_dict[MIN_LENGTH] <= len(prog[0]))
            score = score and constraint_dict[MIN_LENGTH] <= len(prog[0]) + 1
            if self.verbose:
                print("max length:", len(prog[0]) <= constraint_dict[MAX_LENGTH] - 1)
            score = score and len(prog[0]) <= constraint_dict[MAX_LENGTH] - 1
            return int(score)

        scores = [get_score(i) for i in posterior]
        if self.verbose:
            print("scores:", scores)
            print("return:", sum(scores) * 1.0 / len(posterior))
        try:
            final_score = sum(scores) * 1.0 / len(posterior)
        except ZeroDivisionError:
            final_score = 0.0
        return final_score

    def has_more_apis(self, posterior_dist, constraint_dict):
        posterior = self.get_top_k_gen_progs(posterior_dist, constraint_dict)
        posterior = [get_apis_set(i) for i in posterior]

        if self.verbose:
            print("\n\n\n--------HAS_MORE_APIS----------")
            print("posterior len:", len(posterior))

        def get_score(prog):
            prog.discard(DSUBTREE)
            prog.discard(DSTOP)
            if self.verbose:
                print("")
                print("prog:", prog)
                print("constraints:", constraint_dict[INCLUDE])
                print("")
            return int(len(prog) > len(set(constraint_dict[INCLUDE])) and len(prog) > constraint_dict[MIN_LENGTH])

        scores = [get_score(i) for i in posterior]

        if self.verbose:
            print("scores:", scores)
            # TODO: FIGURE OUT HOW TO ACCOUNT FOR CFS
            print("return:", sum(scores) * 1.0 / len(posterior))
        try:
            final_score = sum(scores) * 1.0 / len(posterior)
        except ZeroDivisionError:
            final_score = 0.0
        return final_score

    def jaccard_distance_test_set(self, posterior_dist, constraint_dict, ga):
        test_progs_meet_constraints = ga.get_programs_with_multiple_apis(constraint_dict[INCLUDE],
                                                                         exclude=constraint_dict[EXCLUDE],
                                                                         min_length=constraint_dict[MIN_LENGTH],
                                                                         max_length=constraint_dict[MAX_LENGTH])

        if self.verbose:
            print("\n\n\n--------JACCARD DISTANCE TEST SET----------")
            print("num test progs that meet constraint:", len(test_progs_meet_constraints))
        test_progs_meet_constraints = \
            [get_apis_set((i[0][1], i[1][1], i[2][1])) for i in test_progs_meet_constraints]

        if self.verbose:
            print("apis of test progs:", test_progs_meet_constraints)
        posterior = self.get_top_k_gen_progs(posterior_dist, constraint_dict)
        api_sets = [get_apis_set(i) for i in posterior]

        if self.verbose:
            print("apis in posterior", api_sets)
        jacc_distances = []
        for api_set in api_sets:
            for test_set in test_progs_meet_constraints:
                api_set.discard('DSubTree')
                api_set.discard('DStop')
                test_set.discard('DSubTree')
                test_set.discard('DStop')
                jacc_dist = self.get_jaccard_distance(api_set, test_set)
                if self.verbose:
                    print("api set:", api_set)
                    print("jacc distance:", jacc_dist)
                jacc_distances.append(jacc_dist)

        if self.verbose:
            print("return:", min(jacc_distances))
            print("")
        return min(jacc_distances)

    def relevant_additions(self, posterior_dist, constraint_dict, ga, is_test_ga=False):
        posterior = self.get_top_k_gen_progs(posterior_dist, constraint_dict)
        api_sets = set([tuple(get_apis_set(i)) for i in posterior])
        num_progs = len(api_sets)

        if self.verbose:
            print("\n\n\n--------RELEVANT ADDITIONS: " + "is test ga: " + str(is_test_ga) + "----------")
            print("top k api sets:", api_sets)
            print("num api sets:", num_progs)

        avg_score = 0.0
        for prog in api_sets:
            added_apis = list(set(prog).difference(set(constraint_dict[INCLUDE])) - {DSUBTREE, DSTOP})

            if self.verbose:
                print("added apis:", added_apis)
            num_added_apis = len(added_apis)
            if num_added_apis == 0:
                continue
            score = 0.0
            for added_api in added_apis:
                api_appears_with_constraint = False
                for constraint in constraint_dict[INCLUDE]:
                    api_appears_with_constraint = api_appears_with_constraint or len(
                        ga.get_program_ids_with_multiple_apis([added_api, constraint])) > 0

                    if self.verbose:
                        print("added_api:", added_api)
                        print("constraint:", constraint, "appears:", len(
                            ga.get_program_ids_with_multiple_apis([added_api, constraint])) > 0)
                        print("")

                score += int(api_appears_with_constraint)
                if self.verbose:
                    print("score for ", added_api, " :", score)
            avg_score += 1.0 * score / num_added_apis

        if self.verbose:
            print("return:", 1.0 * avg_score / num_progs)
        return 1.0 * avg_score / num_progs


class Experiments:
    def __init__(self, data_dir_name, model_dir_path, experiment_dir_name, num_iterations, save_mcmc_progs=True,
                 train_test_set_dir_name='/train_test_sets/', verbose=False):

        self.num_iter = num_iterations * 1.0
        self.model_dir_path = model_dir_path
        self.data_dir_name = data_dir_name

        experiments_path = os.path.dirname(os.path.realpath(__file__)) + "/experiments"

        if not os.path.exists(experiments_path):
            os.mkdir(experiments_path)

        if not os.path.exists(experiments_path + "/" + experiment_dir_name):
            os.mkdir(experiments_path + "/" + experiment_dir_name)

        self.exp_dir_path = experiments_path + "/" + experiment_dir_name

        self.all_ga = GraphAnalyzer(data_dir_name, load_reader=True, shuffle_data=False,
                                    train_test_set_dir_name=train_test_set_dir_name)

        self.train_ga = GraphAnalyzer(data_dir_name, train_test_split='train', filename='all_training_data',
                                      load_reader=True, shuffle_data=False,
                                      train_test_set_dir_name=train_test_set_dir_name)
        self.all_test_ga = GraphAnalyzer(data_dir_name, train_test_split='test', filename='test_set', load_reader=True,
                                         shuffle_data=False,
                                         train_test_set_dir_name=train_test_set_dir_name)  # TODO: fix
        self.test_ga = GraphAnalyzer(data_dir_name, train_test_split='small_test', filename='small_test_set',
                                     load_reader=True, shuffle_data=False,
                                     train_test_set_dir_name=train_test_set_dir_name)

        data_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     "../data_extractor/data/" + data_dir_name)
        testing_path = data_dir_path + "/" + train_test_set_dir_name + "test"

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

        with open(data_dir_path + train_test_set_dir_name + "curated_test_sets.pickle", "rb") as f:
            self.all_curated_test_sets = pickle.load(f)
            print(self.all_curated_test_sets.keys())

        with open(data_dir_path + train_test_set_dir_name + "dataset_creator.pickle", "rb") as f:
            self.dataset_creator = pickle.load(f)

        self.save_mcmc_progs = save_mcmc_progs
        self.posterior_dists = {}
        self.avg_metrics = {}
        self.avg_bayou_metrics = {}

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
                    if type(data_point[1]) == int:
                        dp1 = tuple([self.dataset_creator.ga.node2vocab[data_point[1]]])
                    else:
                        dp1 = tuple([self.dataset_creator.ga.node2vocab[i] for i in data_point[1]])
                    dp2 = data_point[2]
                    if category != MIN_EQ and category != MAX_EQ and category != RAND:
                        dp2 = self.dataset_creator.ga.node2vocab[data_point[2]]
                    self.curated_test_sets_idxs[category][label].add((dp0, dp1, dp2))

        self.metrics = Metrics(num_iterations * 1.0, self.all_ga.vocab2node[DSTOP], verbose=verbose)
        self.metric_labels = [JACC_TEST_SET, HAS_MORE_APIS, REL_ADD, MEET_CONST, IN_SET, TEST_REL_ADD]

        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(""))
        parser.add_argument('--continue_from', type=str, default=model_dir_path,
                            help='ignore config options and continue training model checkpointed here')
        clargs = parser.parse_args()

        self.encoder = BayesianPredictor(clargs.continue_from, batch_size=1)
        beam_width = 1
        self.decoder = BayesianPredictor(clargs.continue_from, depth='change', batch_size=beam_width)

        self.verbose = verbose

        print(self.posterior_dists)

    def get_constraint_dict(self, data_point, category, convert_from_node=False):
        if convert_from_node:
            dp0 = self.test_prog_ids_to_idx[data_point[0]]
            if type(data_point[1]) == int:
                dp1 = tuple([self.dataset_creator.ga.node2vocab[data_point[1]]])
            else:
                dp1 = tuple([self.dataset_creator.ga.node2vocab[i] for i in data_point[1]])
            dp2 = data_point[2]
            if category != MIN_EQ and category != MAX_EQ and category != RAND:
                dp2 = self.dataset_creator.ga.node2vocab[data_point[2]]
            dp_copy = list(data_point)
            dp_copy[0] = dp0
            dp_copy[1] = dp1
            dp_copy[2] = dp2
            data_point = dp_copy

        constraints = list(data_point[1])
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
            raise ValueError("ERROR: CATEGORY NOT ALLOWED")

        constraint_dict = {INCLUDE: constraints, EXCLUDE: exclude, MIN_LENGTH: min_length, MAX_LENGTH: max_length}

        return constraint_dict

    def get_mcmc_prog_and_ast(self, data_point, category, in_random_order, num_apis_to_add_to_constraint, verbose=False,
                              session=None, proposal_probs=None):
        verbose = verbose or self.verbose

        prog_id = data_point[0]
        nodes, edges, targets, return_type, fp, fp_targets = self.test_ga.fetch_all_list_data_without_delim(prog_id)
        ast = (nodes, edges, targets)

        constraint_dict = self.get_constraint_dict(data_point, category)
        constraints = constraint_dict[INCLUDE]

        # train_prog_ids = self.train_ga.get_programs_with_multiple_apis(constraint_dict[INCLUDE], exclude=constraint_dict[EXCLUDE])
        # print("should be zero:")
        # for prog_id in train_prog_ids:
        #     print(prog_id)
        #     # print(self.train_ga.fetch_all_list_data_without_delim(prog_id))
        # print("\n\ntraining data:")
        # train_prog_ids = self.train_ga.get_programs_with_multiple_apis(constraint_dict[INCLUDE])
        # for prog_id in train_prog_ids:
        #     # print(self.train_ga.fetch_all_list_data_without_delim(prog_id))
        #     print(prog_id)
        #
        # print("test progs with all constraints:")
        # test_prog_ids = self.all_test_ga.get_programs_with_multiple_apis(constraint_dict[INCLUDE], exclude=constraint_dict[EXCLUDE])
        # for prog_id in test_prog_ids:
        #     print(prog_id)
        # print("\n\nnot accurate test data:")
        # test_prog_ids = self.all_test_ga.get_programs_with_multiple_apis(constraint_dict[INCLUDE])
        # for prog_id in test_prog_ids:
        #     print(prog_id)

        def check_ast_meets_constraints(prog):
            score = True
            apis = get_apis_set(prog)
            if verbose:
                print("")
                print("apis in prog", apis)
                print("constraint apis", constraint_dict[INCLUDE])
                print(set(constraint_dict[INCLUDE]).issubset(apis))
                print("")
            score = score and set(constraint_dict[INCLUDE]).issubset(apis)
            if len(constraint_dict[EXCLUDE]) != 0:
                if verbose:
                    print("exclude:", not set(constraint_dict[EXCLUDE]).issubset(apis))
                score = score and not set(constraint_dict[EXCLUDE]).issubset(apis)
            if verbose:
                print("min length:", constraint_dict[MIN_LENGTH] <= len(prog[0]))
            score = score and constraint_dict[MIN_LENGTH] <= len(prog[0])
            if verbose:
                print("max length:", len(prog[0]) <= constraint_dict[MAX_LENGTH])
            score = score and len(prog[0]) <= constraint_dict[MAX_LENGTH]
            return score

        assert check_ast_meets_constraints(ast) is True

        def order_include_constraints(ast, constraints):
            ordered_constraints = []
            for i in range(len(ast[NODES_IDX])):
                for api in [ast[NODES_IDX][i], ast[TARGETS_IDX][i]]:
                    if api in constraints:
                        constraints.remove(api)
                        ordered_constraints.append(api)
                    if len(constraints) == 0:
                        return ordered_constraints
            assert len(constraints) == 0
            return ordered_constraints

        constraints = order_include_constraints(ast, constraints)
        constraint_dict[INCLUDE] = constraints

        # init MCMCProgram
        mcmc_prog = MCMCProgram(self.model_dir_path, verbose=verbose, session=session, encoder=self.encoder,
                                decoder=self.decoder)

        if proposal_probs is not None:
            mcmc_prog.proposal_probs = proposal_probs

        mcmc_prog.init_program(constraints, return_type, fp, exclude=constraint_dict[EXCLUDE], min_length=constraint_dict[MIN_LENGTH],
                               max_length=constraint_dict[MAX_LENGTH], ordered=True)
        # mcmc_prog = None

        return mcmc_prog, ast, return_type, fp, constraint_dict

    def get_test_prog_id(self, train_prog_id):
        return self.dataset_creator.test_set_new_prog_ids[train_prog_id]

    def dump_metrics(self, category, label):
        with open(self.exp_dir_path + "/metrics_" + category + "_" + label + ".json", "w+") as f:
            f.write(json.dumps(self.avg_metrics[category][label]))
            f.close()

    def dump_posterior_dist(self, category, label):
        with open(self.exp_dir_path + "/post_dist_" + category + "_" + label + ".pickle", "wb") as f:
            pickle.dump(self.posterior_dists[category][label], f)
            f.close()

    def run_mcmc(self, category, label, save_run=True, num_test_progs=None, in_random_order=True,
                 num_apis_to_add_to_constraint=PLUS_0, verbose=False, test_prog_range=None, use_gpu=False,
                 use_xla=False, save_step=5, proposal_probs=None):
        verbose = verbose or self.verbose

        post_dist_dict = self.posterior_dists[category][label]
        test_progs = list(self.curated_test_sets_idxs[category][label])
        test_progs = list(sorted(test_progs, key=lambda x: x[0]))

        analysis_f = open(self.exp_dir_path + "/" + category + "_" + label + "_analysis.txt", "w+")
        analysis_f.write("data dir name: " + self.data_dir_name + "\n")
        analysis_f.write("model dir path: " + self.model_dir_path + "\n")
        analysis_f.write("num iterations: " + str(self.num_iter) + "\n")
        analysis_f.write("category: " + category + "\n")
        analysis_f.write("label: " + label + "\n")

        if test_prog_range is not None:
            test_progs = test_progs[test_prog_range[0]: test_prog_range[1]]

        if num_test_progs is None:
            num_test_progs = len(test_progs)

        analysis_f.write("num test programs: " + str(num_test_progs) + "\n\n")
        analysis_f.flush()
        os.fsync(analysis_f.fileno())

        def write_to_skipped_file(skipped_dp):
            skipped_dp_f = open(self.exp_dir_path + "/" + category + "_" + label + "_skipped_dp.pickle", "wb")
            pickle.dump(skipped_dp, skipped_dp_f)
            skipped_dp_f.close()

        skipped_dp = {}
        num_skipped_dp = 0

        # Launch the graph
        config = tf.ConfigProto(
            device_count={'GPU': 0 if not use_gpu else 1},
        )

        if use_gpu:
            config.gpu_options.allow_growth = True
            config.gpu_options.force_gpu_compatible = use_gpu

        if use_xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        session = tf.Session(config=config)

        for i in range(num_test_progs):
            data_point = test_progs[i]
            try:
                print("\n\n--------------i:", i, "data_point:", data_point)

                mcmc_prog, ast, ret_type, fp, constraint_dict = self.get_mcmc_prog_and_ast(data_point, category,
                                                                                           in_random_order,
                                                                                           num_apis_to_add_to_constraint,
                                                                                           verbose=verbose,
                                                                                           session=session,
                                                                                           proposal_probs=proposal_probs)

                for j in range(int(self.num_iter)):
                    if verbose:
                        print("iteration:", j)
                    mcmc_prog.mcmc(j)

                self.add_to_post_dist(category, label, get_str_posterior_distribution(mcmc_prog),
                                      data_point, ast, ret_type, fp, constraint_dict)

                if i % save_step == 0 and save_run:
                    analysis_f.write("\n")
                    analysis_f.write("counter: " + str(i) + "\n")
                    analysis_f.write("num skipped: " + str(num_skipped_dp) + "\n")
                    analysis_f.write(str(constraint_dict))
                    analysis_f.write("\n")
                    analysis_f.write(str(post_dist_dict[data_point][0]))
                    analysis_f.write("\n")
                    analysis_f.flush()
                    os.fsync(analysis_f.fileno())
                    self.posterior_dists[category][label] = post_dist_dict
                    self.dump_posterior_dist(category, label)
                    try:
                        self.calculate_metrics(category, label, num_test_progs)
                        self.dump_metrics(category, label)
                    except Exception as e:
                        print("Error calculating metrics:", str(e))

            except Exception as e:
                num_skipped_dp += 1
                skipped_dp[data_point] = str(e)
                write_to_skipped_file(skipped_dp)

        self.posterior_dists[category][label] = post_dist_dict

        if save_run:
            self.dump_posterior_dist(category, label)
            try:
                self.calculate_metrics(category, label, num_test_progs)
                print("\n\nFINAL METRICS:", self.avg_metrics[category][label])
                self.dump_metrics(category, label)
            except Exception as e:
                print("Error calculating metrics:", str(e))

        analysis_f.write("\n\nfinal metrics: " + str(self.avg_metrics[category][label]) + "\n")
        analysis_f.flush()
        os.fsync(analysis_f.fileno())
        analysis_f.close()

    def add_to_post_dist(self, category, label, posterior_dist, data_point, ast, ret_type, fp, constraint_dict):
        self.posterior_dists[category][label][data_point] = (posterior_dist, ast, ret_type, fp, constraint_dict)

    def calculate_metrics(self, category, label, num_test_progs=None):
        post_dist_dict = self.posterior_dists[category][label]

        if num_test_progs is None:
            num_test_progs = len(post_dist_dict)

        print("num test progs:", num_test_progs)

        metrics = {}
        for metric in self.metric_labels:
            metrics[metric] = 0.0

        train_ga = self.train_ga
        test_ga = self.all_test_ga

        for posterior_dist, ast, ret_type, fp, constraint_dict in post_dist_dict.values():
            prog_metrics = \
                self.metrics.get_all_averaged_metrics(posterior_dist, ret_type, fp, constraint_dict, test_ga, train_ga,
                                                      num_test_progs)

            metrics[IN_SET] += prog_metrics[IN_SET]
            metrics[JACC_TEST_SET] += prog_metrics[JACC_TEST_SET]
            metrics[HAS_MORE_APIS] += prog_metrics[HAS_MORE_APIS]
            metrics[REL_ADD] += prog_metrics[REL_ADD]
            metrics[TEST_REL_ADD] += prog_metrics[TEST_REL_ADD]
            metrics[MEET_CONST] += prog_metrics[MEET_CONST]

        self.avg_metrics[category][label] = metrics

        print("\n\nFINAL METRICS for", category, " ", label, ":", self.avg_metrics[category][label])

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

    def load_posterior_distribution_from_file(self, dir_path, category, label):
        try:
            dir_path += "/"
            post_dist_file_path = os.path.join(dir_path, "post_dist_" + category + "_" + label + ".pickle")
            post_dist_f = open(post_dist_file_path, "rb")
            post_dist = dict(pickle.load(post_dist_f))
            try:
                self.posterior_dists[category][label].update(post_dist)
                print("Successfully loaded posterior distribution into class:", post_dist_file_path)
                post_dist_f.close()
            except KeyError as e:
                print("Could not load posterior distribution:", e)
                post_dist_f.close()
        except FileNotFoundError as e:
            print("Could not load posterior distribution:", e)

    def load_bayou_data_calculate_metrics(self, bayou_categories_dir_path, bayou_categories_filename,
                                          small_test_set_path, all_json_data_path, save=False, load_rt_fp=True):
        small_test_set = pickle.load(open(small_test_set_path, "rb"))

        rt = 'returnType'
        fp = 'formalParam'

        if load_rt_fp:
            rt_fp_f = open(bayou_categories_dir_path + "/" + bayou_categories_filename + "_rt_fp.pickle", "rb")
            ret_type_fp = pickle.load(rt_fp_f)
        else:
            ret_type_fp = {}
            for category in small_test_set:
                for novelty in small_test_set[category]:
                    for dp in small_test_set[category][novelty]:
                        prog_id = dp[0]
                        ret_type_fp[prog_id] = {rt: [], fp: []}

            all_data_f = open(all_json_data_path, "r")
            counter = 0
            for program in ijson.items(all_data_f, 'programs.item'):
                if counter in ret_type_fp:
                    ret_type_fp[counter][rt] = program[rt]
                    ret_type_fp[counter][fp] = program[fp]
                counter += 1
                if counter % 100000 == 0:
                    print("Loaded", counter, "items")

            all_data_f.close()

            if save:
                rt_fp_f = open(bayou_categories_dir_path + "/" + bayou_categories_filename + "_rt_fp.pickle", "wb")
                pickle.dump(ret_type_fp, rt_fp_f)

        print("Loaded return types and formal parameters")

        categories_f = open(bayou_categories_dir_path + "/" + bayou_categories_filename + ".pickle", "rb")
        categories = pickle.load(categories_f)
        all_metrics = {}
        train_ga = self.train_ga
        test_ga = self.all_test_ga

        for category in [MIN_EQ]:
            print("\n\nCATEGORY:", category)
            metrics = all_metrics[category] = {}
            for metric in self.metric_labels:
                metrics[metric] = 0.0

            num_test_progs = len(categories[category])
            num_progs_skipped = 0

            counter = 0
            for dp_key in categories[category]:
                datapoint = categories[category][dp_key]
                posterior_dist = datapoint['posterior_dist']
                if len(posterior_dist) == 0:
                    counter += 1  # TODO: think about this
                    num_progs_skipped += 1
                    print("num progs skipped:", num_progs_skipped)
                    continue
                prog_id = dp_key[0]
                ret_type = ret_type_fp[prog_id][rt]
                formal_param = ret_type_fp[prog_id][fp]
                assert set(datapoint['types']) == set([ret_type] + formal_param)
                constraint_dict = self.get_constraint_dict(dp_key, category, convert_from_node=True)
                assert set(datapoint['apicalls']) == set(constraint_dict[INCLUDE])

                prog_metrics = \
                    self.metrics.get_all_metrics(posterior_dist, ret_type, formal_param, constraint_dict, test_ga,
                                                          train_ga)

                metrics[IN_SET] += prog_metrics[IN_SET]
                metrics[JACC_TEST_SET] += prog_metrics[JACC_TEST_SET]
                metrics[HAS_MORE_APIS] += prog_metrics[HAS_MORE_APIS]
                metrics[REL_ADD] += prog_metrics[REL_ADD]
                metrics[TEST_REL_ADD] += prog_metrics[TEST_REL_ADD]
                metrics[MEET_CONST] += prog_metrics[MEET_CONST]

                counter += 1
                if counter % 50 == 0:
                    print("num calculated:", counter)
                    total_progs = counter
                    print("in set:", metrics[IN_SET] / total_progs)
                    print("has more apis:", metrics[HAS_MORE_APIS] / total_progs)
                    print("rel add:", metrics[REL_ADD] / total_progs)
                    print("test rel add:", metrics[TEST_REL_ADD] / total_progs)
                    print("meet const:", metrics[MEET_CONST] / total_progs)

            total_progs = counter
            metrics[IN_SET] /= total_progs
            metrics[JACC_TEST_SET] /= total_progs
            metrics[HAS_MORE_APIS] /= total_progs
            metrics[REL_ADD] /= total_progs
            metrics[TEST_REL_ADD] /= total_progs
            metrics[MEET_CONST] /= total_progs
            self.avg_metrics[category] = metrics

            print("\n\nFINAL METRICS for", category,  ":", self.avg_metrics[category], "\n\n")

        categories_f.close()



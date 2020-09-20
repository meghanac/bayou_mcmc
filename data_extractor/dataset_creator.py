import itertools
import json
import pickle
import time
import random
import os

import ijson
import numpy as np
from data_extractor.graph_analyzer import GraphAnalyzer, ALL_DATA_NO_DUP, MAX_AST_DEPTH, MIN_EQ, MAX_EQ, MIN, MAX, EQ
import networkx as nx
import math


API = 'api'
CS = 'control_structure'
LEN = 'length'
IN = 'include'
EX = 'exclude'
IN_API = 'include_api'
EX_API = 'exclude_api'
IN_CS = 'include_cs'
EX_CS = 'exclude_cs'
HIGH = 'high'
MID = 'mid'
LOW = 'low'
FREQ = 'frequency'
WEIGHT = 'weight'
DP2_API = [IN_API, EX_API]
DP2_CS = [IN_CS, EX_CS]
RANDOM = 'random'

SEEN = 'accuracy'
NEW = 'novelty'

DBRANCH = 'DBranch'
DLOOP = 'DLoop'
DEXCEPT = 'DExcept'
DSTOP = 'DStop'
DSUBTREE = 'DSubTree'

PID = 0
API = 1
DP2 = 2
LEN = 3


class DatasetCreator:
    """

    graph analyzer + database on:
    - 1k vocab dataset
    - entire dataset
    """
    def __init__(self, data_dir_path, train_test_set_name, save_reader=False, min_prog_per_category=1200, verbose=False, test_mode=True):
        self.data_dir_path = data_dir_path

        if save_reader:
            self.ga = GraphAnalyzer(data_dir_path, save_reader=True, shuffle_data=False,
                                    load_g_without_control_structs=False, pickle_friendly=True)
        else:
            self.ga = GraphAnalyzer(data_dir_path, load_reader=True, load_g_without_control_structs=False,
                                    pickle_friendly=True, shuffle_data=False)

        for i in range(10):
            print(self.ga.fetch_data_with_targets(i))
            print([self.ga.node2vocab[j] for j in self.ga.fetch_nodes_as_list(i)])
            print(self.ga.get_json_ast(i))
            print("\n")

        # self.training_data = set(range(self.ga.num_programs))
        # self.test_data = set([])
        # self.novelty_test_set = set([])
        # self.accuracy_test_set = set([])

        self.ranks = sorted(nx.get_node_attributes(self.ga.g, 'frequency').items(), key=lambda x: x[1], reverse=True)
        self.ranks = list(filter(lambda x: x[0] in self.ga.vocab2node, self.ranks))
        self.ranks = [(self.ranks[i][0], (i, self.ranks[i][1])) for i in range(len(self.ranks))]
        self.ranks_dict = dict(self.ranks)
        self.ranks = [i[0] for i in self.ranks]
        self.ranks.remove(DBRANCH)
        self.ranks.remove(DLOOP)
        self.ranks.remove(DEXCEPT)

        self.num_apis = len(self.ranks)

        self.include_api_set = {SEEN: set([]), NEW: set([])}
        self.include_cs_set = {SEEN: set([]), NEW: set([])}
        self.exclude_api_set = {SEEN: set([]), NEW: set([])}
        self.exclude_cs_set = {SEEN: set([]), NEW: set([])}
        self.min_api_set = {SEEN: set([]), NEW: set([])}
        self.max_api_set = {SEEN: set([]), NEW: set([])}
        self.random_progs_set = {HIGH: set([]), MID: set([]), LOW: set([])}

        self.categories = {IN_API: (self.include_api_set, API), EX_API: (self.exclude_api_set, API),
                           IN_CS: (self.include_cs_set, CS), EX_CS: (self.exclude_cs_set, CS),
                           MAX_EQ: (self.max_api_set, LEN), MIN_EQ: (self.min_api_set, LEN),
                           RANDOM: (self.random_progs_set, None)}

        self.high_range = (0, math.floor(self.num_apis / 3))
        self.mid_range = (math.floor(self.num_apis / 3), math.floor(self.num_apis * 2/3))
        self.low_range = (math.floor(self.num_apis * 2/3), self.num_apis)
        self.full_range = (0, self.num_apis)

        self.freq_pairs = [(LOW, LOW), (MID, LOW), (HIGH, LOW), (HIGH, MID), (HIGH, HIGH)]
        self.idx_ranges = {HIGH: self.full_range, MID: self.full_range, LOW: self.full_range}

        # for category in self.categories:
        #     test_set = self.categories[category][0]
        #     for freq_pair in self.freq_pairs:
        #         test_set[NEW][freq_pair] = set([])
        #         test_set[SEEN][freq_pair] = set([])

        # if verbose:
        #     print(self.high_range)
        #     print(self.mid_range)
        #     print(self.low_range)

        self.min_prog_per_category = min_prog_per_category
        self.min_prog_per_freq = math.ceil(min_prog_per_category / 3)
        self.min_pairs_per_freq = math.ceil(self.min_prog_per_freq / 2)

        self.control_structs = [DBRANCH, DEXCEPT, DLOOP]

        self.test_api_to_prog_ids = {}
        self.test_set_new_prog_ids = None

        self.control_limit = 0.5

        self.verbose = verbose
        self.test_mode = test_mode

        self.dir_path = self.ga.dir_path + "/" + train_test_set_name

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        self.analysis_f = open(self.dir_path + "/analysis.txt", "w+")

    def add_include_or_exclude_test_progs(self, added_to_test_set_func, category, novelty_label):
        assert category in DP2_API or category in DP2_CS, "Error: category must be " + IN_CS + ", " + IN_API + ", " \
                                                          + EX_API + ", " + ", or " + EX_CS
        api_idx_range = list(range(self.full_range[0], self.full_range[1]))
        non_apis = [self.ga.vocab2node[i] for i in [DBRANCH, DLOOP, DEXCEPT, DSTOP, DSUBTREE]]
        for non_api in non_apis:
            api_idx_range.remove(non_api)
        random.shuffle(api_idx_range)

        self.analysis_f.write("category: " + category + " label: " + novelty_label + "\n")
        #
        # if category in DP2_API:
        #     dp2_idx_range = list(range(self.full_range[0], self.full_range[1]))
        #     random.shuffle(dp2_idx_range)
        #     data_points = self.ranks
        # else:
        #     dp2_idx_range = range(len(self.control_structs))
        #     data_points = self.control_structs

        test_set = self.categories[category][0][novelty_label]

        num_pairs_added = 0

        # all_possible_data_points = itertools.product(api_idx_range, dp2_idx_range)

        # def suitable_candidate(idxs):
        #     progs_with_both = len(
        #         self.ga.get_program_ids_with_multiple_apis([self.ranks[idxs[0]], data_points[idxs[1]]]))
        #     progs_with_api = len(self.ga.get_program_ids_for_api(self.ranks[idxs[0]]))
        #     progs_with_dp2 = len(self.ga.get_program_ids_for_api(data_points[idxs[1]]))
        #     return 1 < progs_with_both < progs_with_api and progs_with_dp2 > progs_with_both
        #
        start_time = time.time()
        # valid_data_points = filter(lambda idxs: suitable_candidate(idxs), all_possible_data_points)

        if self.test_mode:
            print("Time taken for valid_data_points:", start_time - time.time())

        added_apis = set([])

        counter = 0
        start_time = time.time()

        for api_idx in api_idx_range:
            api = self.ranks[api_idx]

            if api in added_apis or api in {'DSubTree', 'DStop', DBRANCH, DEXCEPT, DLOOP}:
                continue

            progs_with_api = self.ga.get_program_ids_for_api(api)

            apis_with_api = [self.ga.get_apis_in_prog_set(i) for i in progs_with_api]
            apis_with_api = set([y for x in apis_with_api for y in x])

            if category == IN_API:
                added_api_nums = list(added_apis.copy())
                added_api_nums = set([self.ga.vocab2node[i] for i in added_api_nums])
                non_apis = set([self.ga.vocab2node[i] for i in [api, 'DSubTree', 'DStop', DBRANCH, DLOOP, DEXCEPT]])
                apis_with_api = list(apis_with_api.difference(added_api_nums).difference(non_apis))
            else:
                apis_with_api = list(
                    apis_with_api.intersection(set([self.ga.vocab2node[i] for i in [DBRANCH, DLOOP, DEXCEPT]])))

            random.shuffle(apis_with_api)
            for dp2_idx in apis_with_api:
                dp2 = data_point2 = self.ga.node2vocab[dp2_idx]

                if api == dp2 or dp2 in {'DSubTree', 'DStop'}:
                    continue

                if category in DP2_API and dp2 in added_apis:
                    continue
                # if len(self.ga.get_program_ids_with_multiple_apis([api, dp2])) == 0:
                #     continue

                progs_with_dp2 = self.ga.get_program_ids_for_api(dp2)
                progs_with_both = progs_with_api.intersection(progs_with_dp2)

                if len(progs_with_both) == len(progs_with_api.union(progs_with_dp2)):
                    continue

                if not (2 < len(progs_with_both) < len(progs_with_api) and len(progs_with_dp2) > len(progs_with_both) > 2):
                    continue

                apis_with_dp2 = [self.ga.get_apis_in_prog_set(i) for i in progs_with_dp2]
                apis_with_dp2 = set([y for x in apis_with_dp2 for y in x])

                counter += 1
                if counter % 5000 == 0:
                    print("counter:", counter)
                    print("num pairs added:", num_pairs_added)
                    print("num progs added:", len(test_set))
                    self.analysis_f.write("counter: " + str(counter) + " num pairs added: " + str(
                        num_pairs_added) + " num progs added: " + str(len(test_set)) + "\n")
                    self.analysis_f.flush()
                    os.fsync(self.analysis_f.fileno())
                    if self.test_mode:
                        print("time taken:", start_time - time.time())
                        start_time = time.time()

                if api != data_point2 and api not in added_apis and not (category in DP2_API and data_point2 in added_apis):

                    # if not self.ga.g.has_edge(api, data_point2):
                    #     raise ValueError("shouldn't be here!!!")

                    if added_to_test_set_func(api, data_point2, novelty_label, test_set, progs_with_api, progs_with_dp2,
                                              progs_with_both):
                        print(api, dp2, len(progs_with_both))
                        # if api in {DBRANCH, DLOOP, DEXCEPT} or dp2 in {DBRANCH, DLOOP, DEXCEPT}:
                        #     print("----------HERE")
                        #     print("api:", api)
                        #     print("")
                        num_pairs_added += 1
                        added_apis.add(api)
                        if category in DP2_API:
                            added_apis.add(data_point2)
                        break

            if num_pairs_added >= self.min_prog_per_category:
                break

        print("Category:", category)
        print("Novelty label: " + novelty_label)
        print("Num API + " + category + " pairs added: " + str(num_pairs_added))
        print("Total programs added: " + str(len(test_set)))

    def added_to_include_test_set(self, api, data_point2, novelty_label, test_set, progs_with_api, progs_with_dp2,
                                  progs_with_both):

        num_progs_with_both = len(progs_with_both)
        num_progs_with_api = len(progs_with_api)
        num_progs_with_dp2 = len(progs_with_dp2)

        if data_point2 in {DBRANCH, DLOOP, DEXCEPT}:
            max_progs = 50
        else:
            max_progs = 500

        if self.verbose and self.test_mode:
            print("num progs with both:", num_progs_with_both)
            print("num progs with api:", num_progs_with_api)
            print("num progs with dp2:", num_progs_with_dp2)
            print("control api:", self.control_limit * num_progs_with_api)
            print("control dp2:", self.control_limit * num_progs_with_dp2)

        if num_progs_with_both <= self.control_limit * num_progs_with_api and \
                num_progs_with_both <= self.control_limit * num_progs_with_dp2:
            if novelty_label == NEW and 0 < num_progs_with_both <= max_progs:
                prog_ids = progs_with_both

                if len(prog_ids) == 0:
                    return False

                if self.verbose:
                    print("api:", api, "data_point:", data_point2)
                    print("num progs added:", len(prog_ids))
                self.add_to_test_set([api], data_point2, prog_ids, test_set)
                return True

            if novelty_label == SEEN:
                if data_point2 in {DBRANCH, DLOOP, DEXCEPT}:
                    upper_limit = 10
                else:
                    upper_limit = 15
                limit = min(math.ceil(num_progs_with_both / 4), self.control_limit * num_progs_with_api, upper_limit)
                prog_ids = set(itertools.islice(progs_with_both, limit))

                if len(prog_ids) == 0:
                    return False

                if self.verbose:
                    print("api:", api, "data_point:", data_point2)
                    print("num progs added:", len(prog_ids))
                self.add_to_test_set([api], data_point2, prog_ids, test_set)
                return True

        return False

    def add_exclude_test_progs(self, category, novelty_label):
        assert category == EX_CS or category == EX_API
        api_idx_range = list(range(self.full_range[0], self.full_range[1]))
        non_apis = [self.ga.vocab2node[i] for i in [DBRANCH, DLOOP, DEXCEPT, DSTOP, DSUBTREE]]
        for non_api in non_apis:
            api_idx_range.remove(non_api)
        random.shuffle(api_idx_range)

        self.analysis_f.write("category: " + category + " label: " + novelty_label + "\n")

        # all_idx_range = list(range(self.full_range[0], self.full_range[1] + len(self.control_structs)))
        #
        # if category in DP2_API:
        #     dp2_idx_range = list(range(self.full_range[0], self.full_range[1]))
        #     random.shuffle(dp2_idx_range)
        #     data_points = self.ranks
        # else:
        #     dp2_idx_range = range(len(self.control_structs))
        #     data_points = self.control_structs

        test_set = self.categories[category][0][novelty_label]

        num_pairs_added = 0
        min_test_set_progs = 2

        # all_possible_data_points = itertools.product(api_idx_range, all_idx_range, dp2_idx_range)

        def suitable_candidate(idxs):
            # if idxs[0] == idxs[1] or idxs[1] == idxs[2] or idxs[2] == idxs[0]:
            #     return False
            #
            # if idxs[1] in range(self.full_range[1]):
            #     i3 = self.ranks[idxs[1]]
            # else:
            #     i3 = self.control_structs[idxs[1] % self.full_range[1]]
            # api = self.ranks[idxs[0]]
            # dp2 = data_points[idxs[2]]

            api = idxs[0]
            i3 = idxs[1]
            dp2 = idxs[2]

            progs_with_all = len(
                self.ga.get_program_ids_with_multiple_apis([api, i3, dp2]))
            progs_with_api = len(self.ga.get_program_ids_for_api(api))
            progs_with_dp2 = len(self.ga.get_program_ids_for_api(dp2))
            progs_with_i3 = len(self.ga.get_program_ids_for_api(i3))
            progs_with_api_i3 = len(self.ga.get_program_ids_with_multiple_apis([api, i3], exclude=[dp2]))
            progs_with_dp2_i3 = len(self.ga.get_program_ids_with_multiple_apis([dp2, i3], exclude=[api]))
            progs_with_api_dp2 = len(self.ga.get_program_ids_with_multiple_apis([api, dp2], exclude=[i3]))

            num_training_set_progs = progs_with_api + progs_with_dp2 + progs_with_all + progs_with_i3 \
                + progs_with_api_i3 + progs_with_dp2_i3
            num_test_set_progs = progs_with_api_dp2

            valid = num_test_set_progs > 1 and num_training_set_progs > 1
            return valid
        #
        # start_time = time.time()
        # valid_data_points = filter(lambda idxs: suitable_candidate(idxs), all_possible_data_points)
        #
        # if self.test_mode:
        #     print("Time taken for valid_data_points:", start_time - time.time())

        added_apis = set([])

        counter = 0
        start_time = time.time()
        # for api_idx, i3_idx, dp_idx in valid_data_points:

        for api_idx in api_idx_range:
            api = self.ranks[api_idx]
            if api in added_apis or api in {'DSubTree', 'DStop', DBRANCH, DEXCEPT, DLOOP}:
                continue

            api_prog_ids = self.ga.get_program_ids_for_api(api)


            apis_with_api = [self.ga.get_apis_in_prog_set(i) for i in api_prog_ids]
            apis_with_api = set([y for x in apis_with_api for y in x])

            # if category == EX_API:

            added_api_nums = list(added_apis.copy())
            added_api_nums = set([self.ga.vocab2node[i] for i in added_api_nums])
            non_apis = set([self.ga.vocab2node[i] for i in [api, 'DSubTree', 'DStop']])
            apis_with_api = list(apis_with_api.difference(added_api_nums).difference(non_apis))

            # else:
            #     apis_with_api = list(
            #         apis_with_api.intersection(set([self.ga.vocab2node[i] for i in [DBRANCH, DLOOP, DEXCEPT]])))

            random.shuffle(apis_with_api)
            # print(apis_with_api)

            for i2_idx in apis_with_api:
                i2 = self.ga.node2vocab[i2_idx]

                if api == i2 or i2 in {'DSubTree', 'DStop'}:
                    continue

                if i2 in added_apis:
                    continue
                # if len(self.ga.get_program_ids_with_multiple_apis([api, dp2])) == 0:
                #     continue

                i2_prog_ids = self.ga.get_program_ids_for_api(i2)

                if len(api_prog_ids.intersection(i2_prog_ids)) == len(api_prog_ids.union(i2_prog_ids)):
                    continue

                if len(api_prog_ids.intersection(i2_prog_ids)) <= min_test_set_progs:
                    continue

                apis_with_i2 = [self.ga.get_apis_in_prog_set(i) for i in i2_prog_ids]
                apis_with_i2 = set([y for x in apis_with_i2 for y in x])

                if category == EX_API:
                    added_api_nums = list(added_apis.copy())
                    added_api_nums = set([self.ga.vocab2node[i] for i in added_api_nums])
                    non_apis = set([self.ga.vocab2node[i] for i in [api, i2, 'DSubTree', 'DStop', DBRANCH, DLOOP, DEXCEPT]])
                    apis_with_both = list(set(apis_with_api).union(apis_with_i2).difference(added_api_nums).difference(non_apis))
                else:
                    apis_with_both = list(
                        set(apis_with_api).union(apis_with_i2).intersection(set([self.ga.vocab2node[i] for i in [DBRANCH, DLOOP, DEXCEPT]])))

                random.shuffle(apis_with_both)

                for i3_idx in apis_with_both:
                    i3 = self.ga.node2vocab[i3_idx]

                    if category in DP2_API and i3 in added_apis or i3 in {'DSubTree', 'DStop'}:
                        continue

                    if i3 == api or i3 == i2:
                        continue

                    if category in DP2_CS and i3 not in {DBRANCH, DEXCEPT, DLOOP}:
                        print("\n\n\n\n ----------------------------- ERROR")
                        continue

                    i3_prog_ids = self.ga.get_program_ids_for_api(i3)

                    progs_with_api_i2 = api_prog_ids.intersection(i2_prog_ids).difference(i3_prog_ids)
                    if len(progs_with_api_i2) <= min_test_set_progs:
                        continue

                    # don't want to try to create programs with apis that only appear with a CFS
                    if i3 in {DBRANCH, DEXCEPT, DLOOP} and (len(i2_prog_ids.difference(i3_prog_ids)) == 0 or len(
                            api_prog_ids.difference(i3_prog_ids)) == 0):
                        continue

                    counter += 1
                    if counter % 5000 == 0:
                        print("counter:", counter)
                        print("num pairs added:", num_pairs_added)
                        print("num progs added:", len(test_set))
                        self.analysis_f.write("counter: " + str(counter) + " num pairs added: " + str(
                            num_pairs_added) + " num progs added: " + str(len(test_set)) + "\n")
                        self.analysis_f.flush()
                        os.fsync(self.analysis_f.fileno())
                        if self.test_mode:
                            print("time taken:", start_time - time.time())
                            start_time = time.time()


                    if api not in added_apis and not (category in DP2_API and i2 in added_apis) and not (category in DP2_API and i3 in added_apis):
                        if i3 in {DBRANCH, DLOOP, DEXCEPT}:
                            max_progs = 50
                        else:
                            max_progs = 200

                        num_progs_with_only_api = len(api_prog_ids.difference(i2).difference(i3))
                        num_progs_with_only_i2 = len(i2_prog_ids.difference(i3).difference(api))
                        num_progs_with_all = len(api_prog_ids.intersection(i2_prog_ids).intersection(i3_prog_ids))
                        num_progs_with_only_i3 = len(i3_prog_ids.difference(api).difference(i2))
                        num_progs_with_api_i3 = len(api_prog_ids.intersection(i3_prog_ids).difference(i2_prog_ids))
                        num_progs_with_i2_i3 = len(i2_prog_ids.intersection(i3_prog_ids).difference(api_prog_ids))
                        num_progs_with_api_i2 = len(progs_with_api_i2)

                        num_training_set_progs = num_progs_with_only_api + num_progs_with_only_i2 + num_progs_with_all \
                                                 + num_progs_with_only_i3 + num_progs_with_api_i3 + num_progs_with_i2_i3
                        num_test_set_progs = num_progs_with_api_i2

                        if self.verbose and self.test_mode:
                            print("api:", api, "num progs api", num_progs_with_only_api)
                            print("i3:", i3, "num progs i3", num_progs_with_only_i3)
                            print("i2", i2, "num progs i2", num_progs_with_only_i2)
                            print("num progs api, i3", num_progs_with_api_i3)
                            print("num progs i2, i3", num_progs_with_i2_i3)
                            print("num progs all", num_progs_with_all)
                            print("num progs api i2", num_progs_with_api_i2)

                        success = False

                        if num_test_set_progs <= num_training_set_progs * self.control_limit:
                            if novelty_label == NEW and min_test_set_progs < num_test_set_progs <= max_progs:
                                prog_ids = progs_with_api_i2
                                if self.verbose:
                                    print("include:", api, i2, "exclude:", i3)
                                    print("num progs added:", len(prog_ids))

                                if len(prog_ids) != 0:
                                    self.add_to_test_set((api, i2), i3, prog_ids, test_set)
                                    success = True

                            if novelty_label == SEEN:
                                if category == EX_API:
                                    upper_limit = 10
                                else:
                                    upper_limit = 30
                                limit = min(math.ceil(num_test_set_progs / 4), upper_limit)
                                prog_ids = progs_with_api_i2

                                if len(prog_ids) != 0:
                                    prog_ids = itertools.islice(prog_ids, limit)

                                    if self.verbose:
                                        print("include:", api, i2, "exclude:", i3)
                                    self.add_to_test_set((api, i2), i3, prog_ids, test_set)

                                    success = True

                        if success:
                            print(api, i2, i3, num_test_set_progs)
                            num_pairs_added += 1
                            added_apis.add(api)
                            if i2 not in {DBRANCH, DLOOP, DEXCEPT}:
                                added_apis.add(i2)
                            if i3 not in {DBRANCH, DLOOP, DEXCEPT}:
                                added_apis.add(i3)
                            break
                break
            if num_pairs_added >= self.min_prog_per_category:
                break

        print("Category:", category)
        print("Novelty label: " + novelty_label)
        print("Num API + " + category + " pairs added: " + str(num_pairs_added))
        print("Total programs added: " + str(len(test_set)))

    # def added_to_exclude_test_set(self, api, data_point2, novelty_label, test_set, progs_with_api, progs_with_dp2,
    #                               progs_with_both):
    #     num_progs_with_both = len(progs_with_both)
    #     num_progs_with_api = len(progs_with_api)
    #     num_progs_with_dp2 = len(progs_with_dp2)
    #
    #     if data_point2 in {DBRANCH, DLOOP, DEXCEPT}:
    #         max_progs = 50
    #     else:
    #         max_progs = 200
    #
    #     if self.verbose and self.test_mode:
    #         print("num progs api", num_progs_with_api)
    #         print("num progs dp2", num_progs_with_dp2)
    #         print("num progs both", num_progs_with_both)
    #         print("control limit:", num_progs_with_api * self.control_limit)
    #         print("difference:", num_progs_with_api - num_progs_with_both)
    #     if num_progs_with_api - num_progs_with_both <= num_progs_with_api * self.control_limit:
    #         if novelty_label == NEW and 0 < num_progs_with_api - num_progs_with_both <= max_progs:
    #             prog_ids = progs_with_api - progs_with_both
    #             if self.verbose:
    #                 print("api:", api, "data_point:", data_point2)
    #                 print("num progs added:", len(prog_ids))
    #
    #             if len(prog_ids) == 0:
    #                 return False
    #
    #             self.add_to_test_set(api, data_point2, prog_ids, test_set)
    #             return True
    #
    #         if novelty_label == SEEN:
    #             limit = min(math.ceil((num_progs_with_api - num_progs_with_both) / 4),
    #                         self.control_limit * num_progs_with_api, 10)
    #             prog_ids = progs_with_api - progs_with_both
    #
    #             if len(prog_ids) == 0:
    #                 return False
    #
    #             prog_ids = itertools.islice(prog_ids, limit)
    #
    #             if self.verbose:
    #                 print("api:", api, "data_point:", data_point2)
    #             self.add_to_test_set(api, data_point2, prog_ids, test_set)
    #             return True
    #
    #     return False

    def add_to_test_set(self, include_api_list, data_point2, prog_ids_set, test_set, dp2type_is_int=False):
        api_num = []
        for i in include_api_list:
            api_num.append(self.ga.vocab2node[i])

        if dp2type_is_int:
            dp_num = data_point2
        else:
            dp_num = self.ga.vocab2node[data_point2]

        for prog_id in prog_ids_set:
            test_set.add((prog_id, tuple(api_num), dp_num))
            apis = self.ga.get_apis_in_prog_set(prog_id)
            for api in apis:
                self.ga.api_to_prog_ids[api].discard(prog_id)
                if api in self.test_api_to_prog_ids:
                    self.test_api_to_prog_ids[api].add(prog_id)
                else:
                    self.test_api_to_prog_ids[api] = {prog_id}

    def add_length_constrained_test_progs(self, category, novelty_label):
        assert category == MAX_EQ or category == MIN_EQ

        # for i in range(self.num_apis):
        #     api = self.ranks[i]
        #     print("num total:", len(self.ga.get_program_ids_for_api(api)))
        #     prog_ids = self.ga.get_program_ids_for_api_length_k(api, EQ, 2)
        #     print("with k:", len(prog_ids))
        #     prog_ids = [self.ga.fetch_data(prog_id) for prog_id in prog_ids]
        #     for nodes in prog_ids:
        #         print(nodes)
        self.analysis_f.write("category: " + category + " label: " + novelty_label + "\n")

        api_idx_range = list(range(self.full_range[0], self.full_range[1]))
        non_apis = [self.ga.vocab2node[i] for i in [DBRANCH, DLOOP, DEXCEPT, DSTOP, DSUBTREE]]
        for non_api in non_apis:
            api_idx_range.remove(non_api)
        random.shuffle(api_idx_range)

        len_range = list(range(3, 8))
        random.shuffle(len_range)

        test_set = self.categories[category][0][novelty_label]

        num_pairs_added = 0

        all_possible_data_points = itertools.product(api_idx_range, len_range)

        def suitable_candidate(idxs):
            progs_with_both = len(
                self.ga.get_program_ids_for_api_length_k(self.ranks[idxs[0]], category, idxs[1]))
            progs_with_api = len(self.ga.get_program_ids_for_api(self.ranks[idxs[0]]))
            # if self.verbose and self.test_mode:
            #     print("num progs with both:", progs_with_both)
            #     print("num progs with api:", progs_with_api)
            return 1 < progs_with_both < progs_with_api

        start_time = time.time()
        valid_data_points = filter(lambda idxs: suitable_candidate(idxs), all_possible_data_points)

        if self.test_mode:
            print("Time taken for valid_data_points:", start_time - time.time())

        if category == MIN_EQ:
            max_progs = 50
        else:
            max_progs = 200

        added_apis = set([])

        counter = 0
        start_time = time.time()
        for api_idx, dp_idx in valid_data_points:
            counter += 1
            if counter % 5000 == 0:
                print("counter:", counter)
                print("num pairs added:", num_pairs_added)
                print("num progs added:", len(test_set))
                self.analysis_f.write("counter: " + str(counter) + " num pairs added: " + str(
                    num_pairs_added) + " num progs added: " + str(len(test_set)) + "\n")
                self.analysis_f.flush()
                os.fsync(self.analysis_f.fileno())
                if self.test_mode:
                    print("time taken:", start_time - time.time())
                    start_time = time.time()

            api = self.ranks[api_idx]
            length = dp_idx

            if api not in added_apis and api not in {DBRANCH, DLOOP, DEXCEPT, 'DSubTree', 'DStop'}:

                valid_prog_ids = self.ga.get_program_ids_for_api_length_k(api, category, length)
                num_valid_progs = len(valid_prog_ids)
                progs_with_api = self.ga.get_program_ids_for_api(api)
                num_progs_with_api = len(progs_with_api)

                if self.verbose and self.test_mode:
                    print("num progs that meet length criteria:", num_valid_progs)
                    print("num progs with api:", num_progs_with_api)
                    print("control api:", self.control_limit * num_progs_with_api)

                if num_valid_progs <= self.control_limit * num_progs_with_api:
                    if novelty_label == NEW and 2 < num_valid_progs <= max_progs:
                        if self.verbose:
                            print("api:", api, "length:", length)
                            print("num progs added:", len(valid_prog_ids))
                        self.add_to_test_set([api], length, valid_prog_ids, test_set, dp2type_is_int=True)
                        num_pairs_added += 1
                        added_apis.add(api)
                        print(api, length)

                    if novelty_label == SEEN:
                        limit = min(math.ceil(num_valid_progs / 4), self.control_limit * num_progs_with_api, 10)
                        prog_ids = set(itertools.islice(valid_prog_ids, limit))

                        if len(prog_ids) == 0:
                            continue

                        if self.verbose:
                            print("api:", api, "length:", length)
                            print("num progs added:", len(prog_ids))

                        self.add_to_test_set([api], length, prog_ids, test_set, dp2type_is_int=True)
                        num_pairs_added += 1
                        added_apis.add(api)

                    if num_pairs_added >= self.min_prog_per_category:
                        break

        print("Category:", category)
        print("Novelty label: " + novelty_label)
        print("Num API + " + category + " pairs added: " + str(num_pairs_added))
        print("len added apis:", len(added_apis))
        print("Total programs added: " + str(len(test_set)))
        return False

    def build_and_save_train_test_sets(self):
        # self.add_random_programs()
        self.create_curated_dataset()
        self.pickle_dump_curated_test_sets()
        self.build_and_save_sets()
        self.pickle_dump_self()

    def add_random_programs(self):
        ranges = {HIGH: (10000, np.inf), MID: (100, 10000), LOW: (2, 100)}
        for freq in [HIGH, MID, LOW]:
            lower_bound, upper_bound = ranges[freq]
            vocab_range = list(filter(lambda x: upper_bound > self.ga.g.nodes[x]['frequency'] > lower_bound, self.ranks))
            selected_vocab_idx = random.choices(vocab_range, k=self.min_prog_per_category)

            for i in range(len(selected_vocab_idx)):
                prog_id = self.ga.get_program_ids_for_api(selected_vocab_idx[i], limit=1)
                self.add_to_test_set([selected_vocab_idx[i]], -1, prog_id, self.random_progs_set[freq],
                                     dp2type_is_int=True)

        print("Number of random programs added:", len(self.random_progs_set))

    def create_curated_dataset(self):
        print("Creating Curated Tests Dataset\n")

        for novelty_label in [NEW]:  # Create novelty test set first

            print("\n\n\n-----------------------------------")
            print("INCLUDE API: ")
            start_time = time.time()
            self.add_include_or_exclude_test_progs(self.added_to_include_test_set, IN_API, novelty_label)
            print("test set len:", len(self.categories[IN_API][0][novelty_label]), "\n")
            if self.test_mode:
                print("Time taken for include api:", start_time - time.time())

            print("\n\n\n-----------------------------------")
            print("INCLUDE CS: ")
            start_time = time.time()
            self.add_include_or_exclude_test_progs(self.added_to_include_test_set, IN_CS, novelty_label)
            print("test set len:", len(self.categories[IN_CS][0][novelty_label]), "\n")
            if self.test_mode:
                print("Time taken for include cs:", start_time - time.time())

            print("\n\n\n-----------------------------------")
            print("EXCLUDE CS: ")
            start_time = time.time()
            self.add_exclude_test_progs(EX_CS, novelty_label)
            print("test set len:", len(self.categories[EX_CS][0][novelty_label]), "\n")
            if self.test_mode:
                print("Time taken for exclude cs:", start_time - time.time())

            print("\n\n\n-----------------------------------")
            print("EXCLUDE API: ")
            start_time = time.time()
            self.add_exclude_test_progs(EX_API, novelty_label)
            print("test set len:", len(self.categories[EX_API][0][novelty_label]), "\n")
            if self.test_mode:
                print("Time taken for exclude api:", start_time - time.time())

            print("\n\n\n-----------------------------------")
            print("MIN LENGTH")
            start_time = time.time()
            self.add_length_constrained_test_progs(MIN_EQ, novelty_label)
            print("test set len:", len(self.categories[MIN_EQ][0][novelty_label]), "\n")
            if self.test_mode:
                print("Time taken for min length:", start_time - time.time())

            print("\n\n\n-----------------------------------")
            print("MAX LENGTH")
            start_time = time.time()
            self.add_length_constrained_test_progs(MAX_EQ, novelty_label)
            print("test set len:", len(self.categories[MAX_EQ][0][novelty_label]), "\n")
            if self.test_mode:
                print("Time taken for max length:", start_time - time.time())

    def get_freq_label(self, api):
        rank = self.ranks.index(api)
        if self.high_range[0] <= rank < self.high_range[1]:
            return HIGH
        elif self.mid_range[0] <= rank < self.mid_range[1]:
            return MID
        else:
            return LOW

    def pickle_dump_curated_test_sets(self, path=None):
        if path is None:
            path = self.dir_path

        if not os.path.exists(path):
            os.mkdir(path)

        with open(path + "/curated_test_sets.pickle", 'wb') as f:
            pickle.dump(self.categories, f)
            f.close()

        if not os.path.exists(path + "/test/"):
            os.mkdir(path + "/test/")

        with open(path + "/test/test_set_new_prog_ids.pickle", 'wb') as f:
            pickle.dump(self.test_set_new_prog_ids, f)
            f.close()

    def pickle_dump_self(self, path=None):
        if path is None:
            path = self.dir_path
        if not os.path.exists(path):
            os.mkdir(path)

        self.ga.json_asts = None
        analysis_f = self.analysis_f
        self.analysis_f = None

        with open(path + "/dataset_creator.pickle", 'wb') as f:
            pickle.dump(self, f)
            f.close()

        self.analysis_f = analysis_f

    def build_and_save_sets(self):
        test_set = set([])
        for category in self.categories:
            cat_test_set = self.categories[category][0]
            for t in cat_test_set.values():
                test_set.update(t)

        test_set = list(test_set)
        test_set_progs = set([i[0] for i in test_set])

        training_set = set([])
        for api in self.ga.api_to_prog_ids:
            training_set.update(self.ga.api_to_prog_ids[api])

        all_progs = test_set_progs.copy()
        all_progs.update(training_set)
        assert all_progs == set(range(0, self.ga.num_programs))

        if not os.path.exists(self.dir_path + "/train"):
            os.mkdir(self.dir_path + "/train")

        if not os.path.exists(self.dir_path + "/test"):
            os.mkdir(self.dir_path + "/test")

        train_f = open(self.dir_path + "/train/training_data.json", "w+")
        test_f = open(self.dir_path + "/test/test_set.json", "w+")

        # start data files
        train_f.write("{\n")
        train_f.write("\"programs\": [\n")
        test_f.write("{\n")
        test_f.write("\"programs\": [\n")

        data_filename = self.ga.clargs.data_filename + ".json"
        data_f = open(os.path.join(self.ga.dir_path, data_filename))
        self.ga.json_asts = ijson.items(data_f, 'programs.item')

        self.test_set_new_prog_ids = {}

        prog_id = 0
        test_prog_counter = 0
        train_prog_counter = 0
        for program in self.ga.json_asts:
            if prog_id in test_set_progs:
                if test_prog_counter != 0:
                    test_f.write(",\n")
                test_f.write(json.dumps(program))

                # update
                self.test_set_new_prog_ids[prog_id] = test_prog_counter
                prog_id += 1
                test_prog_counter += 1

            elif prog_id in training_set:
                if train_prog_counter != 0:
                    train_f.write(",\n")
                train_f.write(json.dumps(program))

                # update
                prog_id += 1
                train_prog_counter += 1

        # end new json data file
        train_f.write("\n")
        train_f.write("]\n")
        train_f.write("}\n")
        test_f.write("\n")
        test_f.write("]\n")
        test_f.write("}\n")
        test_f.close()
        train_f.close()

        print("Added", test_prog_counter, "programs to test set")
        print("Added", train_prog_counter, "programs to training set")


def build_sets_from_saved_creator(data_dir_path, creator_dir_name):
    creator_path = data_dir_path + creator_dir_name + "/dataset_creator.pickle"
    creator_dir_path = data_dir_path + creator_dir_name
    print("\n\n\nBuilding Training and Test Sets\n")
    f = open(creator_path, "rb")
    dataset_creator = pickle.load(f)

    test_set = set([])
    for category in dataset_creator.categories:
        cat_test_set = dataset_creator.categories[category][0]
        for t in cat_test_set.values():
            test_set.update(t)

    test_set = list(test_set)
    test_set_progs = set([i[0] for i in test_set])

    training_set = set([])
    for api in dataset_creator.ga.api_to_prog_ids:
        training_set.update(dataset_creator.ga.api_to_prog_ids[api])

    all_progs = test_set_progs.copy()
    all_progs.update(training_set)
    assert all_progs == set(range(0, dataset_creator.ga.num_programs))

    if not os.path.exists(creator_dir_path + "/train"):
        os.mkdir(creator_dir_path + "/train")

    if not os.path.exists(creator_dir_path + "/test"):
        os.mkdir(creator_dir_path + "/test")

    train_f = open(creator_dir_path + "/train/training_data.json", "w+")
    test_f = open(creator_dir_path + "/test/test_set.json", "w+")

    # start data files
    train_f.write("{\n")
    train_f.write("\"programs\": [\n")
    test_f.write("{\n")
    test_f.write("\"programs\": [\n")

    data_filename = dataset_creator.ga.clargs.data_filename + ".json"
    data_f = open(os.path.join(data_dir_path, data_filename))
    dataset_creator.ga.json_asts = ijson.items(data_f, 'programs.item')

    test_set_new_prog_ids = {}

    prog_id = 0
    test_prog_counter = 0
    train_prog_counter = 0
    for program in dataset_creator.ga.json_asts:
        if prog_id in test_set_progs:
            if test_prog_counter != 0:
                test_f.write(",\n")
            test_f.write(json.dumps(program))

            # update
            test_set_new_prog_ids[prog_id] = test_prog_counter
            prog_id += 1
            test_prog_counter += 1

        elif prog_id in training_set:
            if train_prog_counter != 0:
                train_f.write(",\n")
            train_f.write(json.dumps(program))

            # update
            prog_id += 1
            train_prog_counter += 1

        else:
            print("Not in either set: ", prog_id)
            prog_id += 1

    # end new json data file
    train_f.write("\n")
    train_f.write("]\n")
    train_f.write("}\n")
    test_f.write("\n")
    test_f.write("]\n")
    test_f.write("}\n")
    test_f.close()
    train_f.close()

    print("Added", test_prog_counter, "programs to test set")
    print("Added", train_prog_counter, "programs to training set")

    with open(creator_dir_path + "/test/test_set_new_prog_ids.pickle", 'wb') as f:
        pickle.dump(test_set_new_prog_ids, f)
        f.close()


def pickle_dump_test_sets(data_dir_path, data_dir_name, train_test_set_name):
    creator_dir_path = data_dir_path + "/" + train_test_set_name + "/"
    creator_path = creator_dir_path + "/dataset_creator.pickle"
    print("\n\n\nBuilding Training and Test Sets\n")
    f = open(creator_path, "rb")
    dataset_creator = pickle.load(f)

    test_set = set([])
    for category in dataset_creator.categories:
        cat_test_set = dataset_creator.categories[category][0]
        for t in cat_test_set.values():
            test_set.update(t)

    test_set = list(test_set)
    test_set_progs = set([i[0] for i in test_set])

    training_set = set([])
    for api in dataset_creator.ga.api_to_prog_ids:
        training_set.update(dataset_creator.ga.api_to_prog_ids[api])

    all_progs = test_set_progs.copy()
    all_progs.update(training_set)
    assert all_progs == set(range(0, dataset_creator.ga.num_programs))

    if not os.path.exists(creator_dir_path  + "/train"):
        os.mkdir(creator_dir_path + "/train")

    if not os.path.exists(creator_dir_path  + "/test"):
        os.mkdir(creator_dir_path + "/test")

    data_filename = data_dir_name + ".json"
    data_f = open(os.path.join(data_dir_path, data_filename), "r")
    dataset_creator.ga.json_asts = ijson.items(data_f, 'programs.item')

    test_set_new_prog_ids = {}

    prog_id = 0
    test_prog_counter = 0
    for _ in dataset_creator.ga.json_asts:
        if prog_id in test_set_progs:
            # update
            test_set_new_prog_ids[prog_id] = test_prog_counter
            prog_id += 1
            test_prog_counter += 1

    with open(creator_dir_path + "/test/test_set_new_prog_ids.pickle", 'wb') as f:
        pickle.dump(test_set_new_prog_ids, f)
        f.close()


def create_small_test_set_one_per_api(category, t, test_set, smaller_test_set, prog_ids, min_length):
    curr_api_id = -1
    for i in range(len(test_set)):
        # if on a new API
        if test_set[i][API] != curr_api_id:
            # if prog ID already saved but this is the only program with API, add it
            if test_set[i][PID] in prog_ids:
                if i < len(test_set) - 1 and test_set[i][API] != test_set[i + 1][API]:
                    smaller_test_set[category][t].add(test_set[i])
            else:
                if i < len(test_set) - 1 and test_set[i][API] != test_set[i + 1][API]:
                    curr_api_id = test_set[i][API]
                    smaller_test_set[category][t].add(test_set[i])
                    prog_ids.add(test_set[i][PID])
                else:
                    if test_set[i][LEN] >= min_length:
                        curr_api_id = test_set[i][API]
                        smaller_test_set[category][t].add(test_set[i])
                        prog_ids.add(test_set[i][PID])

    return test_set, smaller_test_set, prog_ids


def create_small_test_set_multiple_per_api(category, t, num_per_api, num_progs_per_category, test_set, smaller_test_set,
                                           prog_ids, min_length):
    curr_api_id = -1
    num_of_api_added = 0
    for i in range(len(test_set)):
        if test_set[i][API] != curr_api_id:
            if test_set[i][PID] in prog_ids:
                if num_of_api_added < num_per_api:
                    smaller_test_set[category][t].add(test_set[i])
                    num_of_api_added += 1
            else:
                if i < len(test_set) - 1 and test_set[i][API] != test_set[i + 1][API]:
                    curr_api_id = test_set[i][API]
                    num_of_api_added = 1
                    smaller_test_set[category][t].add(test_set[i])
                    prog_ids.add(test_set[i][PID])
                else:
                    if test_set[i][LEN] >= min_length:
                        curr_api_id = test_set[i][API]
                        num_of_api_added = 1
                        smaller_test_set[category][t].add(test_set[i])
                        prog_ids.add(test_set[i][PID])

    if len(smaller_test_set[category][t]) < num_progs_per_category:
        random.shuffle(test_set)
        for i in range(len(test_set)):
            if test_set[i][PID] not in prog_ids and test_set[i] not in smaller_test_set[category][t]:
                if test_set[i][LEN] >= min_length:
                    smaller_test_set[category][t].add(test_set[i])
                    prog_ids.add(test_set[i][PID])

            if len(smaller_test_set[category][t]) >= num_progs_per_category:
                return test_set, smaller_test_set, prog_ids

        for i in range(len(test_set)):
            if test_set[i] not in smaller_test_set[category][t]:
                if test_set[i][LEN] >= min_length:
                    smaller_test_set[category][t].add(test_set[i])
                    prog_ids.add(test_set[i][PID])

            if len(smaller_test_set[category][t]) >= num_progs_per_category:
                return test_set, smaller_test_set, prog_ids

        # for i in range(len(test_set)):
        #     if test_set[i] not in smaller_test_set[category][t]:
        #         smaller_test_set[category][t].add(test_set[i])
        #         prog_ids.add(test_set[i][PID])
        #
        #     if len(smaller_test_set[category][t]) >= num_progs_per_category:
        #         return test_set, smaller_test_set, prog_ids

    return test_set, smaller_test_set, prog_ids


def create_smaller_test_set(data_dir_path, data_dir_name, train_test_set_name, num_progs_per_category=1200, save=True, min_length=3):
    print("\n\n\nBuilding Smaller Test Sets\n")
    creator_dir_path = data_dir_path + "/" + train_test_set_name + "/"
    print(creator_dir_path)
    f = open(creator_dir_path + "/dataset_creator.pickle", "rb")
    dataset_creator = pickle.load(f)

    print(dataset_creator.categories[EX_CS][0])

    smaller_test_set = {}
    small_test_set_progs = set([])
    for category in dataset_creator.categories:
        cat_test_set = dataset_creator.categories[category][0]
        smaller_test_set[category] = {}
        for t in cat_test_set.keys():
            if len(cat_test_set[t]) == 0:
                continue
            smaller_test_set[category][t] = set([])
            prog_ids = set([])
            test_set = list(cat_test_set[t].copy())
            random.shuffle(test_set)
            test_set = sorted(test_set, key=lambda x: x[1])
            num_unique_apis = len(set([x[1] for x in test_set]))

            if num_unique_apis < num_progs_per_category:
                num_progs_per_t = min(len(test_set), num_progs_per_category)
                num_per_api = int(math.ceil(num_progs_per_t/num_unique_apis))

                test_set, smaller_test_set, prog_ids = create_small_test_set_multiple_per_api(category, t, num_per_api,
                                                                                              num_progs_per_category,
                                                                                              test_set,
                                                                                              smaller_test_set,
                                                                                              prog_ids, min_length)
            else:
                test_set, smaller_test_set, prog_ids = create_small_test_set_one_per_api(category, t, test_set,
                                                                                         smaller_test_set, prog_ids, min_length)

            print("\n\nCategory:", category, t)
            print("Prog ids added:", len(set(prog_ids)))
            print("Pairs added:", len(smaller_test_set[category][t]))
            small_test_set_progs.update(prog_ids)

    print("Num progs in smaller test set:", len(small_test_set_progs))
    print("Keys in small test set:", smaller_test_set.keys())

    if save:
        test_dir_path = "/test/small_min_length_" + str(min_length) + "/"
        if not os.path.exists(creator_dir_path + test_dir_path):
            os.mkdir(creator_dir_path + "/test/small_min_length_" + str(min_length) + "/")

        test_f = open(creator_dir_path + test_dir_path + "small_test_set.json", "w+")

        # start data files
        test_f.write("{\n")
        test_f.write("\"programs\": [\n")

        data_filename = data_dir_name + ".json"
        data_f = open(os.path.join(data_dir_path, data_filename), "r")
        dataset_creator.ga.json_asts = ijson.items(data_f, 'programs.item')

        test_set_new_prog_ids = {}

        prog_id = 0
        test_prog_counter = 0
        for program in dataset_creator.ga.json_asts:
            if prog_id in small_test_set_progs:
                if test_prog_counter != 0:
                    test_f.write(",\n")
                test_f.write(json.dumps(program))

                # update
                test_set_new_prog_ids[prog_id] = test_prog_counter
                test_prog_counter += 1

            prog_id += 1

        # end new json data file
        test_f.write("\n")
        test_f.write("]\n")
        test_f.write("}\n")
        test_f.close()

        print("Added", test_prog_counter, "programs to test set")

        with open(creator_dir_path + test_dir_path + "small_test_set_new_prog_ids.pickle", 'wb') as f:
            pickle.dump(test_set_new_prog_ids, f)
            f.close()

        with open(creator_dir_path + test_dir_path + "small_curated_test_sets.pickle", 'wb') as f:
            pickle.dump(smaller_test_set, f)
            f.close()


def build_bayou_datasets(mcmc_data_dir_path, bayou_data_dir_path, bayou_data_folder_name):
    # f = open(creator_path, "rb")
    # dataset_creator = pickle.load(f)

    train_path = bayou_data_dir_path + "/train"
    test_path = bayou_data_dir_path + "/test"
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    # train_f = open(bayou_data_dir_path + "/train/training_data.json", "w+")
    # test_f = open(bayou_data_dir_path + "/test/small_test_set.json", "w+")
    small_test_f = open(bayou_data_dir_path + "/test/small_test_set.json", "w+")

    # start data files
    # train_f.write("{\n")
    # train_f.write("\"programs\": [\n")
    # test_f.write("{\n")
    # test_f.write("\"programs\": [\n")
    small_test_f.write("{\n")
    small_test_f.write("\"programs\": [\n")

    data_f = open(bayou_data_dir_path + bayou_data_folder_name + ".json", "rb")

    # mcmc_test_f = open(mcmc_data_dir_path + "train_test_sets/test/test_set.json", "rb")
    mcmc_small_test_f = open(mcmc_data_dir_path + "train_test_sets/test/small_test_set.json", "rb")

    # test_prog_set = set([])
    # test_progs = ijson.items(mcmc_test_f, 'programs.item')
    # for program in test_progs:
    #     key = (json.dumps(program['ast']), json.dumps(program['returnType']), json.dumps(program['formalParam']))
    #     test_prog_set.add(key)
    #
    # print("Number of programs in test set:", len(test_prog_set))

    small_test_prog_set = set([])
    test_progs = ijson.items(mcmc_small_test_f, 'programs.item')
    for program in test_progs:
        key = (json.dumps(program['ast']), json.dumps(program['returnType']), json.dumps(program['formalParam']))
        small_test_prog_set.add(key)

    print("Number of programs in small test set:", len(small_test_prog_set))

    test_set_new_prog_ids = {}
    small_test_set_new_prog_ids = {}
    prog_id = 0
    test_prog_counter = 0
    train_prog_counter = 0
    small_test_prog_counter = 0
    for program in ijson.items(data_f, 'programs.item'):
        key = (json.dumps(program['ast']), json.dumps(program['returnType']), json.dumps(program['formalParam']))

        if key in small_test_prog_set:
            if small_test_prog_counter != 0:
                small_test_f.write(",\n")
            small_test_set_new_prog_ids[prog_id] = small_test_prog_counter
            small_test_f.write(json.dumps(program))
            small_test_prog_counter += 1

        # if key in test_prog_set:
        #     if test_prog_counter != 0:
        #         test_f.write(",\n")
        #     test_set_new_prog_ids[prog_id] = test_prog_counter
        #     test_f.write(json.dumps(program))
        #     test_prog_counter += 1
        # else:
        #     if train_prog_counter != 0:
        #         train_f.write(",\n")
        #     train_f.write(json.dumps(program))
        #     train_prog_counter += 1
        prog_id += 1

    # end new json data file
    # train_f.write("\n")
    # train_f.write("]\n")
    # train_f.write("}\n")
    # test_f.write("\n")
    # test_f.write("]\n")
    # test_f.write("}\n")
    small_test_f.write("\n")
    small_test_f.write("]\n")
    small_test_f.write("}\n")
    # test_f.close()
    # train_f.close()
    small_test_f.close()

    # print("Added", test_prog_counter, "programs to test set")
    # print("Added", train_prog_counter, "programs to training set")
    print("Added", small_test_prog_counter, "programs to small test set")
    print("Total programs in old dataset:", prog_id - 1)

    # with open(bayou_data_dir_path + "/test/test_set_new_prog_ids.pickle", 'wb') as f:
    #     pickle.dump(test_set_new_prog_ids, f)
    #     f.close()

    with open(bayou_data_dir_path + "/test/small_test_set_new_prog_ids.pickle", 'wb') as f:
        pickle.dump(small_test_set_new_prog_ids, f)
        f.close()


def add_prog_length_to_dataset_creator(data_dir_path, train_test_set_name, save=False):
    creator_dir_path = data_dir_path + "/" + train_test_set_name + "/"
    f = open(creator_dir_path + "/dataset_creator.pickle", "rb")
    dataset_creator = pickle.load(f)

    for category in dataset_creator.categories:
        cat_test_set = dataset_creator.categories[category][0]
        for t in cat_test_set.keys():
            new_test_set = set([])
            for data in cat_test_set[t]:
                prog_id = data[PID]
                api = data[API]
                dp2 = data[DP2]
                length = np.argmax(dataset_creator.ga.reader.nodes[prog_id] == 0)
                new_test_set.add((prog_id, api, dp2, length))
                # print("old:", data)
                # print(dataset_creator.ga.reader.nodes[prog_id])
                # print("new:", (prog_id, api, dp2, length), "\n")

            dataset_creator.categories[category][0][t] = new_test_set

    if save:
        dataset_creator.pickle_dump_self(path=creator_dir_path)
        dataset_creator.pickle_dump_curated_test_sets(path=creator_dir_path)














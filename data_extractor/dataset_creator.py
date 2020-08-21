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


class DatasetCreator:
    """

    graph analyzer + database on:
    - 1k vocab dataset
    - entire dataset
    """
    def __init__(self, data_dir_path, save_reader=False, min_prog_per_category=1200, verbose=False, test_mode=True):
        self.data_dir_path = data_dir_path

        if save_reader:
            self.ga = GraphAnalyzer(data_dir_path, save_reader=True, shuffle_data=False,
                                    load_g_without_control_structs=False, pickle_friendly=True)
        else:
            self.ga = GraphAnalyzer(data_dir_path, load_reader=True, load_g_without_control_structs=False,
                                    pickle_friendly=True)

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

        self.dir_path = self.ga.dir_path + "/train_test_sets3/"


    # def create_novelty_test_set(self):

    def add_include_or_exclude_test_progs(self, added_to_test_set_func, category, novelty_label):
        assert category in DP2_API or category in DP2_CS, "Error: category must be " + IN_CS + ", " + IN_API + ", " \
                                                          + EX_API + ", " + ", or " + EX_CS
        api_idx_range = list(range(self.full_range[0], self.full_range[1]))
        random.shuffle(api_idx_range)

        if category in DP2_API:
            dp2_idx_range = list(range(self.full_range[0], self.full_range[1]))
            random.shuffle(dp2_idx_range)
            data_points = self.ranks
        else:
            dp2_idx_range = range(len(self.control_structs))
            data_points = self.control_structs

        test_set = self.categories[category][0][novelty_label]

        num_pairs_added = 0

        all_possible_data_points = itertools.product(api_idx_range, dp2_idx_range)

        def suitable_candidate(idxs):
            progs_with_both = len(
                self.ga.get_program_ids_with_multiple_apis([self.ranks[idxs[0]], data_points[idxs[1]]]))
            progs_with_api = len(self.ga.get_program_ids_for_api(self.ranks[idxs[0]]))
            progs_with_dp2 = len(self.ga.get_program_ids_for_api(data_points[idxs[1]]))
            return 1 < progs_with_both < progs_with_api and progs_with_dp2 > progs_with_both

        start_time = time.time()
        valid_data_points = filter(lambda idxs: suitable_candidate(idxs), all_possible_data_points)

        if self.test_mode:
            print("Time taken for valid_data_points:", start_time - time.time())

        added_apis = set([])

        counter = 0
        start_time = time.time()
        for api_idx, dp_idx in valid_data_points:
            counter += 1
            if counter % 5000 == 0:
                print("counter:", counter)
                print("num pairs added:", num_pairs_added)
                if self.test_mode:
                    print("time taken:", start_time - time.time())
                    start_time = time.time()

            api = self.ranks[api_idx]
            data_point2 = data_points[dp_idx]
            # print(api)
            # print(data_point2)
            if api != data_point2 and api not in added_apis and not (category in DP2_API and data_point2 in added_apis):
                progs_with_api = self.ga.get_program_ids_for_api(api)
                progs_with_dp2 = self.ga.get_program_ids_for_api(data_point2)

                if not self.ga.g.has_edge(api, data_point2):
                    raise ValueError("shouldn't be here!!!")

                progs_with_both = self.ga.get_program_ids_with_multiple_apis([api, data_point2])

                if added_to_test_set_func(api, data_point2, novelty_label, test_set, progs_with_api, progs_with_dp2,
                                          progs_with_both):
                    num_pairs_added += 1
                    added_apis.add(api)
                    if category in DP2_API:
                        added_apis.add(data_point2)
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
                self.add_to_test_set(api, data_point2, prog_ids, test_set)
                return True

            if novelty_label == SEEN:
                limit = min(math.ceil(num_progs_with_both / 4), self.control_limit * num_progs_with_api, 10)
                prog_ids = set(itertools.islice(progs_with_both, limit))

                if len(prog_ids) == 0:
                    return False

                if self.verbose:
                    print("api:", api, "data_point:", data_point2)
                    print("num progs added:", len(prog_ids))
                self.add_to_test_set(api, data_point2, prog_ids, test_set)
                return True

        return False

    def added_to_exclude_test_set(self, api, data_point2, novelty_label, test_set, progs_with_api, progs_with_dp2,
                                  progs_with_both):
        num_progs_with_both = len(progs_with_both)
        num_progs_with_api = len(progs_with_api)
        num_progs_with_dp2 = len(progs_with_dp2)

        if data_point2 in {DBRANCH, DLOOP, DEXCEPT}:
            if novelty_label == SEEN:
                max_progs = 100
            else:
                max_progs = 50
        else:
            max_progs = 200

        if self.verbose and self.test_mode:
            print("num progs api", num_progs_with_api)
            print("num progs dp2", num_progs_with_dp2)
            print("num progs both", num_progs_with_both)
            print("control limit:", num_progs_with_api * self.control_limit)
            print("difference:", num_progs_with_api - num_progs_with_both)
        if num_progs_with_api - num_progs_with_both <= num_progs_with_api * self.control_limit:
            if novelty_label == NEW and 0 < num_progs_with_api - num_progs_with_both <= max_progs:
                prog_ids = progs_with_api - progs_with_both
                if self.verbose:
                    print("api:", api, "data_point:", data_point2)
                    print("num progs added:", len(prog_ids))

                if len(prog_ids) == 0:
                    return False

                self.add_to_test_set(api, data_point2, prog_ids, test_set)
                return True

            if novelty_label == SEEN:
                limit = min(math.ceil((num_progs_with_api - num_progs_with_both) / 4),
                            self.control_limit * num_progs_with_api, 10)
                prog_ids = progs_with_api - progs_with_both

                if len(prog_ids) == 0:
                    return False

                prog_ids = itertools.islice(prog_ids, limit)

                if self.verbose:
                    print("api:", api, "data_point:", data_point2)
                self.add_to_test_set(api, data_point2, prog_ids, test_set)
                return True

        return False

    def add_to_test_set(self, data_point1, data_point2, prog_ids_set, test_set, dp2type_is_int=False):
        api_num = self.ga.vocab2node[data_point1]

        if dp2type_is_int:
            dp_num = data_point2
        else:
            dp_num = self.ga.vocab2node[data_point2]
        for prog_id in prog_ids_set:
            test_set.add((prog_id, api_num, dp_num))
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

        api_idx_range = list(range(self.full_range[0], self.full_range[1]))
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

        added_apis = set([])

        counter = 0
        start_time = time.time()
        for api_idx, dp_idx in valid_data_points:
            counter += 1
            if counter % 5000 == 0:
                print("num pairs added:", num_pairs_added)
                if self.test_mode:
                    print("time taken:", start_time - time.time())
                    start_time = time.time()

            api = self.ranks[api_idx]
            length = dp_idx

            if api not in added_apis:

                valid_prog_ids = self.ga.get_program_ids_for_api_length_k(api, category, length)
                num_valid_progs = len(valid_prog_ids)
                progs_with_api = self.ga.get_program_ids_for_api(api)
                num_progs_with_api = len(progs_with_api)

                if self.verbose and self.test_mode:
                    print("num progs that meet length criteria:", num_valid_progs)
                    print("num progs with api:", num_progs_with_api)
                    print("control api:", self.control_limit * num_progs_with_api)

                if num_valid_progs <= self.control_limit * num_progs_with_api:
                    if novelty_label == NEW and 0 < num_valid_progs <= 50:
                        if self.verbose:
                            print("api:", api, "length:", length)
                            print("num progs added:", len(valid_prog_ids))
                        self.add_to_test_set(api, length, valid_prog_ids, test_set, dp2type_is_int=True)
                        num_pairs_added += 1
                        added_apis.add(api)

                    if novelty_label == SEEN:
                        limit = min(math.ceil(num_valid_progs / 4), self.control_limit * num_progs_with_api, 10)
                        prog_ids = set(itertools.islice(valid_prog_ids, limit))

                        if len(prog_ids) == 0:
                            continue

                        if self.verbose:
                            print("api:", api, "length:", length)
                            print("num progs added:", len(prog_ids))

                        self.add_to_test_set(api, length, prog_ids, test_set, dp2type_is_int=True)
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
        self.add_random_programs()
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
                self.add_to_test_set(selected_vocab_idx[i], -1, prog_id, self.random_progs_set[freq],
                                     dp2type_is_int=True)

        print("Number of random programs added:", len(self.random_progs_set))

    def create_curated_dataset(self):
        print("Creating Curated Tests Dataset\n")

        for novelty_label in [NEW, SEEN]:  # Create novelty test set first
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
            print("EXCLUDE API: ")
            start_time = time.time()
            self.add_include_or_exclude_test_progs(self.added_to_exclude_test_set, EX_API, novelty_label)
            print("test set len:", len(self.categories[EX_API][0][novelty_label]), "\n")
            if self.test_mode:
                print("Time taken for exclude api:", start_time - time.time())

            print("\n\n\n-----------------------------------")
            print("EXCLUDE CS: ")
            start_time = time.time()
            self.add_include_or_exclude_test_progs(self.added_to_exclude_test_set, EX_CS, novelty_label)
            print("test set len:", len(self.categories[EX_CS][0][novelty_label]), "\n")
            if self.test_mode:
                print("Time taken for exclude cs:", start_time - time.time())

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

    def pickle_dump_curated_test_sets(self):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        with open(self.dir_path + "/curated_test_sets.pickle", 'wb') as f:
            pickle.dump(self.categories, f)
            f.close()

    def pickle_dump_self(self):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        self.ga.json_asts = None

        with open(self.dir_path + "/dataset_creator.pickle", 'wb') as f:
            pickle.dump(self, f)
            f.close()

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


def build_sets_from_saved_creator(creator_path):
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

    if not os.path.exists(dataset_creator.dir_path + "/train"):
        os.mkdir(dataset_creator.dir_path + "/train")

    if not os.path.exists(dataset_creator.dir_path + "/test"):
        os.mkdir(dataset_creator.dir_path + "/test")

    train_f = open(dataset_creator.dir_path + "/train/training_data.json", "w+")
    test_f = open(dataset_creator.dir_path + "/test/test_set.json", "w+")

    # start data files
    train_f.write("{\n")
    train_f.write("\"programs\": [\n")
    test_f.write("{\n")
    test_f.write("\"programs\": [\n")

    data_filename = dataset_creator.ga.clargs.data_filename + ".json"
    data_f = open(os.path.join(dataset_creator.ga.dir_path, data_filename))
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

    with open(dataset_creator.dir_path + "/test/test_set_new_prog_ids.pickle", 'wb') as f:
        pickle.dump(test_set_new_prog_ids, f)
        f.close()


def create_smaller_test_set(creator_path, save=True):
    print("\n\n\nBuilding Smaller Test Sets\n")
    f = open(creator_path, "rb")
    dataset_creator = pickle.load(f)


    PID = 0
    API = 1
    DP2 = 2

    smaller_test_set = {}
    small_test_set_progs = set([])
    for category in dataset_creator.categories:
        cat_test_set = dataset_creator.categories[category][0]
        smaller_test_set[category] = {}
        for t in cat_test_set.keys():
            smaller_test_set[t] = set([])
            prog_ids = set([])
            test_set = sorted(list(cat_test_set[t]), key=lambda x: x[1])
            curr_api_id = -1
            for i in range(len(test_set)):
                if test_set[i][API] != curr_api_id:
                    if test_set[i][PID] in prog_ids:
                        if i < len(test_set) - 1 and test_set[i][API] != test_set[i+1][API]:
                            smaller_test_set[t].add(test_set[i])
                    else:
                        curr_api_id = test_set[i][API]
                        smaller_test_set[t].add(test_set[i])
                        prog_ids.add(test_set[i][PID])
            print("\n\nCategory:", category, t)
            print("Prog ids added:", len(prog_ids))
            print("Pairs added:", len(smaller_test_set[t]))
            small_test_set_progs.update(prog_ids)

    print("Num progs in smaller test set:", len(small_test_set_progs))

    if save:
        test_f = open(dataset_creator.dir_path + "/test/small_test_set.json", "w+")

        # start data files
        test_f.write("{\n")
        test_f.write("\"programs\": [\n")

        data_filename = dataset_creator.ga.clargs.data_filename + ".json"
        data_f = open(os.path.join(dataset_creator.ga.dir_path, data_filename))
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

        with open(dataset_creator.dir_path + "/test/small_test_set_new_prog_ids.pickle", 'wb') as f:
            pickle.dump(test_set_new_prog_ids, f)
            f.close()

        with open(dataset_creator.dir_path + "/test/small_curated_test_sets.pickle", 'wb') as f:
            pickle.dump(smaller_test_set, f)
            f.close()


# def rebuild_curated_set(creator_path):
#     print("\n\n\nBuilding Smaller Test Sets\n")
#     f = open(creator_path, "rb")
#     dataset_creator = pickle.load(f)














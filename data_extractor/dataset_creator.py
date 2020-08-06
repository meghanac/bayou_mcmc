import itertools
import time

from data_extractor.graph_analyzer import GraphAnalyzer, ALL_DATA_NO_DUP
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
MIN = 'min'
MAX = 'max'
HIGH = 'high'
MID = 'mid'
LOW = 'low'
FREQ = 'frequency'
WEIGHT = 'weight'
DP2_API = [IN_API, EX_API]
DP2_CS = [IN_CS, EX_CS]

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
                                    load_g_without_control_structs=False)
        else:
            self.ga = GraphAnalyzer(data_dir_path, load_reader=True, load_g_without_control_structs=False)

        # self.training_data = set(range(self.ga.num_programs))
        # self.test_data = set([])
        # self.novelty_test_set = set([])
        # self.accuracy_test_set = set([])

        self.ranks = sorted(nx.get_node_attributes(self.ga.g, 'frequency').items(), key=lambda x: x[1], reverse=True)
        self.ranks = [(self.ranks[i][0], (i, self.ranks[i][1])) for i in range(len(self.ranks))]
        self.ranks_dict = dict(self.ranks)
        self.ranks = [i[0] for i in self.ranks]
        self.ranks.remove(DBRANCH)
        self.ranks.remove(DLOOP)
        self.ranks.remove(DEXCEPT)
        self.num_apis = len(self.ranks)

        self.include_api_set = {SEEN: {}, NEW: {}}
        self.include_cs_set = {SEEN: {}, NEW: {}}
        self.exclude_api_set = {SEEN: {}, NEW: {}}
        self.exclude_cs_set = {SEEN: {}, NEW: {}}
        self.min_api_set = {SEEN: {}, NEW: {}}
        self.max_api_set = {SEEN: {}, NEW: {}}

        self.categories = {IN_API: (self.include_api_set, API), EX_API: (self.exclude_api_set, API),
                           IN_CS: (self.include_cs_set, CS), EX_CS: (self.exclude_cs_set, CS),
                           MAX: (self.max_api_set, LEN), MIN: (self.min_api_set, LEN)}

        self.high_range = (0, math.floor(self.num_apis / 3))
        self.mid_range = (math.floor(self.num_apis / 3), math.floor(self.num_apis * 2/3))
        self.low_range = (math.floor(self.num_apis * 2/3), self.num_apis)
        self.freq_pairs = [(LOW, LOW), (MID, LOW), (HIGH, LOW), (HIGH, MID), (HIGH, HIGH)]
        self.idx_ranges = {HIGH: self.high_range, MID: self.mid_range, LOW: self.low_range}

        for category in self.categories:
            test_set = self.categories[category][0]
            for freq_pair in self.freq_pairs:
                test_set[NEW][freq_pair] = set([])
                test_set[SEEN][freq_pair] = set([])

        if verbose:
            print(self.high_range)
            print(self.mid_range)
            print(self.low_range)

        self.min_prog_per_category = min_prog_per_category
        self.min_prog_per_freq = math.ceil(min_prog_per_category / 3)
        self.min_pairs_per_freq = math.ceil(self.min_prog_per_freq / 2)

        self.control_structs = [DBRANCH, DEXCEPT, DLOOP]

        self.test_api_to_prog_ids = {}

        self.control_limit = 0.5

        self.verbose = verbose
        self.test_mode = test_mode

    # def create_novelty_test_set(self):

    def add_include_or_exclude_test_progs(self, added_to_test_set_func, category, freq_pair, novelty_label):
        assert category in DP2_API or category in DP2_CS, "Error: category must be " + IN_CS + ", " + IN_API + ", " \
                                                          + EX_API + ", " + ", or " + EX_CS

        api_idx_range = range(self.idx_ranges[freq_pair[0]][0], self.idx_ranges[freq_pair[0]][1])

        if category in DP2_API:
            dp2_idx_range = range(self.idx_ranges[freq_pair[1]][0], self.idx_ranges[freq_pair[1]][1])
            data_points = self.ranks
        else:
            dp2_idx_range = range(len(self.control_structs))
            data_points = self.control_structs

        test_set = self.categories[category][0][novelty_label][freq_pair]

        num_pairs_added = 0

        all_possible_data_points = itertools.product(api_idx_range, dp2_idx_range)

        start_time = time.time()
        valid_data_points = filter(lambda idxs: self.ga.g.has_edge(self.ranks[idxs[0]], data_points[idxs[1]]),
                                   all_possible_data_points)
        if self.test_mode:
            print("Time taken for valid_data_points:", start_time - time.time())

        added_apis = set([])

        # print(len(list(valid_data_points)))
        if self.verbose:
            print(freq_pair)

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
            data_point2 = data_points[dp_idx]
            # print(api)
            # print(data_point2)
            if api != data_point2 and api not in added_apis and not (category in DP2_API and data_point2 in added_apis):
                progs_with_api = self.ga.get_program_ids_for_api(api)
                progs_with_dp2 = self.ga.get_program_ids_for_api(data_point2)

                if not self.ga.g.has_edge(api, data_point2):
                    raise ValueError("shouldn't be here!!!")

                progs_with_both = self.ga.get_program_ids_with_multiple_apis([api, data_point2])

                if added_to_test_set_func(api, data_point2, novelty_label, test_set, progs_with_api, progs_with_dp2, progs_with_both):
                    num_pairs_added += 1
                    added_apis.add(api)
                    if category in DP2_API:
                        added_apis.add(data_point2)
                    if num_pairs_added >= self.min_pairs_per_freq and len(test_set) >= self.min_prog_per_freq:
                        print("Category:", category)
                        print("Novelty label: " + novelty_label)
                        print("Frequency Pair: " + str(freq_pair))
                        print("Num API + " + category + " pairs added: " + str(num_pairs_added))
                        print("Total programs added: " + str(len(test_set)))
                        return

    def added_to_include_test_set(self, api, data_point2, novelty_label, test_set, progs_with_api, progs_with_dp2,
                                  progs_with_both):
        # print("num progs with both:", num_progs_with_both)
        # print("num progs with api:", num_progs_with_api)
        # print("num progs with dp2:", num_progs_with_dp2)
        # print("control api:", self.control_limit * num_progs_with_api)
        # print("control dp2:", self.control_limit * num_progs_with_dp2)
        num_progs_with_both = len(progs_with_both)
        num_progs_with_api = len(progs_with_both)
        num_progs_with_dp2 = len(progs_with_dp2)

        # if self.verbose and self.test_mode:
        #     print("num api:", num_progs_with_api)
        #     print("num dp2:", num_progs_with_dp2)
        #     print("num both:", num_progs_with_both)

        if num_progs_with_both <= self.control_limit * num_progs_with_api and \
                num_progs_with_both <= self.control_limit * num_progs_with_dp2:
            if novelty_label == NEW and 0 < num_progs_with_both <= 1000:
                prog_ids = self.ga.get_program_ids_with_multiple_apis([api, data_point2])

                if len(prog_ids) == 0:
                    return False

                if self.verbose:
                    print("api:", api, "data_point:", data_point2)
                    print("num progs added:", len(prog_ids))
                self.add_to_test_set(api, data_point2, prog_ids, test_set)
                return True

            if novelty_label == SEEN:
                limit = min(math.floor(num_progs_with_both / 4), 10)
                prog_ids = self.ga.get_program_ids_with_multiple_apis([api, data_point2], limit=limit)

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
        num_progs_with_api = len(progs_with_both)
        num_progs_with_dp2 = len(progs_with_dp2)
        if num_progs_with_api - num_progs_with_both <= num_progs_with_api * self.control_limit:
            if novelty_label == NEW and 0 < num_progs_with_api - num_progs_with_both <= 500:
                both_apis_prog_ids = self.ga.get_program_ids_with_multiple_apis([api, data_point2])
                api_prog_ids = self.ga.get_program_ids_for_api(api)
                prog_ids = api_prog_ids - both_apis_prog_ids
                if self.verbose:
                    print("api:", api, "data_point:", data_point2)
                    print("num progs added:", len(prog_ids))

                if len(prog_ids) == 0:
                    return False

                self.add_to_test_set(api, data_point2, prog_ids, test_set)
                return True

            if novelty_label == SEEN:
                limit = min(math.floor((num_progs_with_api - num_progs_with_both) / 4), 10)
                both_apis_prog_ids = self.ga.get_program_ids_with_multiple_apis([api, data_point2])
                api_prog_ids = self.ga.get_program_ids_for_api(api)
                prog_ids = api_prog_ids - both_apis_prog_ids

                if len(prog_ids) == 0:
                    return False

                prog_ids = itertools.islice(prog_ids, limit)

                if self.verbose:
                    print("api:", api, "data_point:", data_point2)
                self.add_to_test_set(api, data_point2, prog_ids, test_set)
                return True

        return False

    def add_to_test_set(self, data_point1, data_point2, prog_ids_set, test_set):
        api_num = self.ga.vocab2node[data_point1]
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

    def add_exclude_test_progs(self, dp2_type, freq_pair, novel):
        pass

    def create_accuracy_test_set(self):
        pass

    def add_length_constrained_test_progs(self, category, freq_pair, novel):
        pass

    def add_random_programs(self):
        pass

    def create_dataset(self):
        print("Creating Custom Test Dataset\n")
        for novelty_label in [NEW, SEEN]:  # Create novelty test set first
            for freq_pair in self.freq_pairs:
                print("\n\n\n-----------------------------------")
                print("EXCLUDE CS: ", freq_pair)
                start_time = time.time()
                self.add_include_or_exclude_test_progs(self.added_to_exclude_test_set, EX_CS,
                                                       freq_pair, novelty_label)
                print("test set len:", len(self.categories[EX_CS][0][novelty_label]), "\n")
                if self.test_mode:
                    print("Time taken for exclude cs", freq_pair, ":", start_time - time.time())

                print("\n\n\n-----------------------------------")
                print("INCLUDE API: ", freq_pair)
                start_time = time.time()
                self.add_include_or_exclude_test_progs(self.added_to_include_test_set, IN_API,
                                                       freq_pair, novelty_label)
                print("test set len:", len(self.categories[IN_API][0][novelty_label]), "\n")
                if self.test_mode:
                    print("Time taken for include api", freq_pair, ":", start_time - time.time())

                # self.add_include_or_exclude_test_progs(self.added_to_include_test_set, IN_CS, freq_pair, novel)
                # self.add_include_or_exclude_test_progs(self.added_to_exclude_test_set, EX_API, freq_pair, novel)

                # print("\n\n\n-----------------------------------")
                # print("MIN LENGTH")
                # self.add_length_constrained_test_progs(MIN, freq_pair, novelty_label)
                # print("test set len:", len(self.categories[MIN][0][novelty_label]), "\n")
                #
                # print("\n\n\n-----------------------------------")
                # print("MAX LENGTH")
                # self.add_length_constrained_test_progs(MAX, freq_pair, novelty_label)
                # print("test set len:", len(self.categories[MAX][0][novelty_label]), "\n")

        self.add_random_programs()

    def get_freq_label(self, api):
        rank = self.ranks.index(api)
        if self.high_range[0] <= rank < self.high_range[1]:
            return HIGH
        elif self.mid_range[0] <= rank < self.mid_range[1]:
            return MID
        else:
            return LOW






# if num_progs_with_both <= self.control_limit * num_progs_with_api and \
#         num_progs_with_both <= self.control_limit * num_progs_with_dp2:
#     if novel and num_progs_with_both < 200:
#         prog_ids = self.ga.get_program_ids_with_multiple_apis([api, data_point2])
#         self.add_to_test_set(api, data_point2, prog_ids, test_set)
#         num_pairs_added += 1
#         if num_pairs_added >= self.min_pairs_per_freq and len(test_set) >= self.min_prog_per_freq:
#             print("Category: Include " + dp2_type)
#             print("Novel: " + novel)
#             print("Frequency Pair: " + str(freq_pair))
#             print("Num API + " + dp2_type + " pairs added: " + str(num_pairs_added))
#             print("Total programs added: " + str(len(test_set)))
#             return
#         break
#
#     if not novel:
#         limit = min(math.floor(num_progs_with_both / 4), 10)
#         prog_ids = self.ga.get_program_ids_with_multiple_apis([api, data_point2], limit=limit)
#         self.add_to_test_set(api, data_point2, prog_ids, test_set)
#         num_pairs_added += 1
#         if num_pairs_added >= self.min_pairs_per_freq and len(test_set) >= self.min_prog_per_freq:
#             print("Category: Include " + dp2_type)
#             print("Novel: " + novel)
#             print("Frequency Pair: " + str(freq_pair))
#             print("Num API + " + dp2_type + " pairs added: " + str(num_pairs_added))
#             print("Total programs added: " + str(len(test_set)))
#             return
#         break





from graph_analyzer import GraphAnalyzer
import networkx as nx


API = 'api'
CS = 'control_structure'
LEN = 'length'
IN_API = 'include_api'
EX_API = 'exclude_api'
IN_CS = 'include_cs'
EX_CS = 'exclude_cs'
MIN = 'min'
MAX = 'max'
HIGH = 'high'
MID = 'mid'
LOW = 'low'


class DatasetCreator:
    """

    graph analyzer + database on:
    - 1k vocab dataset
    - entire dataset









    """
    def __init__(self, data_dir_path, save_reader=False):
        self.data_dir_path = data_dir_path

        if save_reader:
            self.ga = GraphAnalyzer(data_dir_path, save_reader=True, shuffle_data=False)
        else:
            self.ga = GraphAnalyzer(data_dir_path, load_reeader=True)

        self.training_data = set(range(self.ga.num_programs))
        self.test_data = set([])
        self.novelty_test_set = set([])
        self.accuracy_test_set = set([])

        self.ranks = sorted(nx.get_node_attributes(self.ga.g, 'frequency').items(), key=lambda x: x[1], reverse=True)
        self.num_apis = len(self.ranks)
        self.ranks = [(self.ranks[i][0], (i, self.ranks[i][1])) for i in range(len(self.ranks))]
        self.ranks_dict = dict(self.ranks)
        self.ranks = [i[0] for i in self.ranks]

        self.categories = {IN_API: (API, API), EX_API: (API, API), IN_CS: (API, CS), EX_CS: (API, CS), MAX: (API, LEN),
                         MIN: (API, LEN)}
        self.high_range = (0, self.num_apis / 3)
        self.mid_range = (self.num_apis / 3, self.num_apis * 2/3)
        self.low_range = (self.num_apis * 2/3, self.num_apis)
        self.freq_pairs = {(HIGH, HIGH), (HIGH, MID), (HIGH, LOW), (MID, LOW), (LOW, LOW)}
        self.idx_ranges = {HIGH: self.high_range, MID: self.mid_range, LOW: self.low_range}

        self.min_prog_per_category = 1200

    # def create_novelty_test_set(self):


    def add_include_test_progs(self, dp2_type, freq_pair, novel):
        num_progs_in_test_set = len(self.test_data)

    def add_exclude_test_progs(self, dp2_type, freq_pair, novel):
        pass

    def create_accuracy_test_set(self):
        pass

    def add_length_constrained_test_progs(self, min, freq_pair, novel):
        pass

    def add_random_programs(self):
        pass

    def create_dataset(self):
        for novel in [True, False]:  # Create novelty test set first
            for freq_pair in self.freq_pairs:
                self.add_include_test_progs(API, freq_pair, novel)
                self.add_include_test_progs(CS, freq_pair, novel)
                self.add_exclude_test_progs(API, freq_pair, novel)
                self.add_exclude_test_progs(CS, freq_pair, novel)
                self.add_length_constrained_test_progs(True, freq_pair, novel)
                self.add_length_constrained_test_progs(False, freq_pair, novel)

        self.add_random_programs()

    def get_freq_label(self, api):
        rank = self.ranks.index(api)
        if self.high_range[0] <= rank < self.high_range[1]:
            return HIGH
        elif self.mid_range[0] <= rank < self.mid_range[1]:
            return MID
        else:
            return LOW





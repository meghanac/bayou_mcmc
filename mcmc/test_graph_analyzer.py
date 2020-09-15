import pickle
import random

import ijson
import numpy as np
import unittest
import scipy.stats
import networkx as nx
from data_extractor.json_data_extractor import copy_json_data_change_return_types, copy_bayou_json_data_change_apicalls, \
    copy_json_data_limit_vocab, copy_data_remove_duplicate, create_identical_bayou_dataset, analyze_file
from data_extractor.graph_analyzer import GraphAnalyzer, STR_BUF, STR_APP, READ_LINE, CLOSE, STR_LEN, STR_BUILD, \
    STR_BUILD_APP, LOWERCASE_LOCALE, DATA_DIR_PATH, ALL_DATA_1K_VOCAB, TESTING, NEW_VOCAB, APIS, RT, FP, TOP, MID, \
    LOW, ALL_DATA_1K_VOCAB_NO_DUP, ALL_DATA, ALL_DATA_NO_DUP, MIN_EQ, MAX_EQ
from data_extractor.dataset_creator import DatasetCreator, build_sets_from_saved_creator, create_smaller_test_set, \
    build_bayou_datasets, pickle_dump_test_sets, add_prog_length_to_dataset_creator
from test_suite import MOST_COMMON_APIS, MID_COMMON_APIS, UNCOMMON_APIS, MID_COMMON_DISJOINT_PAIRS, \
    MOST_COMMON_DISJOINT_PAIRS, UNCOMMON_DISJOINT_PAIRS


class TestGraphAnalyzer(unittest.TestCase):

    def testing(self, data_path=ALL_DATA_1K_VOCAB_NO_DUP):
        data_path = 'all_data_2k_vocab_no_duplicates'
        graph_analyzer = GraphAnalyzer(data_path, save_reader=True)
        print("Num progs:", graph_analyzer.num_programs)
        graph_analyzer.get_connected_nodes(LOWERCASE_LOCALE)
        # prog_id =
        print("\n\n")
        graph_analyzer.print_lists(graph_analyzer.get_programs_for_api('java.lang.Long.getLong(java.lang.String,long)'))
        # print(len(self.get_programs_for_api('java.lang.Long.getLong(java.lang.String,long)')))
        print("\n\n")
        # print(len(self.get_programs_with_multiple_apis(
        #     ['java.lang.Long.getLong(java.lang.String,long)', 'java.lang.System.currentTimeMillis()'])))
        graph_analyzer.print_lists(graph_analyzer.get_programs_with_multiple_apis(
            ['java.lang.Long.getLong(java.lang.String,long)', 'java.lang.System.currentTimeMillis()']))

    def test_2(self, data_path=ALL_DATA_1K_VOCAB):
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)
        progs = graph_analyzer.get_programs_for_api('DBranch', limit=1, get_jsons=False)
        graph_analyzer.print_lists([i for i in progs if i[0][-1] != 'DBranch'])
        graph_analyzer.plot_ast(progs[0][0][1], progs[0][1][1], progs[0][2][1])

    def test_3(self, data_path=ALL_DATA_1K_VOCAB):
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)
        js = {"ast": {"node": "DSubTree", "_nodes": [{"node": "DBranch", "_cond": [
            {"node": "DAPICall", "_throws": [], "_returns": "java.lang.String",
             "_call": "java.lang.System.getenv(java.lang.String)"}], "_else": [], "_then": [
            {"node": "DAPICall", "_throws": [], "_returns": "java.lang.String",
             "_call": "java.lang.System.getenv(java.lang.String)"},
            {"node": "DAPICall", "_throws": [], "_returns": "java.lang.String",
             "_call": "java.lang.System.setProperty(java.lang.String,java.lang.String)"}]}]}}
        graph_analyzer.get_vectors_and_plot(js)

    def test_4(self, data_path=ALL_DATA_1K_VOCAB):
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)
        levels = [MOST_COMMON_APIS, MID_COMMON_APIS, UNCOMMON_APIS]
        for level in levels:
            pairs = {}
            for api in level:
                pairs[api] = {}
                pairs[api][TOP] = graph_analyzer.get_disjoint_api(api, level='top', k=5)
                pairs[api][MID] = graph_analyzer.get_disjoint_api(api, level='mid', k=5)
                pairs[api][LOW] = graph_analyzer.get_disjoint_api(api, level='low', k=5)
            print(pairs)

            # print("\n\n", api)
            # print(self.get_disjoint_api(api, level='top', k=5))
            # print(self.get_disjoint_api(api, level='mid', k=5))
            # print(self.get_disjoint_api(api, level='low', k=5))

    def test_5(self, data_path=ALL_DATA_1K_VOCAB):
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)
        apis = [UNCOMMON_APIS[0], UNCOMMON_DISJOINT_PAIRS[UNCOMMON_APIS[0]][MID][0][0]]
        print(apis)
        rt, fp = graph_analyzer.get_top_k_rt_fp(apis)
        rt = [(graph_analyzer.num2rettype[i[0]], i[1]) for i in rt]
        fp = [(graph_analyzer.num2fp[i[0]], i[1]) for i in fp]
        print(rt)
        print(fp)

    def test_prog_ids(self, data_path=ALL_DATA_1K_VOCAB):
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)
        prog_ids = graph_analyzer.get_program_ids_with_multiple_apis([
            'java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()',
            'java.lang.StringBuilder.append(long)'

                                                                ])
        graph_analyzer.print_summary_stats(prog_ids)
        graph_analyzer.print_programs_from_ids(prog_ids, limit=20)

        prog_ids = graph_analyzer.get_program_ids_for_api('java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()')
        graph_analyzer.print_summary_stats(prog_ids)
        graph_analyzer.print_programs_from_ids(prog_ids, limit=20)

        prog_ids = graph_analyzer.get_program_ids_for_api('java.lang.StringBuilder.append(long)')
        graph_analyzer.print_summary_stats(prog_ids)
        graph_analyzer.print_programs_from_ids(prog_ids, limit=20)

        # prog_ids = graph_analyzer.get_program_ids_with_multiple_apis([
        #     'java.lang.StringBuilder.append(long)', 'java.lang.String.isEmpty()'
        # ])
        # graph_analyzer.print_summary_stats(prog_ids)

    def test_get_cooccurring_apis(self, data_path=ALL_DATA_1K_VOCAB):
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)

        api = 'java.util.Map<java.lang.String,byte[]>.hashCode()'
        apis, rt, fp = graph_analyzer.get_k_cooccurring_apis_rt_fp(api, 'low', k=10)

        print([graph_analyzer.node2vocab[api] for api in apis])

    def test_unique_test_data_api_pairs(self, data_path=ALL_DATA_1K_VOCAB):
        test_ga = GraphAnalyzer(data_path, test=True, load_reader=True)
        train_ga = GraphAnalyzer(data_path, load_reader=True)

        unique_pairs = {}
        checked_pairs = {}
        counter = 0
        for api in test_ga.api_to_prog_ids.keys():
            prog_ids = list(test_ga.api_to_prog_ids[api])
            api_name = test_ga.node2vocab[api]
            checked_pairs[api_name] = set([])
            if api_name in {'DSubTree', '__delim__', 'DStop', 'DBranch', 'DLoop', 'DExcept'}:
                continue
            nodes = []
            for prog in prog_ids:
                nodes.extend(test_ga.nodes[prog].tolist())
                nodes.extend(test_ga.targets[prog].tolist())
                nodes = list(set(nodes))
            cooccurring_apis = [test_ga.node2vocab[api] for api in nodes]
            cooccurring_apis = set(cooccurring_apis)
            cooccurring_apis -= {'DSubTree', '__delim__', 'DStop', 'DBranch', 'DLoop', 'DExcept'}
            cooccurring_apis = list(cooccurring_apis)
            for api2 in cooccurring_apis:
                if api_name != api2 and api2 != 'DSubTree' and api2 not in checked_pairs[api_name]:
                    if api2 in checked_pairs and api_name in checked_pairs[api2]:
                        continue
                    print(api_name)
                    print(api2)
                    checked_pairs[api_name].add(api2)
                    cooccurring_prog_ids = train_ga.get_programs_with_multiple_apis([api_name, api2], limit=1)
                    if len(cooccurring_prog_ids) == 0:
                        counter += 1
                        print(counter)
                        if api_name in unique_pairs:
                            unique_pairs[api_name].add(api2)
                        else:
                            unique_pairs[api_name] = {api2}


        unique_pairs_set = set([])
        for api in unique_pairs.keys():
            apis = list(unique_pairs[api])
            for api2 in apis:
                if (api2, api) not in unique_pairs_set:
                    unique_pairs_set.add((api, api2))

        print(len(unique_pairs_set))
        print(unique_pairs_set)

        return unique_pairs_set

    def test_get_progs_for_unique_pairs(self, data_path=ALL_DATA_1K_VOCAB):
        unique_pairs = self.test_unique_test_data_api_pairs(data_path=data_path)
        unique_pairs = list(unique_pairs)
        test_ga = GraphAnalyzer(data_path, test=True, load_reader=True)
        for pair in unique_pairs:
            print('\n', pair)
            prog_ids = test_ga.get_program_ids_with_multiple_apis(list(pair))
            print(prog_ids)
            test_ga.print_summary_stats(prog_ids)
            programs = test_ga.get_programs_with_multiple_apis(list(pair), get_targets=True)
            test_ga.print_lists(programs)

    def test_nodes_edges(self, data_path=ALL_DATA_1K_VOCAB):
        ga = GraphAnalyzer(data_path, load_reader=True)
        print(nx.get_node_attributes(ga.g, 'frequency'))
        for node in ga.g.nodes.keys():
            print(node)
            prog_ids = ga.get_program_ids_for_api(node)
            all_nodes = [ga.fetch_data_with_targets(prog_id)[0] for prog_id in prog_ids]
            ga.print_programs_from_ids(prog_ids, limit=20)
            all_nodes = list(filter(lambda x: len(x) > 0, all_nodes))
            print("1 node progs:", len(all_nodes))
            print(ga.g.nodes[node]['frequency'])
            neighbors = list(ga.g.neighbors(node))
            print(len(neighbors))
            print(sorted([ga.g[node][i]['weight'] for i in neighbors], reverse=True))
            print(sum([ga.g[node][i]['weight'] for i in neighbors]))

    def test_freq(self, data_path=ALL_DATA_NO_DUP):
        ga = GraphAnalyzer(data_path, load_reader=True, load_g_without_control_structs=False)
        self.assertEqual(ga.g.nodes[STR_BUILD]['frequency'], len(ga.get_program_ids_for_api(STR_BUILD)))
        self.assertEqual(ga.g.edges[STR_BUILD, STR_BUILD_APP]['weight'],
                         len(ga.get_program_ids_with_multiple_apis([STR_BUILD, STR_BUILD_APP])))
        self.assertEqual(ga.g.edges[STR_BUILD, 'DBranch']['weight'],
                         len(ga.get_program_ids_with_multiple_apis([STR_BUILD, 'DBranch'])))
        print(len(ga.get_program_ids_with_multiple_apis([STR_BUILD, 'DBranch'])))

    def test_remove_duplicates(self, data_path=ALL_DATA_NO_DUP):
        ga = GraphAnalyzer(data_path, load_reader=True)
        prog_ids = ga.get_program_ids_for_api('DSubTree')
        print(len(prog_ids))
        prog_ids = [ga.fetch_hashable_data_with_targets(prog_id) for prog_id in prog_ids]
        prog_ids_set = list(set(prog_ids))
        print('prog ids set:', len(prog_ids_set))
        prog_id_nodes = [prog_id[0] for prog_id in prog_ids]
        all_nodes = list(filter(lambda x: x[2] == 0, prog_id_nodes))
        rest_nodes = list(filter(lambda x: x[2] != 0, prog_id_nodes))
        print('all nodes len:', len(all_nodes))
        set_nodes = [i[0] for i in prog_ids_set]
        set_nodes_less_than_2 = list(filter(lambda x: x[2] == 0, set_nodes))
        print('set nodes less than 2:', len(set_nodes_less_than_2))

        print('rest of nodes:', len(rest_nodes))
        set_nodes_more_than_2 = list(filter(lambda x: x[2] != 0, set_nodes))
        print('set nodes more than 2:', len(set_nodes_more_than_2))

    def test_remove_json_duplicates(self, data_path=ALL_DATA_1K_VOCAB):
        ga = GraphAnalyzer(data_path, load_reader=True)
        json_set = set([])
        for program in ga.json_asts:
            json_set.add(str(program))
        print(len(json_set))

    def test_create_5k_no_dup(self):
        old_data_filename_path = '/Users/meghanachilukuri/bayou_mcmc/data_extractor/data/all_data_5k_vocab_no_duplicates/all_data_5k_vocab_no_duplicates.json'
        new_data_filename = 'new_all_data_5k_vocab_no_duplicates.json'
        # copy_json_data_limit_vocab("all_data_no_duplicates.json", new_data_filename, 1000, old_data_dir_path='data/all_data_no_duplicates/')
        copy_data_remove_duplicate(old_data_filename_path, new_data_filename)

    def test_create_2k_no_dup_limit_vocab(self):
        new_data_filename = 'all_data_2k_vocab_no_duplicates.json'
        copy_json_data_limit_vocab("all_data_no_duplicates.json", new_data_filename, 2000,
                                   old_data_dir_path='/Users/meghanachilukuri/bayou_mcmc/data_extractor/data/all_data_no_duplicates/')

    def test_dataset_creator(self, data_path=ALL_DATA_NO_DUP):
        # data_path = '/Users/meghanachilukuri/bayou_mcmc/data_extractor/data/all_data_10k_vocab_no_duplicates/'
        data_path = 'new_all_data_1k_vocab_no_duplicates'
        train_test_set_name = "/final_train_test_sets/"
        dataset_creator = DatasetCreator(data_path, train_test_set_name, verbose=False, min_prog_per_category=1000)
        # dataset_creator.create_curated_dataset()

        dataset_creator.build_and_save_train_test_sets()
        data_dir_name = data_path
        data_path = '../data_extractor/data/new_all_data_1k_vocab_no_duplicates/'
        # data_path = '../data_extractor/data/new_all_data_1k_vocab_no_duplicates/train_test_sets/dataset_creator.pickle'
        create_smaller_test_set(data_path, data_dir_name, train_test_set_name, num_progs_per_category=1000)

    def test_analyze_file(self):
        analyze_file('/Users/meghanachilukuri/bayou_mcmc/data_extractor/data/new_all_data_1k_vocab_no_duplicates/train_test_sets/test/small_min_length_3/', "small_test_set.json", vocab_freq_saved=False)

    def test_build_identical_bayou_dataset(self):
        all_data_bayou_dataset_name = '/Users/meghanachilukuri/bayou_mcmc/data_extractor/data/data_surrounding_methods.json'
        mcmc_dataset_path = '/Users/meghanachilukuri/bayou_mcmc/data_extractor/data/new_all_data_1k_vocab_no_duplicates/train_test_sets/train/all_training_data.json'
        new_bayou_dataset_name = 'training_1k_vocab_apicalls.json'
        bayou_path = '/Users/meghanachilukuri/bayou/src/main/python/bayou/models/low_level_evidences/data/'
        create_identical_bayou_dataset(all_data_bayou_dataset_name, mcmc_dataset_path, new_bayou_dataset_name, bayou_path)

    def test_build_sets_from_creator(self):
        data_path = '../data_extractor/data/new_all_data_1k_vocab_no_duplicates/'
        creator_dir_name = 'fixed_train_test_sets'
        build_sets_from_saved_creator(data_path, creator_dir_name)

    def test_create_small_test_set(self):
        data_path = '../data_extractor/data/new_all_data_1k_vocab_no_duplicates/'
        create_smaller_test_set(data_path, 'new_all_data_1k_vocab_no_duplicates', num_progs_per_category=1000, save=True)

    def test_connected_components(self):
        data_dir_name = 'new_all_data_1k_vocab_no_duplicates'
        train_ga = GraphAnalyzer(data_dir_name, train_test_split='train', filename='all_training_data',
                                 load_reader=True, load_g_without_control_structs=False)
        data_dir_path = '../data_extractor/data/new_all_data_1k_vocab_no_duplicates/'
        creator_dir_path = data_dir_path + "/train_test_sets/"
        f = open(creator_dir_path + "/dataset_creator.pickle", "rb")
        dataset_creator = pickle.load(f)

        api_count = {}

        for category in dataset_creator.categories:
            cat_test_set = dataset_creator.categories[category][0]
            for t in cat_test_set.keys():
                if category != MIN_EQ and category != MAX_EQ and category != 'random':
                    print("category:", category, "label:", t)
                    num_joint = 0
                    num_disjoint = 0

                    for data in cat_test_set[t]:
                        prog_id = data[0]
                        api = dataset_creator.ga.node2vocab[data[1]]
                        dp2 = dataset_creator.ga.node2vocab[data[2]]
                        # len = data[3]

                        # intersection = set(train_ga.g.neighbors(api)).intersection(set(train_ga.g.neighbors(dp2)))
                        intersection = set(train_ga.g[api].keys()).intersection(set(train_ga.g[dp2].keys()))

                        if len(intersection) == 0:
                            num_disjoint += 1
                        else:
                            num_joint += 1

                        for i in [api, dp2]:
                            if i in api_count:
                                api_count[i] += 1
                            else:
                                api_count[i] = 1

                    print("num joint:", num_joint)
                    print("num disjoint", num_disjoint)

        print(sorted(api_count.items(), key=lambda x: x[1], reverse=True))
        print(len(api_count))

    def test_add_length_to_creator(self):
        data_dir_path = '../data_extractor/data/new_all_data_1k_vocab_no_duplicates/'
        train_test_name = 'train_test_sets_new_ex'
        add_prog_length_to_dataset_creator(data_dir_path, train_test_name, save=True)

    def test_build_bayou_test_set(self):
        bayou_data_dir_path = '../data_extractor/data/all_data_no_duplicates_bayou/'
        bayou_data_folder_name = 'all_data_no_duplicates_bayou'
        mcmc_data_dir_path = '../data_extractor/data/all_data_no_duplicates/'

        build_bayou_datasets(mcmc_data_dir_path, bayou_data_dir_path, bayou_data_folder_name)

    def test_pickle_test_set(self):
        data_path = '../data_extractor/data/new_all_data_1k_vocab_no_duplicates/'
        pickle_dump_test_sets(data_path, 'new_all_data_1k_vocab_no_duplicates')

    def test_change_return_type(self):
        old_data_filename_path = '/data/all_data_no_duplicates/train_test_sets/train/all_training_data.json'
        new_data_filename = '/data/all_data_no_duplicates_rt/train/all_training_data.json'
        copy_json_data_change_return_types(old_data_filename_path, new_data_filename)

    def test_change_api_calls(self):
        old_data_filename_path = '/data/all_data_no_duplicates_bayou/train/training_data.json'
        new_data_filename = '/data/all_data_no_duplicates_bayou/test/apicalls_training_data.json'
        copy_bayou_json_data_change_apicalls(old_data_filename_path, new_data_filename)

    def test_ga_info(self):
        data_path = 'all_data_1k_vocab_no_duplicates'
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)
        print("Num progs:", graph_analyzer.num_programs)

        data_path = 'new_all_data_1k_vocab_no_duplicates'
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)
        print("Num progs:", graph_analyzer.num_programs)

    def test_nothing(self):
        # numbers1 = [random.uniform(0, 1) for _ in range(10000)]
        # numbers2 = [random.uniform(0, 1) for _ in range(10000)]
        #
        # numbers1 = np.exp(numbers1)/sum(np.exp(numbers1))
        # numbers2 = np.exp(numbers2)/sum(np.exp(numbers2))
        #
        # check = np.outer(numbers1, numbers2).ravel()
        # print("sum of outer:", sum(check))

        print(scipy.stats.norm([0, 0], 1).pdf([0.5, 0.5]))

    def test_nothing2(self):
        # ga = GraphAnalyzer(ALL_DATA_NO_DUP, load_reader=True)
        # print(ga.num_programs)

        counter = 0
        mcmc_data_dir_path = '../data_extractor/data/all_data_no_duplicates/'
        mcmc_train_f = open(mcmc_data_dir_path + "train_test_sets/train/all_training_data.json", "rb")
        training_progs = ijson.items(mcmc_train_f, 'programs.item')
        for _ in training_progs:
            counter += 1
        print(counter)
        #1386118 - checks out

    def test_nothing3(self):
        # node = [[[1]], [[0]], [1,2,2]]
        node = [[1, 2, 3]]
        length = 10
        not_temp = np.array([node * length])
        print(not_temp)
        print(not_temp + 3)

        nodes = np.array(range(10)).reshape(10, 1)
        nodes = np.expand_dims(nodes, axis=0)
        print(nodes)

        nodes = np.ones([2,2])
        column = np.zeros([2,10])
        col2 = np.ones([2, 5]) * 5
        nodes = np.append(nodes, column, axis=1)
        nodes = np.append(nodes, col2, axis=1)
        print(nodes)
        print(nodes[:, 2:])

        nodes[:, 4:] = np.ones([2, nodes.shape[1]-4]) * 7
        print(nodes)

        for i in nodes:
            print(i)

        # nodes += (not_temp)
        # print(nodes)

    # def test_4_1(self):
    #     prog_ids = list(self.get_program_ids_for_api('DBranch', limit=10))
    #     for i in range(10):
    #         print(i)
    #         print(prog_ids[i])
    #         js = self.get_json_ast(prog_ids[i])
    #         self.get_vectors_and_plot(js, filename=('temp' + str(i)))
    #
    # def test_4_2(self):
    #     for i in range(10):
    #         print(i)
    #         js = self.get_json_ast(i)
    #         self.get_vectors_and_plot(js, filename=('temp' + str(i)))

    # def all_test_code(self):
        # def load_graph_analyzer(path):
        #     with open(path, 'rb') as f:
        #         return pickle.load(f)

        # graph_analyzer = GraphAnalyzer(TESTING, save_reader=True)
        # graph_analyzer = GraphAnalyzer(TESTING, load_reader=True)

        # graph_analyzer = GraphAnalyzer(ALL_DATA_1K_VOCAB, save_reader=True)

        # graph_analyzer = GraphAnalyzer(ALL_DATA_1K_VOCAB, load_reader=True)
        # graph_analyzer.test_5()

        # prog_ids = graph_analyzer.get_program_ids_with_multiple_apis([

        # 'java.io.ByteArrayInputStream.ByteArrayInputStream(byte[])', 'java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()'
        #                                                         ])
        # graph_analyzer.print_summary_stats(prog_ids)
        # prog_ids = graph_analyzer.get_program_ids_with_multiple_apis([
        #
        #
        #                 'java.util.Map<java.lang.String,byte[]>.hashCode()'
        #                                                         ])
        # graph_analyzer.print_summary_stats(prog_ids)

        # prog_ids = graph_analyzer.get_program_ids_with_multiple_apis(
        #     ['DExcept', 'java.lang.Throwable.printStackTrace()', 'java.io.FileReader.FileReader(java.io.File)'])
        # graph_analyzer.print_summary_stats(prog_ids)
        # graph_analyzer.print_programs_from_ids(prog_ids, limit=20)


if __name__ == '__main__':
    unittest.main()

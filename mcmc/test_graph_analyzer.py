import unittest
from data_extractor.graph_analyzer import GraphAnalyzer, STR_BUF, STR_APP, READ_LINE, CLOSE, STR_LEN, STR_BUILD, \
    STR_BUILD_APP, LOWERCASE_LOCALE, DATA_DIR_PATH, ALL_DATA_1K_VOCAB, TESTING, NEW_VOCAB, APIS, RT, FP, TOP, MID, LOW
from test_suite import MOST_COMMON_APIS, MID_COMMON_APIS, UNCOMMON_APIS, MID_COMMON_DISJOINT_PAIRS, \
    MOST_COMMON_DISJOINT_PAIRS, UNCOMMON_DISJOINT_PAIRS

class TestGraphAnalyzer(unittest.TestCase):

    def testing(self, data_path=ALL_DATA_1K_VOCAB):
        graph_analyzer = GraphAnalyzer(data_path, load_reader=True)
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
            'java.util.ArrayList<javax.xml.transform.Source>.ArrayList<Source>()', 'java.lang.String.isEmpty()'
                                                                ])
        graph_analyzer.print_summary_stats(prog_ids)
        prog_ids = graph_analyzer.get_program_ids_with_multiple_apis([
            'java.lang.StringBuilder.append(long)', 'java.lang.String.isEmpty()'
        ])
        graph_analyzer.print_summary_stats(prog_ids)

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

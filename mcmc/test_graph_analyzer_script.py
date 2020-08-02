import unittest
import networkx as nx
from data_extractor.graph_analyzer import GraphAnalyzer, STR_BUF, STR_APP, READ_LINE, CLOSE, STR_LEN, STR_BUILD, \
    STR_BUILD_APP, LOWERCASE_LOCALE, DATA_DIR_PATH, ALL_DATA_1K_VOCAB, TESTING, NEW_VOCAB, APIS, RT, FP, TOP, MID, \
    LOW, ALL_DATA_1K_VOCAB_NO_DUP, ALL_DATA
from test_suite import MOST_COMMON_APIS, MID_COMMON_APIS, UNCOMMON_APIS, MID_COMMON_DISJOINT_PAIRS, \
    MOST_COMMON_DISJOINT_PAIRS, UNCOMMON_DISJOINT_PAIRS


def test_remove_duplicates(data_path=ALL_DATA):
    ga = GraphAnalyzer(data_path, save_reader=True)
    prog_ids = ga.get_program_ids_for_api('DSubTree')
    print(len(prog_ids))
    print(ga.fetch_data_with_targets(0))
    prog_ids = [ga.fetch_hashable_data_with_targets(prog_id) for prog_id in prog_ids]
    prog_ids_set = list(set(prog_ids))
    print(len(prog_ids_set))
    prog_id_nodes = [prog_id[0] for prog_id in prog_ids]
    all_nodes = list(filter(lambda x: x[2] == 0, prog_id_nodes))
    rest_nodes = list(filter(lambda x: x[2] != 0, prog_id_nodes))
    print(len(all_nodes))
    set_nodes = [i[0] for i in prog_ids_set]
    set_nodes_less_than_2 = list(filter(lambda x: x[2] == 0, set_nodes))
    print(len(set_nodes_less_than_2))

    print(len(rest_nodes))
    set_nodes_more_than_2 = list(filter(lambda x: x[2] != 0, set_nodes))
    print(len(set_nodes_more_than_2))


test_remove_duplicates()
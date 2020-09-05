from json_data_extractor import copy_json_data_limit_vocab, view_graph, build_graph_from_json_file, \
    copy_data_remove_duplicate, copy_json_data_change_return_types, copy_data_remove_duplicate_bayou




# copy_json_data_limit_vocab("data_surrounding_methods.json", "all_data_50k_vocab.json", 50000, split_data=1000)

# copy_json_data_limit_vocab("data_surrounding_methods.json", "delete.json", 1000, num_programs=6000, is_test_data=False)


# copy_json_data_limit_vocab("data_surrounding_methods.json", "no_vocab_constraint_min_3.json", 100000, num_programs=10000,
#                            is_test_data=False, min_length=3)

# view_graph("data/delete-6000/delete-6000_api_graph.json")

# build_graph_from_json_file("data/all_data_50k_vocab", "all_data_50k_vocab.json")

copy_data_remove_duplicate_bayou("data/data_surrounding_methods.json", "all_data_no_duplicates_bayou.json")

# copy_data_remove_duplicate("/Users/meghanachilukuri/Documents/GitHub/bayou_mcmc/data_extractor/data/data_surrounding_methods.json", "all_data_no_duplicates.json")
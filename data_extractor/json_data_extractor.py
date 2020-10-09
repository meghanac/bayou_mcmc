import ijson
import json
import os
from collections import defaultdict
import networkx as nx
from networkx.readwrite import json_graph

TRAINING_DATA_DIR_PATH = "data/"
TEST_DATA_DIR_PATH = "data/test_data"

ADD_TO_TYPES = 'add_to_types'
REPLACE_TYPES = 'replace_types'

FP = 'formalParam'
RT = 'returnType'


def copy_data_remove_duplicate(old_data_filename_path, new_data_filename):
    new_dir_name = new_data_filename[:-5]
    new_dir_path = os.path.join(TRAINING_DATA_DIR_PATH, new_dir_name)
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    new_data_path = os.path.join(new_dir_path, new_data_filename)
    old_f = open(old_data_filename_path, 'rb')
    new_f = open(new_data_path, 'w+')
    # analysis_filename = new_data_filename[:-5] + "_analysis.txt"
    # analysis_f = open(os.path.join(new_dir_path, analysis_filename), 'w+')

    # initialize new json file
    new_f.write("{\n")
    new_f.write("\"programs\": [\n")

    data_types = ['ast', 'formalParam', 'returnType']
    counter = 0
    prog_set = set([])
    programs = []
    for program in ijson.items(old_f, 'programs.item'):
        key = (json.dumps(program['ast']), json.dumps(program['returnType']), json.dumps(program['formalParam']))
        counter += 1
        if counter % 100000 == 0:
            print(counter)
        if key not in prog_set:
            prog = {}
            for t in data_types:
                prog[t] = program[t]
            prog_set.add(key)
            programs.append(json.dumps(prog))

    print("There are " + str(len(prog_set)) + " unique programs in dataset.")

    for i in range(len(programs)):
        if i != 0:
            new_f.write(",\n")
        new_f.write(programs[i])

    print(str(len(prog_set)) + " json objects copied into " + new_data_filename)

    # end new json data file
    new_f.write("\n")
    new_f.write("]\n")
    new_f.write("}\n")
    new_f.close()


def copy_data_remove_duplicate_bayou(old_data_filename_path, new_data_filename):
    new_dir_name = new_data_filename[:-5]
    new_dir_path = os.path.join(TRAINING_DATA_DIR_PATH, new_dir_name)
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    new_data_path = os.path.join(new_dir_path, new_data_filename)
    old_f = open(old_data_filename_path, 'rb')
    new_f = open(new_data_path, 'w+')
    # analysis_filename = new_data_filename[:-5] + "_analysis.txt"
    # analysis_f = open(os.path.join(new_dir_path, analysis_filename), 'w+')

    # initialize new json file
    new_f.write("{\n")
    new_f.write("\"programs\": [\n")

    data_types = ['ast', 'formalParam', 'returnType', 'keywords', 'apicalls', 'types']
    counter = 0
    prog_set = set([])
    programs = []
    for program in ijson.items(old_f, 'programs.item'):
        key = (json.dumps(program['ast']), json.dumps(program['returnType']), json.dumps(program['formalParam']))
        counter += 1
        if counter % 100000 == 0:
            print(counter)
        if key not in prog_set:
            prog = {}
            for t in data_types:
                prog[t] = program[t]
            prog_set.add(key)
            programs.append(json.dumps(prog))

    print("There are " + str(counter) + " programs in the old dataset.")
    print("There are " + str(len(prog_set)) + " unique programs in prog_set.")
    print("There are " + str(len(programs)) + " unique programs in dataset.")

    for i in range(len(programs)):
        if i != 0:
            new_f.write(",\n")
        new_f.write(programs[i])

    print(str(len(prog_set)) + " json objects copied into " + new_data_filename)

    # end new json data file
    new_f.write("\n")
    new_f.write("]\n")
    new_f.write("}\n")
    new_f.close()


def copy_json_data_change_return_types(old_data_filename_path, new_data_filename_path):
    # new_dir_name = new_data_filename[:-5]
    # new_dir_path = os.path.join(TRAINING_DATA_DIR_PATH, new_dir_name)
    # if not os.path.exists(new_dir_path):
    #     os.mkdir(new_dir_path)
    # new_data_path = os.path.join(new_dir_path, new_data_filename)
    print(os.path.dirname(os.path.realpath(__file__)))
    old_f = open(os.path.dirname(os.path.realpath(__file__)) + old_data_filename_path, 'rb')
    new_f = open(os.path.dirname(os.path.realpath(__file__)) + new_data_filename_path, 'w+')

    # initialize new json file
    new_f.write("{\n")
    new_f.write("\"programs\": [\n")

    def get_last_node_returns(program):
        if len(program) == 0:
            return 'void'
        if program[-1]['node'] == 'DBranch':
            last_node = program[-1]['_then']
            return get_last_node_returns(last_node)
        elif program[-1]['node'] == 'DExcept':
            last_node = program[-1]['_try']
            return get_last_node_returns(last_node)
        elif program[-1]['node'] == 'DLoop':
            last_node = program[-1]['_body']
            return get_last_node_returns(last_node)
        else:
            last_node = program[-1]
            return last_node['_returns']

    prog_set = []
    num_rt_changed = 0
    key_error = 0
    num_void = 0
    for program in ijson.items(old_f, 'programs.item'):
        # prog_set.add(json.dumps(program))
        # print(program)
        # print(program['ast']['_nodes'])
        # print(program['ast']['_nodes'][-1])
        # print(program['ast']['_nodes'][-1]['_returns'])
        # print(program['returnType'])
        last_node_returns = get_last_node_returns(program['ast']['_nodes'])
        try:
            if program['returnType'] != last_node_returns:
                num_rt_changed += 1
                if program['returnType'] == 'void':
                    num_void += 1
            program['returnType'] = last_node_returns
            prog_set.append(json.dumps(program))
        except KeyError:
            print(program['ast']['_nodes'][-1])
            key_error += 1
            continue

    print("There are " + str(len(prog_set)) + " programs in dataset.")
    print("There are " + str(len(set(prog_set))) + " unique programs in dataset.")
    print("Number of key errors:", key_error)
    print("Number of return types changed:", num_rt_changed)
    print("Number of void returns changed:", num_void)

    prog_set = list(prog_set)
    for i in range(len(prog_set)):
        if i != 0:
            new_f.write(",\n")
        new_f.write(prog_set[i])

    print(str(len(prog_set)) + " json objects copied into " + new_data_filename_path)

    # end new json data file
    new_f.write("\n")
    new_f.write("]\n")
    new_f.write("}\n")


def get_apis(program, api_set):
    if len(program) == 0:
        return api_set
    for prog in program:
        if prog['node'] == 'DBranch':
            api_set.update(get_apis(prog['_then'], api_set))
            api_set.update(get_apis(prog['_cond'], api_set))
            api_set.update(get_apis(prog['_else'], api_set))
        elif prog['node'] == 'DExcept':
            api_set.update(get_apis(prog['_try'], api_set))
            api_set.update(get_apis(prog['_catch'], api_set))
        elif prog['node'] == 'DLoop':
            api_set.update(get_apis(prog['_body'], api_set))
            api_set.update(get_apis(prog['_cond'], api_set))
        else:
            api_set.add(prog['_call'])

    return api_set


def copy_bayou_json_data_change_apicalls(old_data_filename_path, new_data_filename_path):
    print(os.path.dirname(os.path.realpath(__file__)))
    old_f = open(os.path.dirname(os.path.realpath(__file__)) + old_data_filename_path, 'rb')
    new_f = open(os.path.dirname(os.path.realpath(__file__)) + new_data_filename_path, 'w+')

    # initialize new json file
    new_f.write("{\n")
    new_f.write("\"programs\": [\n")

    prog_set = []
    for program in ijson.items(old_f, 'programs.item'):
        apis = get_apis(program['ast']['_nodes'], set([]))
        # print(apis)
        program['apicalls'] = list(apis)
        prog_set.append(json.dumps(program))

    print("There are " + str(len(prog_set)) + " programs in dataset.")
    print("There are " + str(len(set(prog_set))) + " unique programs in dataset.")

    prog_set = list(prog_set)
    for i in range(len(prog_set)):
        if i != 0:
            new_f.write(",\n")
        new_f.write(prog_set[i])

    print(str(len(prog_set)) + " json objects copied into " + new_data_filename_path)

    # end new json data file
    new_f.write("\n")
    new_f.write("]\n")
    new_f.write("}\n")


def create_identical_bayou_dataset(all_data_bayou_dataset_path, mcmc_dataset_path, new_bayou_dataset_name, new_bayou_data_dir_path, types=None):
    mcmc_f = open(mcmc_dataset_path, 'rb')
    prog_set = set([])
    for program in ijson.items(mcmc_f, 'programs.item'):
        key = (json.dumps(program['ast']), json.dumps(program['returnType']), json.dumps(program['formalParam']))
        prog_set.add(key)

    print("There are " + str(len(prog_set)) + " programs in dataset.")
    # print("There are " + str(len(set(prog_set))) + " unique programs in dataset.")

    mcmc_f.close()


    all_bayou_f = open(all_data_bayou_dataset_path, 'rb')
    new_bayou_f = open(new_bayou_data_dir_path + new_bayou_dataset_name, "w+")

    # initialize new json file
    new_bayou_f.write("{\n")
    new_bayou_f.write("\"programs\": [\n")

    data_types = ['ast', 'types']
    counter = 0
    bayou_prog_set = set([])
    programs = []
    skipped_progs = 0
    for program in ijson.items(all_bayou_f, 'programs.item'):
        key = (json.dumps(program['ast']), json.dumps(program['returnType']), json.dumps(program['formalParam']))
        counter += 1
        if counter % 100000 == 0:
            print(counter)
            print("added progs:", len(bayou_prog_set))
            print("num skipped:", skipped_progs)
            print("")
        if key in prog_set and key not in bayou_prog_set:
            prog = {}
            for t in data_types:
                prog[t] = program[t]
            apis = get_apis(program['ast']['_nodes'], set([]))
            prog['apicalls'] = list(apis)

            if types is not None:
                if types == ADD_TO_TYPES:
                    curr_types = set(prog['types'])
                    curr_types.update(set(program[FP]))
                    curr_types.add(program[RT])
                    curr_types.difference_update(set(prog['types']))
                    prog['types'] += list(curr_types)
                elif types == REPLACE_TYPES:
                    try:
                        rt_fp = set(program[FP])
                    except KeyError:
                        skipped_progs += 1
                        continue
                    rt_fp.add(program[RT])
                    prog['types'] = list(rt_fp)

            bayou_prog_set.add(key)
            programs.append(json.dumps(prog))

        if len(bayou_prog_set) == len(prog_set):
            break

    print("There are " + str(counter) + " programs in the old dataset.")
    print("There are " + str(len(prog_set)) + " unique programs in prog_set.")
    print("There are " + str(len(programs)) + " unique programs in dataset.")

    for i in range(len(programs)):
        if i != 0:
            new_bayou_f.write(",\n")
        new_bayou_f.write(programs[i])

    print(str(len(programs)) + " json objects copied into " + new_bayou_dataset_name)

    # end new json data file
    new_bayou_f.write("\n")
    new_bayou_f.write("]\n")
    new_bayou_f.write("}\n")
    new_bayou_f.close()
    all_bayou_f.close()


def copy_json_data(old_data_filename, new_data_filename, num_programs=None, is_test_data=False,
                   old_data_dir_path=None, new_data_dir_path=None):
    """

    :param new_data_dir_path:
    :param old_data_dir_path:
    :param is_test_data:
    :param old_data_filename:
    :param new_data_filename:
    :param num_programs:
    :return: None
    """
    # update new data filename to have num data points in name
    if num_programs is not None:
        if new_data_filename[-5:] == '.json':
            new_data_filename = new_data_filename[:-5] + "-" + str(num_programs) + ".json"
        else:
            new_data_filename += "-" + str(num_programs) + ".json"

    # Use directory path to old and new data if specified, else use default directory paths
    if old_data_dir_path is not None:
        old_f = open(os.path.join(old_data_dir_path, old_data_filename), 'rb')
        if new_data_dir_path is not None:
            new_f = open(os.path.join(new_data_dir_path, new_data_filename), 'w+')
        else:
            new_f = open(os.path.join(old_data_dir_path, new_data_filename), 'w+')
    else:
        # open and create appropriate data files
        if is_test_data:
            old_f = open(os.path.join(TEST_DATA_DIR_PATH, old_data_filename), 'rb')
            new_f = open(os.path.join(TEST_DATA_DIR_PATH, new_data_filename), 'w+')
        else:
            old_f = open(os.path.join(TRAINING_DATA_DIR_PATH, old_data_filename), 'rb')
            new_f = open(os.path.join(TRAINING_DATA_DIR_PATH, new_data_filename), 'w+')

    # initialize new json file
    new_f.write("{\n")
    new_f.write("\"programs\": [\n")

    # initialize number of programs copied counter
    counter = 0

    data_types = ['ast', 'formalParam', 'returnType', 'keywords']

    for program in ijson.items(old_f, 'programs.item'):
        if counter != 0:
            new_f.write(",\n")

        for i in range(len(data_types)):
            if i == 0:
                new_f.write("{\n")

            data_type = data_types[i]

            new_f.write("\"" + data_type + "\":")
            new_f.write(json.dumps(program[data_type]))

            if i != len(data_types) - 1:
                new_f.write(",\n")
            else:
                new_f.write("}")

        counter += 1

        if num_programs is not None and counter == num_programs:
            break

    print(str(counter) + " json objects copied into " + new_data_filename)

    # end new json data file
    new_f.write("\n")
    new_f.write("]\n")
    new_f.write("}\n")


def add_branched_to_vocab(valid_prog, vocab, vocab_size, node, prog_length, vocab_freq, apis_list, branching, vocab_num):
    if node['node'] == 'DLoop':
        # print("DLOOP keys:", node.keys())
        types = ['_body', '_cond']
        apis_list.append('DLoop')
        vocab_freq['DLoop'][0] += 1
    elif node['node'] == 'DExcept':
        # print("DExcept keys:", node.keys())
        types = ['_catch', '_try']
        apis_list.append("DExcept")
        vocab_freq['DExcept'][0] += 1
    elif node['node'] == 'DBranch':
        # print("DBranch keys:", node.keys())
        types = ['_cond', '_else', '_then']
        apis_list.append('DBranch')
        vocab_freq['DBranch'][0] += 1
    else:
        types = []
        print("skipped node type:", node['node'])

        # print(node['node'])
    if len(types) > 0:
        for t in types:
            for i in range(len(node[t])):
                branching_apis = []
                if node[t][i]['node'] != 'DAPICall':
                    valid, vocab, vocab_size, prog_length, vocab_freq, branching_apis, branching = add_branched_to_vocab(
                        valid_prog,
                        vocab,
                        vocab_size,
                        node[t][i],
                        prog_length,
                        vocab_freq,
                        branching_apis,
                        branching,
                        vocab_num)
                    valid_prog = valid and valid_prog
                else:
                    if len(node[t][i]) > 0:
                        valid, vocab, vocab_size, prog_length, vocab_freq, branching_apis = add_call_to_vocab(
                            valid_prog,
                            vocab,
                            vocab_size,
                            node[t][i][
                                "_call"],
                            prog_length,
                            vocab_freq,
                            branching_apis,
                            vocab_num)
                        valid_prog = valid and valid_prog

                apis_list += branching_apis

                # add branching information to branching dict
                for i in branching_apis:
                    try:
                        branching[node['node']][t][i] += 1
                    except KeyError:
                        branching[node['node']][t][i] = 1

    return valid_prog, vocab, vocab_size, prog_length, vocab_freq, apis_list, branching


def add_call_to_vocab(valid_prog, vocab, vocab_size, call, prog_length, vocab_freq, apis_list, vocab_num):
    if call != '' and call != '__delim__':
        prog_length += 1
        apis_list.append(call)
        try:
            vocab_freq[call][0] += 1
        except KeyError:
            vocab_freq[call] = [1, {}, {}]

    if vocab_size < vocab_num:
        if call != '':
            vocab.add(call)
            vocab_size = len(vocab)
    else:
        in_vocab = call in vocab
        valid_prog = valid_prog and in_vocab

    return valid_prog, vocab, vocab_size, prog_length, vocab_freq, apis_list


def update_api_cofrequencies(apis_list, vocab_freq, prog_length):
    apis_list = list(set(apis_list))
    for api1 in apis_list:
        try:
            vocab_freq[api1][2][prog_length] += 1
        except KeyError:
            vocab_freq[api1][2][prog_length] = 1
        for api2 in apis_list:
            # if api1 != api2:  # by leaving this in we're able to know how many distinct programs this API shows up in
            try:
                vocab_freq[api1][1][api2] += 1
            except KeyError:
                vocab_freq[api1][1][api2] = 1
    return vocab_freq


def get_top_k_cofreq(cofreq_dict, k):
    sorted_items = sorted(cofreq_dict.items(), key=lambda v: v[1], reverse=True)
    if len(sorted_items) > k:
        return sorted_items[:k]
    else:
        return sorted_items


def remove_self_from_cofreq_list(api, api_cofreq_dict):
    try:
        del api_cofreq_dict[api]
    except KeyError as e:
        print("Error: couldn't delete api from it's cofreq dict: ", e)
    return api_cofreq_dict


def get_sorted_api_cofreq(vocab_freq):
    vocab_items = vocab_freq.items()
    sorted_apis = sorted([((i[0], i[1][1][i[0]]), i[1][1], sorted(i[1][2].items())) for i in vocab_items],
                         key=lambda k: k[0][1],
                         reverse=True)
    sorted_apis = [(i[0], remove_self_from_cofreq_list(i[0][0], i[1]), i[2]) for i in sorted_apis]
    return sorted_apis


def get_api_cofrequencies(sorted_apis):
    api_cofreq = [(i[0], get_top_k_cofreq(i[1], 10), i[2]) for i in sorted_apis]
    return api_cofreq


def copy_json_data_limit_vocab(old_data_filename, new_data_filename, vocab_num, num_programs=None, is_test_data=False,
                               old_data_dir_path=None, new_data_dir_path=None, min_length=None, split_data=None):
    """

    :param vocab_num:
    :param new_data_dir_path:
    :param old_data_dir_path:
    :param is_test_data:
    :param old_data_filename:
    :param new_data_filename:
    :param num_programs:
    :return: None
    """
    # update new data filename to have num data points in name
    if num_programs is not None:
        if new_data_filename[-5:] == '.json':
            new_data_filename = new_data_filename[:-5] + "-" + str(num_programs) + ".json"
        else:
            new_data_filename += "-" + str(num_programs) + ".json"

    analysis_filename = new_data_filename[:-5] + "_analysis.txt"

    if is_test_data:
        old_data_path = os.path.join(TEST_DATA_DIR_PATH, old_data_filename)
        new_data_path = os.path.join(TEST_DATA_DIR_PATH, new_data_filename)
        old_f = open(old_data_path, 'rb')
        new_f = open(new_data_path, 'w+')
        analysis_f = open(os.path.join(TEST_DATA_DIR_PATH, analysis_filename), 'w+')
        test_f = None
        new_dir_path = None
    else:
        if old_data_dir_path is None:
            old_data_dir_path = TRAINING_DATA_DIR_PATH

        if new_data_dir_path is None:
            new_data_dir_path = TRAINING_DATA_DIR_PATH

        new_dir_name = new_data_filename[:-5]
        new_dir_path = os.path.join(new_data_dir_path, new_dir_name)
        if not os.path.exists(new_dir_path):
            os.mkdir(new_dir_path)
        old_data_path = os.path.join(old_data_dir_path, old_data_filename)
        new_data_path = os.path.join(new_dir_path, new_data_filename)
        old_f = open(old_data_path, 'rb')
        new_f = open(new_data_path, 'w+')
        analysis_f = open(os.path.join(new_dir_path, analysis_filename), 'w+')

        if split_data is not None:
            test_data_filename = new_data_filename[:-5] + "_test.json"
            test_data_path = os.path.join(new_dir_path, test_data_filename)
            test_f = open(test_data_path, 'w+')
        else:
            test_f = None


    # initialize new json file
    new_f.write("{\n")
    new_f.write("\"programs\": [\n")

    if test_f is not None:
        test_f.write("{\n")
        test_f.write("\"programs\": [\n")

    # initialize number of programs copied counter
    counter = 0
    test_counter = 0

    data_types = ['ast', 'formalParam', 'returnType']

    vocab = set([])
    vocab_size = len(vocab)

    num_skipped = 0

    prog_sizes = defaultdict()
    vocab_freq = defaultdict()
    branching = defaultdict()

    # Add branching to vocab_freq
    vocab_freq['DBranch'] = [0, {}, {}]  # count, dict(api : count), dict(prog_length : count)
    vocab_freq['DLoop'] = [0, {}, {}]
    vocab_freq['DExcept'] = [0, {}, {}]

    # create dictionary to save information about branching
    branching['DBranch'] = {'_cond': {}, '_else': {}, '_then': {}}
    branching['DExcept'] = {'_catch': {}, '_try': {}}
    branching['DLoop'] = {'_body': {}, '_cond': {}}

    valid_prog = False
    added_to_training = False
    added_to_test = False
    for program in ijson.items(old_f, 'programs.item'):
        if added_to_training and valid_prog:
            new_f.write(",\n")

        valid_prog = True
        added_to_test = False
        added_to_training = False
        prog_length = 0
        apis_list = []
        new_vocab = vocab.copy()
        new_vocab_size = vocab_size
        new_vocab_freq = vocab_freq.copy()
        new_branching = branching.copy()
        for node in program['ast']['_nodes']:
            try:
                if node['node'] == 'DAPICall':
                    valid, new_vocab, new_vocab_size, prog_length, new_vocab_freq, apis_list = add_call_to_vocab(
                        valid_prog, vocab, vocab_size, node['_call'], prog_length, new_vocab_freq, apis_list, vocab_num)
                elif node['node'] == 'DSubTree':
                    print("nested Dsubtree")
                    print(node)
                    valid = False
                else:
                    valid, new_vocab, new_vocab_size, prog_length, new_vocab_freq, apis_list, new_branching = add_branched_to_vocab(
                        valid_prog, vocab, vocab_size, node, prog_length, new_vocab_freq, apis_list, new_branching, vocab_num)

                valid_prog = valid and valid_prog
            except KeyError as e:
                print(e)
                print(node)
                num_skipped += 1
                valid_prog = False
                break

        if min_length is not None:
            valid_prog = valid_prog and (prog_length >= min_length)

        if valid_prog:
            if test_f is not None and counter % split_data == 0:
                f = test_f
                if test_counter > 0:
                    f.write(",\n")
                test_counter += 1
            else:
                vocab = new_vocab
                vocab_size = new_vocab_size
                vocab_freq = new_vocab_freq
                branching = new_branching
                try:
                    prog_sizes[prog_length] += 1
                except KeyError:
                    prog_sizes[prog_length] = 1
                vocab_freq = update_api_cofrequencies(apis_list, vocab_freq, prog_length)

                f = new_f
                added_to_training = True

            for i in range(len(data_types)):
                if i == 0:
                    f.write("{\n")

                data_type = data_types[i]

                f.write("\"" + data_type + "\":")
                f.write(json.dumps(program[data_type]))

                if i != len(data_types) - 1:
                    f.write(",\n")
                else:
                    f.write("}")

            counter += 1

            if counter % 100000 == 0:
                print("Copied " + str(counter-test_counter) + " asts to training json file")
                if test_f is not None:
                    print("Copied " + str(test_counter) + " asts to testing json file")

        if num_programs is not None and (counter-test_counter) == num_programs:
            break

    training_counter = counter - test_counter
    print(str(training_counter) + " json objects copied into " + new_data_filename)
    print("Vocab size:", len(vocab))
    print("Num asts skipped because of no '_call' KeyError:", num_skipped)

    # end new json data file
    new_f.write("\n")
    new_f.write("]\n")
    new_f.write("}\n")

    if test_f is not None:
        print(str(test_counter) + " json objects copied into " + test_data_filename)
        test_f.write("\n")
        test_f.write("]\n")
        test_f.write("}\n")

    analysis_f.write("Old data filename: " + old_data_filename + "\n")
    analysis_f.write("New data filename: " + new_data_filename + "\n")
    analysis_f.write("Vocabulary size limit: " + str(vocab_num) + "\n")
    analysis_f.write("Actual vocabulary size: " + str(vocab_size) + "\n")
    analysis_f.write("Number of programs in new dataset: " + str(training_counter) + "\n")
    if min_length is not None:
        analysis_f.write("Minimum program length: " + str(min_length) + "\n")

    analysis_f.write("\n")

    sorted_prog_length = sorted(prog_sizes.items(), key=lambda k: k[1], reverse=True)
    analysis_f.write(
        "Program Length Frequency (In Order of Most-Least Frequent) (Excludes 'DBranch', 'DLoop' and 'DExcept'):\n")
    for i in range(len(sorted_prog_length)):
        analysis_f.write(
            "\t program length of " + str(sorted_prog_length[i][0]) + ": " + str(sorted_prog_length[i][1]) + " (" + str(
                round(100.0 * sorted_prog_length[i][1] / training_counter, 4)) + "%)\n")
    analysis_f.write("\n")

    sorted_vocab_freq_items = sorted(vocab_freq.items(), key=lambda k: k[1][0], reverse=True)
    sorted_vocab_freq = [(i[0], i[1][0]) for i in sorted_vocab_freq_items]
    analysis_f.write("Vocabulary Occurrence (Includes Multiple Occurrences in Single Program):\n")
    for i in range(len(sorted_vocab_freq)):
        analysis_f.write(
            "\t" + str(i + 1) + ") " + sorted_vocab_freq[i][0] + ": " + str(sorted_vocab_freq[i][1]) + "\n")
    analysis_f.write("\n")

    sorted_apis = get_sorted_api_cofreq(vocab_freq)
    api_cofreq = get_api_cofrequencies(sorted_apis)
    analysis_f.write("API Cofrequency- Top 10\n")
    for i in range(len(api_cofreq)):
        analysis_f.write("\t" + str(i + 1) + ") " + api_cofreq[i][0][0] + ": Found in " + str(
            api_cofreq[i][0][1]) + " distinct programs (" + str(
            round((100.0 * api_cofreq[i][0][1]) / training_counter, 4)) + "%)\n")

        analysis_f.write("\t\t Program lengths: {")
        for k in range(len(api_cofreq[i][2])):
            prog_len = api_cofreq[i][2][k]
            analysis_f.write(str(prog_len[0]) + ": " + str(prog_len[1]) + " ("
                             + str(round(100.0*prog_len[1]/api_cofreq[i][0][1], 2)) + "%)")
            if k != len(api_cofreq[i][2]) - 1:
                analysis_f.write(", ")
        analysis_f.write("} \n")

        for j in api_cofreq[i][1]:
            analysis_f.write("\t\t\t" + j[0] + ": " + str(j[1]) + "\n")
        analysis_f.write("\n")

    analysis_f.write("\n")

    analysis_f.write("Top 50 APIs Per Branching Condition (Includes Multiple Occurrences within Single Program)\n")
    for branch_type in branching.keys():
        analysis_f.write(branch_type + "\n")
        for t in branching[branch_type].keys():
            analysis_f.write("\t" + t + ":\n")
            t_cofreq = get_top_k_cofreq(branching[branch_type][t], 50)
            for i in range(len(t_cofreq)):
                analysis_f.write("\t\t" + str(i + 1) + ") " + t_cofreq[i][0] + ": " + str(t_cofreq[i][1]) + "\n")
            analysis_f.write("\n")
        analysis_f.write("\n")

    new_f.close()
    analysis_f.close()
    old_f.close()

    if test_f is not None:
        test_f.close()
        analyze_file(new_dir_path, test_data_filename)

    build_graph(sorted_apis, new_data_filename, new_dir_path)


def build_graph(sorted_apis, new_data_filename, new_dir_path, return_g_without_control_structs=True):
    # build graph
    g = nx.Graph()

    node_attr = [(i[0][0], {'frequency': i[0][1]}) for i in sorted_apis]
    g.add_nodes_from(node_attr)

    nodes_edges = [(i[0][0], i[1].items()) for i in sorted_apis]
    for node, edges in nodes_edges:
        g.add_edges_from([(node, edge[0], {'weight': edge[1]}) for edge in edges])
    # dump graph to json file
    data = json_graph.adjacency_data(g)
    s = json.dumps(data)
    graph_filename = new_data_filename[:-5] + "_graph.json"
    graph_f = open(os.path.join(new_dir_path, graph_filename), "w+")
    graph_f.write(s)
    graph_f.close()

    if not return_g_without_control_structs:
        orig_g = g.copy()
        orig_filename = graph_filename

    g.remove_node('DBranch')
    g.remove_node('DLoop')
    g.remove_node('DExcept')

    # dump graph to json file
    data = json_graph.adjacency_data(g)
    s = json.dumps(data)
    graph_filename = new_data_filename[:-5] + "_api_graph.json"
    graph_f = open(os.path.join(new_dir_path, graph_filename), "w+")
    graph_f.write(s)
    graph_f.close()

    if return_g_without_control_structs:
        return g, os.path.join(new_dir_path, graph_filename)
    else:
        return orig_g, orig_filename


def load_graph(path):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    d = json.load(open(os.path.join(dir_path, path)))
    g = json_graph.adjacency_graph(d)
    return g


def view_graph(path):
    g = load_graph(path)
    print(nx.algorithms.components.number_connected_components(g))


def get_vocab_frequencies(f, vocab_num):
    # initialize number of programs copied counter
    counter = 0

    vocab = set([])
    vocab_size = len(vocab)

    num_skipped = 0

    prog_sizes = defaultdict()
    vocab_freq = defaultdict()
    branching = defaultdict()

    # Add branching to vocab_freq
    vocab_freq['DBranch'] = [0, {}, {}]  # count, dict(api : count), dict(prog_length : count)
    vocab_freq['DLoop'] = [0, {}, {}]
    vocab_freq['DExcept'] = [0, {}, {}]

    # create dictionary to save information about branching
    branching['DBranch'] = {'_cond': {}, '_else': {}, '_then': {}}
    branching['DExcept'] = {'_catch': {}, '_try': {}}
    branching['DLoop'] = {'_body': {}, '_cond': {}}

    for program in ijson.items(f, 'programs.item'):
        valid_prog = True
        prog_length = 0
        apis_list = []
        new_vocab = vocab.copy()
        new_vocab_size = vocab_size
        new_vocab_freq = vocab_freq.copy()
        new_branching = branching.copy()
        for node in program['ast']['_nodes']:
            try:
                if node['node'] == 'DAPICall':
                    valid, new_vocab, new_vocab_size, prog_length, new_vocab_freq, apis_list = add_call_to_vocab(
                        valid_prog, vocab, vocab_size, node['_call'], prog_length, new_vocab_freq, apis_list, vocab_num)
                elif node['node'] == 'DSubTree':
                    print("nested Dsubtree")
                    print(node)
                    valid = False
                else:
                    valid, new_vocab, new_vocab_size, prog_length, new_vocab_freq, apis_list, new_branching = add_branched_to_vocab(
                        valid_prog, vocab, vocab_size, node, prog_length, new_vocab_freq, apis_list, new_branching,
                        vocab_num)

                valid_prog = valid and valid_prog
            except KeyError as e:
                print(e)
                print(node)
                num_skipped += 1
                valid_prog = False
                break

        if valid_prog:
            vocab = new_vocab
            vocab_size = new_vocab_size
            vocab_freq = new_vocab_freq
            branching = new_branching
            try:
                prog_sizes[prog_length] += 1
            except KeyError:
                prog_sizes[prog_length] = 1
            vocab_freq = update_api_cofrequencies(apis_list, vocab_freq, prog_length)
            counter += 1

    data = {"counter": counter, "vocab_freq": vocab_freq, "prog_sizes": prog_sizes,
            "vocab": vocab, "num_skipped": num_skipped, "vocab_size": vocab_size, "branching": branching}

    return data


def build_graph_from_json_file(dir_path, filename, vocab_freq_saved=False, vocab_num=1000000000000,
                               return_g_without_control_structs=True):
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    f = open(os.path.join(dir_path, filename), 'r')
    if vocab_freq_saved:
        vocab_f = open(os.path.join(dir_path, filename[:-5] + "_vocab_freq.json"), "r")
        data = json.load(vocab_f)
        vocab_freq = data['vocab_freq']
    else:
        vocab_freq_data = get_vocab_frequencies(f, vocab_num)
        vocab_freq = vocab_freq_data['vocab_freq']
        dump_vocab_freq(dir_path, filename, vocab_freq_data)
    sorted_apis = get_sorted_api_cofreq(vocab_freq)
    g, _ = build_graph(sorted_apis, filename, dir_path,
                       return_g_without_control_structs=return_g_without_control_structs)
    print(nx.algorithms.components.number_connected_components(g))
    return g


def dump_vocab_freq(dir_path, filename, vocab_freq_data):
    vocab_freq_data['vocab'] = list(vocab_freq_data['vocab'])
    json_data = json.dumps(dict(vocab_freq_data))
    filename = filename[:-5] + "_vocab_freq.json"
    f = open(os.path.join(dir_path, filename), 'w+')
    f.write(json_data)
    f.close()


def analyze_file(dir_path, filename, vocab_freq_saved=True):
    analysis_filename = filename[:-5] + "_analysis.txt"
    f = open(os.path.join(dir_path, filename), 'r')
    analysis_f = open(os.path.join(dir_path, analysis_filename), 'w+')

    vocab_num = 100000000000

    if vocab_freq_saved:
        vocab_f = open(os.path.join(dir_path, filename[:-5] + "_vocab_freq.json"), "r")
        vocab_freq_data = json.load(vocab_f)
    else:
        vocab_freq_data = get_vocab_frequencies(f, vocab_num)
        dump_vocab_freq(dir_path, filename, vocab_freq_data)

    vocab_freq = vocab_freq_data['vocab_freq']
    counter = vocab_freq_data['counter']
    vocab = set(vocab_freq_data['vocab'])
    num_skipped = vocab_freq_data['num_skipped']
    prog_sizes = vocab_freq_data['prog_sizes']
    vocab_size = vocab_freq_data['vocab_size']
    branching = vocab_freq_data['branching']

    print(str(counter) + " json objects in " + filename)
    print("Vocab size:", len(vocab))
    print("Num asts skipped because of no '_call' KeyError:", num_skipped)

    analysis_f.write("Filename: " + filename + "\n")
    analysis_f.write("Actual vocabulary size: " + str(vocab_size) + "\n")
    analysis_f.write("Number of programs in new dataset: " + str(counter) + "\n")
    analysis_f.write("\n")

    sorted_prog_length = sorted(prog_sizes.items(), key=lambda k: k[1], reverse=True)
    analysis_f.write(
        "Program Length Frequency (In Order of Most-Least Frequent) (Excludes 'DBranch', 'DLoop' and 'DExcept'):\n")
    for i in range(len(sorted_prog_length)):
        analysis_f.write(
            "\t program length of " + str(sorted_prog_length[i][0]) + ": " + str(sorted_prog_length[i][1]) + " (" + str(
                round(100.0 * sorted_prog_length[i][1] / counter, 4)) + "%)\n")
    analysis_f.write("\n")

    sorted_vocab_freq_items = sorted(vocab_freq.items(), key=lambda k: k[1][0], reverse=True)
    sorted_vocab_freq = [(i[0], i[1][0]) for i in sorted_vocab_freq_items]
    analysis_f.write("Vocabulary Occurrence (Includes Multiple Occurrences in Single Program):\n")
    for i in range(len(sorted_vocab_freq)):
        analysis_f.write(
            "\t" + str(i + 1) + ") " + sorted_vocab_freq[i][0] + ": " + str(sorted_vocab_freq[i][1]) + "\n")
    analysis_f.write("\n")

    sorted_apis = get_sorted_api_cofreq(vocab_freq)
    api_cofreq = get_api_cofrequencies(sorted_apis)
    analysis_f.write("API Cofrequency- Top 10\n")
    for i in range(len(api_cofreq)):
        analysis_f.write("\t" + str(i + 1) + ") " + api_cofreq[i][0][0] + ": Found in " + str(
            api_cofreq[i][0][1]) + " distinct programs (" + str(
            round((100.0 * api_cofreq[i][0][1]) / counter, 4)) + "%)\n")

        analysis_f.write("\t\t Program lengths: {")
        for k in range(len(api_cofreq[i][2])):
            prog_len = api_cofreq[i][2][k]
            analysis_f.write(str(prog_len[0]) + ": " + str(prog_len[1]) + " ("
                             + str(round(100.0 * prog_len[1] / api_cofreq[i][0][1], 2)) + "%)")
            if k != len(api_cofreq[i][2]) - 1:
                analysis_f.write(", ")
        analysis_f.write("} \n")

        for j in api_cofreq[i][1]:
            analysis_f.write("\t\t\t" + j[0] + ": " + str(j[1]) + "\n")
        analysis_f.write("\n")

    analysis_f.write("\n")

    analysis_f.write("Top 50 APIs Per Branching Condition (Includes Multiple Occurrences within Single Program)\n")
    for branch_type in branching.keys():
        analysis_f.write(branch_type + "\n")
        for t in branching[branch_type].keys():
            analysis_f.write("\t" + t + ":\n")
            t_cofreq = get_top_k_cofreq(branching[branch_type][t], 50)
            for i in range(len(t_cofreq)):
                analysis_f.write("\t\t" + str(i + 1) + ") " + t_cofreq[i][0] + ": " + str(t_cofreq[i][1]) + "\n")
            analysis_f.write("\n")
        analysis_f.write("\n")

    build_graph(sorted_apis, filename, dir_path)

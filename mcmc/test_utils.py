import os

from node import Node, SIBLING_EDGE, CHILD_EDGE, DNODES, DBRANCH, DLOOP, DEXCEPT, START, STOP, EMPTY
from mcmc import MCMCProgram

# Shorthand for nodes
STR_BUF = 'java.lang.StringBuffer.StringBuffer()'
STR_APP = 'java.lang.StringBuffer.append(java.lang.String)'
READ_LINE = 'java.io.BufferedReader.readLine()'
CLOSE = 'java.io.InputStream.close()'
STR_LEN = 'java.lang.String.length()'
STR_BUILD = 'java.lang.StringBuilder.StringBuilder(int)'
STR_BUILD_APP = 'java.lang.StringBuilder.append(java.lang.String)'

def create_base_program(saved_model_path, constraints, ret_type, fp, ordered=True, exclude=None, debug=False,
                        verbose=False):
    test_prog = MCMCProgramWrapper(saved_model_path, constraints, ret_type, fp, debug=debug, verbose=verbose,
                                   exclude=exclude, ordered=ordered)
    test_prog.update_nodes_and_edges()
    expected_nodes = [START]
    expected_edges = []
    for i in test_prog.constraints:
        expected_nodes.append(i)
        expected_edges.append(False)
    return test_prog, expected_nodes, expected_edges

def create_str_buf_base_program(saved_model_path):
    return create_base_program(saved_model_path, [STR_BUF, 'abc'], ["void"], ["__delim__"])

def create_eight_node_program(saved_model_path):
    test_prog, expected_nodes, expected_edges = create_str_buf_base_program(saved_model_path)
    test_prog.add_to_first_available_node(STR_BUF, SIBLING_EDGE)
    test_prog.add_to_first_available_node(STR_APP, CHILD_EDGE)
    test_prog.add_to_first_available_node(READ_LINE, CHILD_EDGE)
    test_prog.add_to_first_available_node(STR_APP, SIBLING_EDGE)
    test_prog.add_to_first_available_node(READ_LINE, SIBLING_EDGE)
    test_prog.add_to_first_available_node(STR_BUF, CHILD_EDGE)
    test_prog.update_nodes_and_edges()
    expected_nodes = [START, STR_APP, READ_LINE, STR_BUF, READ_LINE, STR_APP, STR_BUF, STR_BUF]
    expected_edges = [True, True, True, False, False, False, False]

    # assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
    # assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")

    return test_prog, expected_nodes, expected_edges

def create_dbranch(test_prog, parent=None):
    # expected nodes = [DBRANCH, STR_BUF, STR_APP, STOP, READ_LINE, STOP]
    # expected edges = [CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE, SIBLING_EDGE]
    if parent is None:  # NOTE: must ensure parent is valid on your own
        dbranch = test_prog.add_to_first_available_node(DBRANCH, SIBLING_EDGE)
    else:
        dbranch = test_prog.prog.tree_mod.create_and_add_node(DBRANCH, parent, SIBLING_EDGE)
    cond = test_prog.prog.tree_mod.create_and_add_node(STR_BUF, dbranch, CHILD_EDGE)
    then = test_prog.prog.tree_mod.create_and_add_node(STR_APP, cond, CHILD_EDGE)
    test_prog.prog.tree_mod.create_and_add_node(STOP, then, SIBLING_EDGE)
    else_node = test_prog.prog.tree_mod.create_and_add_node(READ_LINE, cond, SIBLING_EDGE)
    test_prog.prog.tree_mod.create_and_add_node(STOP, else_node, SIBLING_EDGE)
    return test_prog, dbranch

def create_dloop(test_prog, parent=None):
    # expected nodes = [DLOOP, READ_LINE, CLOSE, STOP]
    # expected edges = [CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE]
    if parent is None:
        dloop = test_prog.add_to_first_available_node(DLOOP, SIBLING_EDGE)
    else:
        dloop = test_prog.prog.tree_mod.create_and_add_node(DLOOP, parent, SIBLING_EDGE)
    cond = test_prog.prog.tree_mod.create_and_add_node(READ_LINE, dloop, CHILD_EDGE)
    body = test_prog.prog.tree_mod.create_and_add_node(CLOSE, cond, CHILD_EDGE)
    test_prog.prog.tree_mod.create_and_add_node(STOP, body, SIBLING_EDGE)
    return test_prog, dloop

def create_dexcept(test_prog, parent=None):
    # expected nodes = [DEXCEPT, STR_BUF, CLOSE, STOP]
    # expected edges = [CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE]
    if parent is None:
        dexcept = test_prog.add_to_first_available_node(DEXCEPT, SIBLING_EDGE)
    else:
        dexcept = test_prog.prog.tree_mod.create_and_add_node(DEXCEPT, parent, SIBLING_EDGE)
    catch = test_prog.prog.tree_mod.create_and_add_node(STR_BUF, dexcept, CHILD_EDGE)
    try_node = test_prog.prog.tree_mod.create_and_add_node(CLOSE, catch, CHILD_EDGE)
    test_prog.prog.tree_mod.create_and_add_node(STOP, try_node, SIBLING_EDGE)
    return test_prog, dexcept

def create_all_dtypes_program(saved_model_path):
    test_prog, expected_nodes, expected_edges = create_str_buf_base_program(saved_model_path)
    dbranch_parent = test_prog.prog.tree_mod.get_node_in_position(test_prog.prog.curr_prog, 1)
    test_prog, dbranch = create_dbranch(test_prog, parent=dbranch_parent)
    test_prog, dloop = create_dloop(test_prog, parent=dbranch)
    test_prog, dexcept = create_dexcept(test_prog, parent=dloop)
    test_prog.update_nodes_and_edges()
    expected_nodes = [START, STR_BUF, DBRANCH, STR_BUF, STR_APP, STOP, READ_LINE, STOP, DLOOP, READ_LINE, CLOSE,
                      STOP, DEXCEPT, STR_BUF, CLOSE]
    expected_edges = [SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE, SIBLING_EDGE,
                      SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE, SIBLING_EDGE, SIBLING_EDGE, CHILD_EDGE, CHILD_EDGE]
    # self.assertListEqual(test_prog.nodes, expected_nodes, "Nodes must be equal to expected nodes in program.")
    # self.assertListEqual(test_prog.edges, expected_edges, "Edges must be equal to expected nodes in program.")
    return test_prog, expected_nodes, expected_edges


class MCMCProgramWrapper:
    def __init__(self, save_dir, constraints, return_type, formal_params, ordered=True, exclude=None, debug=True, verbose=True):
        # init MCMCProgram
        self.prog = MCMCProgram(save_dir, debug=debug, verbose=verbose)
        self.prog.init_program(constraints, return_type, formal_params, exclude=exclude, ordered=ordered)

        self.constraints = self.prog.constraints
        self.vocab2node = self.prog.config.vocab2node
        self.node2vocab = self.prog.config.node2vocab

        # init nodes, edges and parents
        self.nodes = []
        self.edges = []
        self.parents = []
        self.update_nodes_and_edges()

    def add_to_first_available_node(self, api_name, edge):
        curr_node = self.prog.curr_prog
        stack = []
        while curr_node is not None:
            if edge == SIBLING_EDGE and curr_node.sibling is None and curr_node.api_name != STOP:
                break

            elif edge == CHILD_EDGE and curr_node.child is None and curr_node.api_name != STOP:
                break

            else:
                if curr_node.child is not None:
                    if curr_node.sibling is not None:
                        stack.append(curr_node.sibling)
                        curr_node = curr_node.child
                elif curr_node.sibling is not None:
                    curr_node = curr_node.sibling
                else:
                    if len(stack) > 0:
                        curr_node = stack.pop()
                    else:
                        curr_node = None

        parent = curr_node

        return self.create_and_add_node(api_name, parent, edge)

    def create_and_add_node(self, api_name, parent, edge):
        node = self.prog.tree_mod.create_and_add_node(api_name, parent, edge)
        self.update_nodes_and_edges()
        return node

    def update_nodes_and_edges(self, verbose=False):
        curr_node = self.prog.curr_prog

        stack = []
        nodes = []
        edges = []
        parents = [None]

        pos_counter = 0

        while curr_node is not None:
            nodes.append(curr_node.api_name)

            if verbose:
                self.prog.verbose_node_info(curr_node, pos=pos_counter)

            pos_counter += 1

            if curr_node.api_name != START:
                edges.append(curr_node.parent_edge)
                parents.append(curr_node.parent.api_name)

            if curr_node.child is not None:
                if curr_node.sibling is not None:
                    stack.append(curr_node.sibling)
                curr_node = curr_node.child
            elif curr_node.sibling is not None:
                curr_node = curr_node.sibling
            else:
                if len(stack) > 0:
                    curr_node = stack.pop()
                else:
                    # # remove last DSTOP node
                    # if curr_node.api_name == STOP:
                    #     curr_node.parent.remove_node(curr_node.parent_edge)
                    #     nodes.pop()
                    #     edges.pop()
                    #     parents.pop()
                    curr_node = None

        if verbose:
            print("\n")

        self.nodes = nodes
        self.edges = edges
        self.parents = parents

    def save_summary_logs(self, logs_f):
        self.update_nodes_and_edges()
        nodes, edges, targets = self.prog.tree_mod.get_nodes_edges_targets(self.prog.curr_prog)
        logs_f.write("\n\n\n-----------------------------------------------------------------------------")
        logs_f.write("\nConstraints: " + str(self.prog.constraints))
        logs_f.write("\nNodes:" + str([self.node2vocab[i] for i in nodes]))
        logs_f.write("\nEdges:" + str(edges))
        logs_f.write("\nTargets:" + str([self.node2vocab[i] for i in targets]))
        logs_f.write("\nFormal Parameters:" + str([self.prog.config.num2fp[i] for i in self.prog.fp[0]]))
        logs_f.write("\nReturn Types:" + str([self.prog.config.num2rettype[i] for i in self.prog.ret_type]))
        logs_f.write("\nTotal accepted transformations:" + str(self.prog.accepted))
        logs_f.write("\nTotal rejected transformations:" + str(self.prog.rejected))
        logs_f.write("\nTotal valid transformations:" + str(self.prog.valid))
        logs_f.write("\nTotal invalid transformations:" + str(self.prog.invalid))
        logs_f.write("\nTotal attempted add transforms:" + str(self.prog.Insert.attempted))
        logs_f.write("\nTotal accepted add transforms:" + str(self.prog.Insert.accepted))
        logs_f.write("\nTotal attempted delete transforms:" + str(self.prog.Delete.attempted))
        logs_f.write("\nTotal accepted delete transforms:" + str(self.prog.Delete.accepted))
        logs_f.write("\nTotal attempted swap transforms:" + str(self.prog.Swap.attempted))
        logs_f.write("\nTotal accepted swap transforms:" + str(self.prog.Swap.accepted))
        logs_f.write("\nPosterior Distribution:")

        posterior = {}
        for prog in self.prog.posterior_dist.keys():
            str_prog = [[self.prog.config.node2vocab[i] for i in prog[0]], prog[1],
                        [self.prog.config.node2vocab[i] for i in prog[2]]]
            str_prog = (tuple(str_prog[0]), tuple(str_prog[1]), tuple(str_prog[2]))
            posterior[str_prog] = self.prog.posterior_dist[prog]

            logs_f.write('\n\t' + str(str_prog[0]))
            logs_f.write('\n\t' + str(str_prog[1]))
            logs_f.write('\n\t' + str(str_prog[2]))
            logs_f.write('\n\t' + str(self.prog.posterior_dist[prog]) + '\n')

    def print_summary_logs(self):
        self.update_nodes_and_edges()
        nodes, edges, targets = self.prog.tree_mod.get_nodes_edges_targets(self.prog.curr_prog)
        print("\n", "-------------------LOGS:-------------------")
        print("Constraints:", self.prog.constraints)
        print("Nodes:", [self.node2vocab[i] for i in nodes])
        print("Edges:", edges)
        print("Targets:", [self.node2vocab[i] for i in targets])
        print("Formal Parameters:", [self.prog.config.num2fp[i] for i in self.prog.fp[0]])
        print("Return Types:", [self.prog.config.num2rettype[i] for i in self.prog.ret_type])
        print("Total accepted transformations:", self.prog.accepted)
        print("Total rejected transformations:", self.prog.rejected)
        print("Total valid transformations:", self.prog.valid)
        print("Total invalid transformations:", self.prog.invalid)
        print("Total attempted add transforms:", self.prog.Insert.attempted)
        print("Total accepted add transforms:", self.prog.Insert.accepted)
        print("Total attempted delete transforms:", self.prog.Delete.attempted)
        print("Total accepted delete transforms:", self.prog.Delete.accepted)
        print("Total attempted swap transforms:", self.prog.Swap.attempted)
        print("Total accepted swap transforms:", self.prog.Swap.accepted)
        print("Posterior Distribution:")

        posterior = {}
        for prog in self.prog.posterior_dist.keys():
            str_prog = [[self.prog.config.node2vocab[i] for i in prog[0]], prog[1],
                        [self.prog.config.node2vocab[i] for i in prog[2]]]
            str_prog = (tuple(str_prog[0]), tuple(str_prog[1]), tuple(str_prog[2]))
            posterior[str_prog] = self.prog.posterior_dist[prog]

            print('\t', str_prog[0])
            print('\t', str_prog[1])
            print('\t', str_prog[2])
            print('\t', self.prog.posterior_dist[prog], '\n')



        # print("Total attempted add dnode transforms:", self.prog.AddDnode.attempted)
        # print("Total accepted add dnode transforms:", self.prog.AddDnode.accepted)
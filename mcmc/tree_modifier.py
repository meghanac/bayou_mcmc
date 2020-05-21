import numpy as np

from mcmc.node import START, Node


class TreeModifier:
    def __init__(self, config):
        self.config = config
        self.max_length = self.config.max_length

    def get_node_in_position(self, curr_prog, pos_num):
        """
        Get Node object for node in given position. Position number is determined by DFS, with child edges taking
        precedence over sibling edges.
        :param pos_num: (int) position number of node required. THIS IS ZERO-INDEXED POSITION NUMBER>
        :return: (Node) required node
        """
        assert pos_num < curr_prog.length, "Position number must be less than total length of program"

        num_explored = 0
        stack = []
        curr_node = curr_prog

        while num_explored != pos_num and curr_node is not None:
            # Fail safe
            if num_explored > self.max_length + 1:
                raise ValueError("WARNING: Caught in infinite loop")

            # Update curr_node
            if curr_node.child is not None:
                if curr_node.sibling is not None:
                    stack.append(curr_node.sibling)
                curr_node = curr_node.child
                num_explored += 1
            elif curr_node.sibling is not None:
                curr_node = curr_node.sibling
                num_explored += 1
            else:
                if len(stack) > 0:
                    curr_node = stack.pop()
                    num_explored += 1
                else:
                    raise ValueError('DFS failed to find node in postion: ' + str(pos_num))

        return curr_node

    def get_nodes_position(self, curr_prog, node):
        """
        Given a node, return it's position number determined by DFS, with child taking precedence over sibling.
        :param node: (Node) node whose position is to be determined
        :return: (int) position number of given node
        """
        # Fail safe
        if node is None:
            print("WARNING: node fed into self.get_nodes_position is None")
            return node

        stack = []
        curr_node = curr_prog
        counter = 0

        # DFS
        while curr_node is not None:
            if curr_node == node:
                return counter

            counter += 1

            if counter > self.max_length + 1:
                raise ValueError("WARNING: Caught in infinite loop")
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
                    raise ValueError('DFS failed to find given node: ' + node.api_name)

        return -1

    def get_vector_representation(self, curr_prog):
        """
        Returns vectors of nodes and edges in the current program that can be fed into the model
        :return: (list of ints) nodes, (list of bools) edges
        """
        stack = []
        curr_node = curr_prog
        nodes_dfs = []
        edges_dfs = []

        while curr_node is not None:
            # Add current node to list
            nodes_dfs.append(curr_node.api_num)

            # Update edges list
            if curr_node.api_name != START:
                edges_dfs.append(curr_node.parent_edge)

            # Find next node
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

        # Fill in blank nodes and take only self.max_length number of nodes in the tree
        nodes = np.zeros([1, self.max_length], dtype=np.int32)
        edges = np.zeros([1, self.max_length], dtype=np.bool)
        nodes[0, :len(nodes_dfs)] = nodes_dfs
        edges[0, :len(edges_dfs)] = edges_dfs

        return nodes[0], edges[0]

    def get_node_names_and_edges(self, curr_prog):
        """
        Returns list of apis (nodes) and edges of the current program
        :return: (list of strings) apis of nodes, (list of booleans) edges
        """
        nodes, edges = self.get_vector_representation(curr_prog)
        nodes = [self.config.node2vocab[node] for node in nodes]
        return nodes, edges

    def create_and_add_node(self, api_name, parent, edge):
        """
        Create a new node and add it to the program
        :param api_name: (string) api name of node to be created
        :param parent: (Node) parent of node to be created
        :param edge: (bool- SIBLING_EDGE or CHILD_EDGE) relation of node to be created to parent node
        :return: (Node) node that has been created
        """
        try:
            api_num = self.config.vocab2node[api_name]
            node = Node(api_name, api_num, parent, edge)

            # Update parent according to edge
            if api_name != START:
                parent.add_node(node, edge)

            return node

        except KeyError:  # api does not exist in vocabulary and hence node cannot be made
            print(api_name, " is not in vocabulary. Node was not added.")
            return None

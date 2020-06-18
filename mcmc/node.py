CHILD_EDGE = True
SIBLING_EDGE = False

STOP = 'DStop'
DBRANCH = 'DBranch'
DLOOP = 'DLoop'
DEXCEPT = 'DExcept'
START = 'DSubTree'
DNODES = {START, STOP, DBRANCH, DLOOP, DEXCEPT}
EMPTY = '__delim__'


class Node:
    """
    Class for a single node in a program. Each node represents a single API.
    A program is one or more Nodes, each connected to each other by Sibling or Child Edges.
    There may be at most one sibling and one child node.
    DSubtree is used as the start node of the tree and may only have 1 sibling node representing the first API in
    the program.
    Only DBranch, DLoop and DExcept and their child nodes may have child edges/nodes.
    DStop nodes may not have a sibling or a child node.
    """

    def __init__(self, api_name, api_num, parent, parent_edge):
        """
        :param id: [DEPRECATED] (int) unique id for Node object
        :param api_name: (string) name of the api
        :param api_num: (int) number assigned to api in vocab dictionary. Must be > 0.
        :param parent: (Node) parent node
        :param parent_edge: (bool - SIBLING_EDGE or CHILD_EDGE) represents the edge between this node and parent node
        """
        self.api_name = api_name
        self.api_num = api_num
        self.child = None  # pointer to child node
        self.sibling = None  # pointer to sibling node
        self.parent = None  # pointer to parent node
        # self.id = id
        self.length = 1  # Number of nodes in subtree starting at this node (i.e., how many nodes stem out of this node)
        self.parent_edge = None

        if self.api_name in DNODES:
            self.non_dnode_length = 0
        else:
            self.non_dnode_length = 1

    def update_length(self, length, non_dnode_length, add_or_sub):
        """
        Updates length of this node and recursively updates all node above it so that length of head of the program
        (DSubtree) accounts for all nodes in program
        :param non_dnode_length: length of program excluding all dnodes
        :param length: (int) length to be added to or subtracted from this node
        :param add_or_sub: (string - ust be either 'add' or 'subtract') flag whether to add length or subtract length
        :return:
        """
        # Add length
        if add_or_sub == 'add':
            self.length += length
            self.non_dnode_length += non_dnode_length
        # Subtract length
        else:
            self.length -= length
            self.non_dnode_length -= non_dnode_length
        # Recursively update parent nodes' length
        if self.parent is not None:
            self.parent.update_length(length, non_dnode_length, add_or_sub)

    def add_node(self, node, edge):
        """
        :param node: (Node) node to be added to self. Can be None (i.e., nothing will be added)
        :param edge: (bool- SIBLING_EDGE or CHILD_EDGE) whether node will be self's child or sibling node
        :return:
        """
        if node is not None:
            if edge == SIBLING_EDGE:
                # Remove existing sibling node if there is one
                if self.sibling is not None:
                    self.remove_node(SIBLING_EDGE)
                # Update self
                self.update_length(node.length, node.non_dnode_length, 'add')
                self.sibling = node
                # Update node
                node.parent = self
                node.parent_edge = edge

            elif edge == CHILD_EDGE:
                # Remove existing child node is there is o e
                if self.child is not None:
                    self.remove_node(CHILD_EDGE)
                # Update self
                self.update_length(node.length, node.non_dnode_length, 'add')
                self.child = node
                # Update node
                node.parent = self
                node.parent_edge = edge
            else:
                raise ValueError('edge must be SIBLING_EDGE or CHILD_EDGE')

        return node

    def remove_node(self, edge):
        """
        Removes node specified by edge. Removes the entire subtree that stems from the node to be removed.
        :param edge: (bool - SIBLING_EDGE or CHILD_EDGE)
        :return:
        """
        if edge == SIBLING_EDGE:
            if self.sibling is not None:
                old_sibling = self.sibling
                self.sibling = None
                old_sibling.parent = None
                old_sibling.parent_edge = None
                self.update_length(old_sibling.length, old_sibling.non_dnode_length, 'sub')
                return old_sibling

        elif edge == CHILD_EDGE:
            if self.child is not None:
                old_child = self.child
                self.child = None
                old_child.parent = None
                old_child.parent_edge = None
                self.update_length(old_child.length, old_child.non_dnode_length, 'sub')
                return old_child

        else:
            raise ValueError('edge must but a sibling or child edge')

    def copy(self):
        """
        Deep copy self including entire subtree. Recursively copies sibling and child nodes.
        :return: (Node) deep copy of self
        """
        new_node = Node(self.api_name, self.api_num, None, None)
        if self.sibling is not None:
            new_sibling_node = self.sibling.copy()
            assert new_sibling_node.length == self.sibling.length, "new sib length: " + str(
                new_sibling_node.length) + ", orig sib length: " + str(self.sibling.length)
            new_node.add_node(new_sibling_node, SIBLING_EDGE)
        if self.child is not None:
            new_child_node = self.child.copy()
            assert new_child_node.length == self.child.length
            new_node.add_node(new_child_node, CHILD_EDGE)
        assert self.length == new_node.length, "self length: " + str(self.length) + " new length: " + str(
            new_node.length)
        return new_node

    def change_api(self, new_api_name, new_api_num):
        self.api_name = new_api_name
        self.api_num = new_api_num

    def get_neighbor(self, edge):
        if edge == SIBLING_EDGE:
            return self.sibling
        else:
            return self.child

    def insert_in_between_after_self(self, node, edge):
        neighbors = self.get_neighbor(edge)
        self.remove_node(edge)
        self.add_node(node, edge)
        node.add_node(neighbors, edge)
        return node

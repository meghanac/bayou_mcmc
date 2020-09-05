from numba import jit, cuda

def print_verbose_tree_info(curr_prog):
    curr_node = curr_prog

    stack = []
    pos_counter = 0

    while curr_node is not None:
        verbose_node_info(curr_node, pos=pos_counter)

        pos_counter += 1

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

    print("\n")


def verbose_node_info(node, pos=None):
    node_info = {"api name": node.api_name, "length": node.length, "api num": node.api_num,
                 "parent edge": node.parent_edge}

    if pos is not None:
        node_info["position"] = pos

    if node.parent is not None:
        node_info["parent"] = node.parent.api_name
    else:
        node_info["parent"] = node.parent

        if node.api_name != 'DSubTree':
            print("WARNING: node does not have a parent", node.api_name)

    if node.sibling is not None:
        node_info["sibling"] = node.sibling.api_name

        if node.sibling.parent is None:
            print("WARNING: sibling parent is None for node", node.api_name, "in pos", pos)
            node_info["sibling parent"] = node.sibling.parent
        else:
            node_info["sibling parent"] = node.sibling.parent.api_name

        node_info["sibling parent edge"] = node.sibling.parent_edge
    else:
        node_info["sibling"] = node.sibling

    if node.child is not None:
        node_info["child"] = node.child.api_name

        if node.child.parent is None:
            print("WARNING: child parent is None for node", node.api_name, "in pos", pos)
            node_info["child parent"] = node.child.parent
        else:
            node_info["child parent"] = node.child.parent.api_name

        node_info["child parent edge"] = node.child.parent_edge

    print(node_info)

    return node_info

# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from graphviz import Digraph
from ast_helper.ast_traverser import AstTraverser
import random

def visualize_from_ast_head_node(ast_head_node):
    ast_traverser = AstTraverser()
    path = ast_traverser.depth_first_search(ast_head_node)
    dot = visualize_from_ast_path(path, 1.0)
    return


def visualize_from_ast_path(ast_path, prob, save_path='temp.gv'):
    dot = Digraph(comment='Program AST', format='eps')
    dot.node(str(prob), str(prob)[:6])
    for dfs_id, item in enumerate(ast_path):
        node_value , parent_id , edge_type = item
        dot.node( str(dfs_id) , node_value )
        label = 'child' if edge_type else 'sibling'
        label += " / " + str(dfs_id)
        if dfs_id > 0:
            dot.edge( str(parent_id) , str(dfs_id), label=label, constraint='true', direction='LR')

    dot.render(save_path)
    return dot
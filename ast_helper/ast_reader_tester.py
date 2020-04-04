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


from ast_helper.ast_reader import AstReader
from graphviz import Digraph


def plot_path(i, path, prob):
    dot = Digraph(comment='Program AST', format='eps')
    dot.node(str(prob), str(prob)[:6])
    for dfs_id, item in enumerate(path):
        node_value, parent_id, edge_type = item
        dot.node(str(dfs_id), node_value)
        label = 'child' if edge_type else 'sibling'
        label += " / " + str(dfs_id)
        if dfs_id > 0:
            dot.edge(str(parent_id), str(dfs_id), label=label, constraint='true', direction='LR')

    stri = colnum_string(i)
    dot.render('plots/' + 'program-ast-' + stri + '.gv')
    return dot

def colnum_string(n):
    n = n + 26 * 26 * 26 + 26 * 26 + 26 + 1
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string

js = {
    "ast": {
        "node": "DSubTree",
        "_nodes": [
            {
                "node": "DBranch",
                "_cond": [
                    {
                        "node": "DAPICall",
                        "_call": "java.nio.Buffer.capacity()"
                    }
                ],
                "_else": [
                    {
                        "node": "DAPICall",
                        "_call": "java.nio.ByteBuffer.allocate(int)"
                    },
                    {
                        "node": "DAPICall",
                        "_call": "java.nio.ByteBuffer.allocate(int)"
                    }
                ],
                "_then": [
                    {
                        "node": "DAPICall",
                        "_call": "java.nio.Buffer.clear()"
                    }
                ]
            },
            {
                "node": "DAPICall",
                "_call": "java.nio.ByteBuffer.array()"
            },
            {
                "node": "DAPICall",
                "_call": "java.nio.Buffer.position()"
            },
            {
                "node": "DAPICall",
                "_call": "java.lang.System.arraycopy(java.lang.Object,int,java.lang.Object,int,int)"
            },
            {
                "node": "DAPICall",
                "_call": "java.lang.System.arraycopy(java.lang.Object,int,java.lang.Object,int,int)"
            },
            {
                "node": "DAPICall",
                "_call": "java.nio.Buffer.limit(int)"
            }
        ]
    }
}

js1 = {
    "ast": {
        "node": "DSubTree",
        "_nodes": [
            {
                "node": "DAPICall",
                "_call": "java.io.File.delete()"
            },
            {
                "node": "DLoop",
                "_cond": [
                    {
                        "node": "DAPICall",
                        "_call": "java.util.Iterator<Tau_E>.hasNext()"
                    }
                ],
                "_body": [
                    {
                        "node": "DAPICall",
                        "_call": "java.util.Iterator<Tau_E>.next()"
                    }
                ]
            }
        ]
    }
}

js2 = {
    "ast": {
        "node": "DSubTree",
        "_nodes": [
            {
                "node": "DExcept",
                "_catch": [
                    {
                        "node": "DAPICall",
                        "_call": "java.lang.Throwable.getMessage()"
                    }
                ],
                "_try": [
                    {
                        "node": "DAPICall",
                        "_call": "java.lang.Class<Tau_T>.getResource(java.lang.String)"
                    },
                    {
                        "node": "DAPICall",
                        "_call": "java.net.URL.openStream()"
                    }
                ]
            }
        ]
    }
}

js3 = {
    "ast": {
        "_nodes": [
            {
                "_try": [
                    {
                        "_throws": [
                            "java.security.NoSuchAlgorithmException"
                        ],
                        "_call": "java.security.SecureRandom.getInstance(java.lang.String)",
                        "node": "DAPICall",
                        "_returns": "java.security.SecureRandom"
                    }
                ],
                "_catch": [
                    {
                        "_throws": [],
                        "_call": "java.lang.Throwable.printStackTrace()",
                        "node": "DAPICall",
                        "_returns": "void"
                    }
                ],
                "node": "DExcept"
            }
        ],
        "node": "DSubTree"
    }
}

js4 = {
    "ast": {
        "_nodes": [
            {
                "_cond": [],
                "node": "DBranch",
                "_then": [
                    {
                        "_throws": [
                            "java.awt.HeadlessException"
                        ],
                        "_call": "javax.swing.JOptionPane.showMessageDialog(java.awt.Component,java.lang.Object)",
                        "node": "DAPICall",
                        "_returns": "void"
                    }
                ],
                "_else": [
                    {
                        "_throws": [
                            "java.awt.HeadlessException"
                        ],
                        "_call": "javax.swing.JOptionPane.showMessageDialog(java.awt.Component,java.lang.Object)",
                        "node": "DAPICall",
                        "_returns": "void"
                    }
                ]
            },
            {
                "_cond": [],
                "node": "DBranch",
                "_then": [
                    {
                        "_throws": [
                            "java.awt.HeadlessException"
                        ],
                        "_call": "javax.swing.JOptionPane.showMessageDialog(java.awt.Component,java.lang.Object)",
                        "node": "DAPICall",
                        "_returns": "void"
                    }
                ],
                "_else": [
                    {
                        "_throws": [
                            "java.awt.HeadlessException"
                        ],
                        "_call": "javax.swing.JOptionPane.showMessageDialog(java.awt.Component,java.lang.Object)",
                        "node": "DAPICall",
                        "_returns": "void"
                    }
                ]
            }
        ],
        "node": "DSubTree"
    }
}

if __name__ == "__main__":
    for i, _js in enumerate([js, js1, js2, js3, js4]):
        ast = AstReader().get_ast_from_json(_js['ast']['_nodes'])
        # path = ast.depth_first_search
        # dot = plot_path(i, path, 0.0)
        # print(path)

        curr_node = ast
        stack = []
        nodes = []
        edges = []
        parents = []
        all = []

        while curr_node is not None:
            if curr_node.val != 'DSubTree':
                nodes.append(curr_node.val)

            if curr_node.child is not None:
                if curr_node.sibling is not None:
                    stack.append((curr_node.sibling, False, curr_node.val))
                parent = curr_node.val
                parents.append(parent)
                curr_node = curr_node.child
                edges.append(True)
                all.append((parent, True, curr_node.val))
            elif curr_node.sibling is not None:
                parent = curr_node.val
                parents.append(parent)
                curr_node = curr_node.sibling
                edges.append(False)
                all.append((parent, False, curr_node.val))
            else:
                if len(stack) > 0:
                    curr_node, edge, parent = stack.pop()
                    edges.append(edge)
                    parents.append(parent)
                    all.append((parent, edge, curr_node.val))
                else:
                    curr_node = None

        # print(nodes)
        # print(edges)
        # print(parents)
        print(all)
        print("")


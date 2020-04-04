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

from ast_helper.node import CHILD_EDGE, SIBLING_EDGE

MAX_LOOP_NUM = 3
MAX_BRANCHING_NUM = 3


class TooLongLoopingException(Exception):
    pass


class TooLongBranchingException(Exception):
    pass


class AstTraverser:

    def check_nested_branch(self, head):
        head = head
        count = 0
        while head != None:
            if head.val == 'DBranch':
                count_Else = self.check_nested_branch(head.child.child)  # else
                count_Then = self.check_nested_branch(head.child.sibling)  # then
                count = 1 + max(count_Then, count_Else)
                if count > MAX_BRANCHING_NUM:
                    raise TooLongBranchingException
            head = head.sibling
        return count

    def check_nested_loop(self, head):
        head = head
        count = 0
        while head != None:
            if head.val == 'DLoop':
                count = 1 + self.check_nested_loop(head.child.child)

                if count > MAX_LOOP_NUM:
                    raise TooLongLoopingException
            head = head.sibling
        return count

    def depth_first_search(self, head):

        buffer = []
        stack = []
        dfs_id = None
        parent_id = 0
        if head is not None:
            stack.append((head, parent_id, SIBLING_EDGE))
            dfs_id = 0

        while len(stack) > 0:
            item_triple = stack.pop()
            item = item_triple[0]
            parent_id = item_triple[1]
            edge_type = item_triple[2]

            buffer.append((item.val, parent_id, edge_type))

            if item.sibling is not None:
                stack.append((item.sibling, dfs_id, SIBLING_EDGE))

            if item.child is not None:
                stack.append((item.child, dfs_id, CHILD_EDGE))

            dfs_id += 1

        return buffer

    def api_extract(self, head):
        buffer = self.depth_first_search(head)
        return [b[0] for b in buffer]

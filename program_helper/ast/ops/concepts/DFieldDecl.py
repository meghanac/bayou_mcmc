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

from program_helper.ast.ops import DVarDecl


# Field is indistinguishable from DVarCall
class DFieldDecl(DVarDecl):
    def __init__(self, node_js, symtab, child=None, sibling=None):
        super().__init__(node_js, symtab, child, sibling)
        self.type = "DFieldDecl"

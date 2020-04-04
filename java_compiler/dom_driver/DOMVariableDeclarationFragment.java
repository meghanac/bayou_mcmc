/*
Copyright 2017 Rice University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package java_compiler.dom_driver;

import java_compiler.dsl.DSubTree;
import java_compiler.dsl.DVarCall;
import java_compiler.dsl.DASTNode;
import org.eclipse.jdt.core.dom.VariableDeclarationFragment;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.IVariableBinding;


public class DOMVariableDeclarationFragment implements Handler {

    final VariableDeclarationFragment fragment;

    public DOMVariableDeclarationFragment(VariableDeclarationFragment fragment) {
        ASTNode parent= fragment.getParent();
        this.fragment = fragment;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();
        IVariableBinding ivb = fragment.resolveBinding();
        if (ivb != null)
            tree.addNode(new DVarCall(ivb));

        DSubTree Tinit = new DOMExpression(fragment.getInitializer()).handle();
        tree.addNodes(Tinit.getNodes());

        return tree;
    }
}

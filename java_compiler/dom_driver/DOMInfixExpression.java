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
import org.eclipse.jdt.core.dom.InfixExpression;

public class DOMInfixExpression implements Handler {

    final InfixExpression expr;

    public DOMInfixExpression(InfixExpression expr) {
        this.expr = expr;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();

        DSubTree Tleft = new DOMExpression(expr.getLeftOperand()).handle();
        DSubTree Tright = new DOMExpression(expr.getRightOperand()).handle();

        tree.addNodes(Tleft.getNodes());
        tree.addNodes(Tright.getNodes());

        return tree;
    }
}

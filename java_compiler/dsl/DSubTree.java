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
package java_compiler.dsl;

import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class DSubTree extends DASTNode {

    String node = "DSubTree";
    List<DASTNode> _nodes;

    public DSubTree() {
        _nodes = new ArrayList<>();
        this.node = "DSubTree";
    }

    public DSubTree(List<DASTNode> _nodes) {
        this._nodes = _nodes;
        this.node = "DSubTree";
    }

    public void addNode(DASTNode node) {
        _nodes.add(node);
    }

    public void addNodes(List<DASTNode> otherNodes) {
        _nodes.addAll(otherNodes);
    }

    public boolean isValid() {
        return !_nodes.isEmpty();
    }

    @Override
    public void updateSequences(List<Sequence> soFar, int max, int max_length) throws TooManySequencesException, TooLongSequenceException {
        if (soFar.size() >= max)
            throw new TooManySequencesException();
        for (DASTNode node : _nodes)
            node.updateSequences(soFar, max, max_length);
    }

    public List<DAPICall> getNodesAsCalls() {
        List<DAPICall> calls = new ArrayList<>();
        for (DASTNode node : _nodes) {
            assert node instanceof DAPICall : "invalid branch condition";
            calls.add((DAPICall) node);
        }
        return calls;
    }

    public List<DASTNode> getNodes() {
        return _nodes;
    }

    @Override
    public int numStatements() {
        int num = 0;
        for (DASTNode node : _nodes)
            num += node.numStatements();
        return num;
    }

    @Override
    public int numLoops() {
        int num = 0;
        for (DASTNode node : _nodes)
            num += node.numLoops();
        return num;
    }

    @Override
    public int numBranches() {
        int num = 0;
        for (DASTNode node : _nodes)
            num += node.numBranches();
        return num;
    }

    @Override
    public int numExcepts() {
        int num = 0;
        for (DASTNode node : _nodes)
            num += node.numExcepts();
        return num;
    }

    @Override
    public Set<DAPICall> bagOfAPICalls() {
        Set<DAPICall> bag = new HashSet<>();
        for (DASTNode node : _nodes)
            bag.addAll(node.bagOfAPICalls());
        return bag;
    }

    @Override
    public Set<Class> exceptionsThrown() {
        Set<Class> ex = new HashSet<>();
        for (DASTNode n : _nodes)
            ex.addAll(n.exceptionsThrown());
        return ex;
    }

    @Override
    public Set<Class> exceptionsThrown(Set<String> eliminatedVars) {
	Set<Class> ex = new HashSet<>();
	for (DASTNode n : _nodes)
	    ex.addAll(n.exceptionsThrown(eliminatedVars));
	return ex;
    }

    public void cleanupCatchClauses(Set<String> eliminatedVars) {
	for (DASTNode n : _nodes) {
	    if (n instanceof DExcept)
		((DExcept)n).cleanupCatchClauses(eliminatedVars);
	}
    }
    
    @Override
    public boolean equals(Object o) {
        if (o == null || ! (o instanceof DSubTree))
            return false;
        DSubTree tree = (DSubTree) o;
        return _nodes.equals(tree.getNodes());
    }

    @Override
    public int hashCode() {
        return _nodes.hashCode();
    }

    @Override
    public String toString() {
        List<String> _nodesStr = _nodes.stream().map(node -> node.toString()).collect(Collectors.toList());
        return String.join("\n", _nodesStr);
    }


}

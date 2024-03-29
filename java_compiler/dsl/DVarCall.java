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

import java.lang.reflect.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DVarCall extends DASTNode
{

    public class InvalidAPICallException extends Exception {}

    String node = "DVarCall";
    String _call;
    int _id;
    //List<String> _throws;
    String _returns;
    transient String retVarName = "";

    /* CAUTION: This field is only available during AST generation */
    transient IVariableBinding variableBinding;
    /* CAUTION: These fields are only available during synthesis (after synthesize(...) is called) */
    transient Method method;
    transient Constructor constructor;

    /* TODO: Add refinement types (predicates) here */

    public DVarCall() {
        //this._call = "";
        this.node = "DVarCall";
    }

    public DVarCall(IVariableBinding variableBinding) {
        this.variableBinding = variableBinding;
        //System.out.println(variableBinding);
        this._id = variableBinding.getVariableId() ; //getClassName() + "." + getSignature();
        //this._throws = new ArrayList<>();
        //for (ITypeBinding exception : methodBinding.getExceptionTypes())
        //    _throws.add(getTypeName(exception, exception.getQualifiedName()));
        this._returns = getTypeName(variableBinding.getType(),
                                            variableBinding.getType().getQualifiedName());
        this.node = "DVarCall";
    }

    @Override
    public void updateSequences(List<Sequence> soFar, int max, int max_length) throws TooManySequencesException, TooLongSequenceException {
        if (soFar.size() >= max)
            throw new TooManySequencesException();
        for (Sequence sequence : soFar) {
            sequence.addCall(_call);
            if (sequence.getCalls().size() > max_length)
                throw new TooLongSequenceException();
        }
    }

    //private String getClassName() throws InvalidAPICallException {
    //    ITypeBinding cls = methodBinding.getDeclaringClass();
    //    String className = cls.getQualifiedName();
    //    if (cls.isGenericType())
    //        className += "<" + String.join(",", Arrays.stream(cls.getTypeParameters()).map(
    //                t -> getTypeName(t, t.getName())
    //        ).collect(Collectors.toList())) + ">";
    //    if (className.equals(""))
    //        throw new InvalidAPICallException();
    //    return className;
    //}

    //private String getSignature() throws InvalidAPICallException {
    //    Stream<String> types = Arrays.stream(methodBinding.getParameterTypes()).map(
    //            t -> getTypeName(t, t.getQualifiedName()));
    //    if (methodBinding.getName().equals(""))
    //        throw new InvalidAPICallException();
    //    return methodBinding.getName() + "(" + String.join(",", types.collect(Collectors.toCollection(ArrayList::new))) + ")";
    //}

    private String getTypeName(ITypeBinding binding, String name) {
        return (binding.isTypeVariable()? "Tau_" : "") + name;
    }

    public void setNotPredicate() {
        this._call = "$NOT$" + this._call;
    }

    @Override
    public int numStatements() {
        return 1;
    }

    @Override
    public int numLoops() {
        return 0;
    }

    @Override
    public int numBranches() {
        return 0;
    }

    @Override
    public int numExcepts() {
        return 0;
    }

    @Override
    public Set<DAPICall> bagOfAPICalls() {
        Set<DAPICall> bag = new HashSet<>();
        //bag.add(this);
        return bag;
    }

    @Override
    public Set<Class> exceptionsThrown() {
        if (constructor != null)
            return new HashSet<>(Arrays.asList(constructor.getExceptionTypes()));
        else
            return new HashSet<>(Arrays.asList(method.getExceptionTypes()));
    }

    @Override
    public Set<Class> exceptionsThrown(Set<String> eliminatedVars) {
        if (!eliminatedVars.contains(this.retVarName))
            return this.exceptionsThrown();
        else
            return new HashSet<>();
    }

    public String getRetVarName() {
        return this.retVarName;
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || ! (o instanceof DAPICall))
            return false;
        DAPICall apiCall = (DAPICall) o;
        return _call.equals(apiCall._call);
    }

    @Override
    public int hashCode() {
        return _call.hashCode();
    }

    @Override
    public String toString() {
        return _call;
    }

}

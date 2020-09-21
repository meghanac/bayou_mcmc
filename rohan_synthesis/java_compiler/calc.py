from jpype import startJVM, shutdownJVM, java, addClassPath, JClass, JInt
startJVM(convertStrings=False)
import jpype.imports
import subprocess

class java_compiler:

	def __init__(self): 
		try:
		    subprocess.run(["javac", "JavaCompiler.java"])    
		    self.comp = JClass('JavaCompiler')
		 
		except Exception as err:
		    print("Exception: {}".format(err))

	def is_assignable_from(self, type1, type2):
		res = self.comp.checker(type1, type2)
		print("Is {} assignable from {} ? :: {}".format(type2, type1, res))

	def close(self):
		subprocess.run(["rm", "JavaCompiler.class"])


if __name__ == "__main__":
	jc = java_compiler()
	res = jc.is_assignable_from("java.util.List", "java.util.ArrayList")
	res = jc.is_assignable_from("java.util.List", "java.awt.Rectangle")
	res = jc.is_assignable_from("java.util.ArrayList", "java.util.List")
	res = jc.is_assignable_from("java.util.ArrayList", "java.util.ArrayList")
	res = jc.is_assignable_from("java.nio.Buffer", "java.nio.ByteBuffer")
	res = jc.is_assignable_from("java.nio.Buffer", "java.nio.CharBuffer")
	res = jc.is_assignable_from("java.awt.Button", "java.awt.Button")
	res = jc.is_assignable_from("java.lang.Object", "java.awt.Button")
	jc.close()


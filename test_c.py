import cppyy
import numpy as np
cppyy.include('cc_examples_2.cpp')


s = cppyy.gbl.Room()
#print(s)
#numpy_array = np.empty(32000, np.float64)
#s.get_numpy_array(numpy_array.data, numpy_array.size)
#print(numpy_array[:20])



s.length = 42.5;
s.breadth = 30.8;
s.height = 19.2;

print(s.calculateArea())
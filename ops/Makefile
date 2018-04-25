# Copyright (c) 2018 Uber Technologies, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

TF_INC := $(shell python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

all: zero_out.so fast_walsh_hadamard.so fast_walsh_hadamard.cu.o

zero_out.so: zero_out.cc
	g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I ${TF_INC} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

fast_walsh_hadamard.so: fast_walsh_hadamard.cc fast_walsh_hadamard.cu.o
	g++ -std=c++11 -shared fast_walsh_hadamard.cc fast_walsh_hadamard.cu.o -o fast_walsh_hadamard.so -fPIC -I ${TF_INC} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -lcudart

fast_walsh_hadamard.cu.o: fast_walsh_hadamard.cu.cc
	nvcc -std=c++11 -c -o fast_walsh_hadamard.cu.o fast_walsh_hadamard.cu.cc -I ${TF_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED --gpu-architecture=compute_53 --gpu-code="sm_53,compute_53"

clean:
	rm -f zero_out.so fast_walsh_hadamard.so fast_walsh_hadamard.cu.o

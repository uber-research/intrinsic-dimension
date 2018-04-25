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

import tensorflow as tf
from fwh import fast_walsh_hadamard as c_fast_walsh_hadamard
import numpy as np
import time

class FastWalshHadamardTest(tf.test.TestCase):
    def testFastWalshHadamard(self):
        with self.test_session():
            for i in range(20, 30):
                V = np.random.RandomState(123).randn(2 ** i).astype(np.float32)
                V = tf.constant(V, tf.float32)

                with tf.device('/cpu'):
                    cpu_out = c_fast_walsh_hadamard(V)
                with tf.device('/gpu'):
                    gpu_out = c_fast_walsh_hadamard(V)

                a_start = time.time()
                a = cpu_out.eval();
                a_end = time.time()
                b = gpu_out.eval();
                b_end = time.time()
                print('Size: 2**{} = {} CPU: {} GPU: {} Speedup: {}'.format(i, 2 ** i, a_end-a_start, b_end-a_end, (a_end-a_start)/(b_end-a_end)))
                self.assertAllClose(a, b, rtol=1e-02, atol=1e-02)

if __name__ == "__main__":
    tf.test.main()

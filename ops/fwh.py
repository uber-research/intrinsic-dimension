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

import os
import tensorflow as tf
from tensorflow.python.framework import ops

fwh_so = os.path.join(os.path.dirname(__file__), 'fast_walsh_hadamard.so')

try:
    fast_walsh_hadamard_module = tf.load_op_library(fwh_so)
except tf.errors.NotFoundError:
    print '\n\nError: could not find compiled fast_walsh_hadamard.so file. Tried loading from this location:\n\n    %s\n\nRun "make all" in lab/ops first.\n\n' % fwh_so
    raise

fast_walsh_hadamard = fast_walsh_hadamard_module.fast_walsh_hadamard


@ops.RegisterGradient("FastWalshHadamard")
def _fast_walsh_hadamard_grad(op, grad):
    '''The gradients for `fast_walsh_hadamard`.
  
    Args:
      op: The `fast_walsh_hadamard` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `fast_walsh_hadamard` op.
  
    Returns:
      Gradients with respect to the input of `fast_walsh_hadamard`.
    '''

    gg = fast_walsh_hadamard(grad)
    return [gg]

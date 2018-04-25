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

from keras.layers import Input
from keras.engine.topology import Container

from .util import full_static_shape



class LazyContainer(Container):
    '''Like Container. But lazy.'''
    
    def __init__(self, container_function, use_method_disposable=True):
        self._container_function = container_function
        self._lazy_has_run = False
        self.use_method_disposable = use_method_disposable
        # Delay rest of construction until first call
        
    def __call__(self, x, mask=None):
        if not self._lazy_has_run:
            # Make short-lived Input Layers for each x this was called with
            # TODO: handle tuple or list x
            x_shape = full_static_shape(x)   # Uses var._keras_shape or var.get_shape()
            if self.use_method_disposable:
                inp_layer = Input(batch_shape=x_shape,
                                  dtype=x.dtype,
                                  name='tmp_input_from__%s' % x.name.replace('/','_').replace(':','_'))
            else:
                print 'Warning: using non-disposable approach. May not work yet.'
                inp_layer = Input(tensor=x,
                                  batch_shape=x_shape,
                                  dtype=x.dtype, name='real_input_from__%s' % x.name.replace('/','_').replace(':','_'))

            # Call function of inputs to get output tensors
            outputs = self._container_function(inp_layer)

            # Initialize entire Container object here (finally)
            super(LazyContainer, self).__init__(inp_layer, outputs)
            
            self._lazy_has_run = True
            if not self.use_method_disposable:
                return outputs

        # Non-disposable mode: actually call the Container only the *second* and later times
        # Disposable mode: call the Container now
        ret = super(LazyContainer, self).__call__(x, mask=mask)
        return ret

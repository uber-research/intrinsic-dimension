// Copyright (c) 2018 Uber Technologies, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include <iostream>
#include "fast_walsh_hadamard.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;
using namespace std;

#define EIGEN_USE_GPU


template <typename T>
__global__ void FastWalshHadamardKernel(const int stride, const T* in, T* out) {
    const auto idx = (threadIdx.x + blockIdx.x * blockDim.x);
    const auto elemIdx = (idx / stride ) * (2 * stride) + (idx % stride);
    const auto tmp = in[elemIdx], tmp2 = in[elemIdx + stride];
    out[elemIdx] = tmp + tmp2;
    out[elemIdx + stride] = tmp - tmp2;
}

template <typename T>
__global__ void FastWalshHadamardSubKernel(const T scalar, T* out) {
    const auto idx = (threadIdx.x + blockIdx.x * blockDim.x);
    out[idx] *= scalar;
}


template <typename T>
void FastWalshHadamardKernelLauncher(const int NN, const int halfLL, const T* in, T* out) {
    // Apply Unnormalized Fast Walsh Hadamard transform
    int stride = halfLL;
    float normalizer = 1.0;
    float sqrt2inv = 0.70710678118654746;
    while (stride >= 1) {
      if(stride == halfLL)
          FastWalshHadamardKernel<T><<<max(1, halfLL/256), min(256, halfLL)>>>(stride, in, out);
      else
          FastWalshHadamardKernel<T><<<max(1, halfLL/256), min(256, halfLL)>>>(stride, out, out);

      stride /= 2;
      normalizer *= sqrt2inv;
    }
    FastWalshHadamardSubKernel<T><<<max(1, NN/256), min(256, NN)>>>(normalizer, out);
}

template void FastWalshHadamardKernelLauncher<float>(const int NN, const int halfLL, const float* in, float* out);

#endif

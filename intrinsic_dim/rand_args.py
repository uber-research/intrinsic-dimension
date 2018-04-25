#! /usr/bin/env python

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


import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate command line args corresponding to randomly sampled network architectures',
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog)
    )
    parser.add_argument('--lenet', action='store_true', help='Output args for lenet models')
    parser.add_argument('--l2', action='store_true', help='Output l2 as well')
    parser.add_argument('--seed', '-s', type=int, default=None, help='Seed for random number generator. If not specified, use a randomly sampled seed for each run')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    argstrs = []

    if args.lenet:
        argstrs.append('--c1 %s' % np.random.choice([1, 2, 3, 4, 6, 10]))
        argstrs.append('--c2 %s' % np.random.choice([1, 2, 3, 4, 6, 10, 16]))
        argstrs.append('--d1 %s' % np.random.choice([2, 3, 5, 10, 20, 50, 100, 120]))
        argstrs.append('--d2 %s' % np.random.choice([2, 3, 5, 10, 20, 50, 84]))
    else:
        depth_choices = [1, 2, 3, 4, 5]
        argstrs.append('--depth %s' % np.random.choice(depth_choices))

        width_choices = [2, 3, 5, 8, 10, 15, 20, 25]
        argstrs.append('--width %s' % np.random.choice(width_choices))

    if args.l2:
        argstrs.append('--l2 %s' % np.random.choice([0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]))

    print ' '.join(argstrs)


if __name__ == '__main__':
    main()

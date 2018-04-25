#!/bin/bash

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

# for depth in {0,1,2,3,4,5}
# width in {10,50,100,200,400,600,800,1000}
# for dim in {0,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000}
# {50,200,500,1000}
# {1500,2000}
# {0,1,2,3,4,5,6,7,8,9,10}
for actv in {'selu',}
	do
	for depth in {0,1,2,3,4,5,6,7,8,9,10}
	do
		for width in {25,50}
		do
			for dim in {0,50,200,500,1000}
			do
				if [ "$dim" = 0 ]; then
					echo dir_"$dim"_"$depth"_"$width"
					CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -r fnn_mnist_sparse_"$actv"_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width --actv $actv &
				else	
					echo lrb_"$dim"_"$depth"_"$width"
					CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -r fnn_mnist_sparse_"$actv"_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistfc --depth $depth --width $width --actv $actv &
				fi
			done
		done
	done
done			


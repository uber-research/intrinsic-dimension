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

# # test mnist MLP lenet model
# for dim in {100,0}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet_dir
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet --fastfoodproj
# 	fi
# done

for dim in {1750,2000,2250,2500}
do
	if [ "$dim" = 0 ]; then
		echo dir_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/results_mnist_MLP_LeNet3 -r LeNet_"$dim" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet_dir &
	else	
		echo lrb_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/results_mnist_MLP_LeNet3 -r LeNet_"$dim" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet --fastfoodproj &
	fi
done

# # test cifar MLP lenet model
# for dim in {100,0}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet_dir
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet --fastfoodproj
# 	fi
# done


for dim in {17500,20000,25000}
do
	if [ "$dim" = 0 ]; then
		echo dir_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/results_cifar_MLP_LeNet3 -r LeNet_"$dim" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet_dir &
	else	
		echo lrb_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/results_cifar_MLP_LeNet3 -r LeNet_"$dim" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet --fastfoodproj &
	fi
done

# # test mnist untied lenet model
# for dim in {100,0}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistUntiedlenet_dir
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistUntiedlenet --fastfoodproj
# 	fi
# done


# # test cifar untied lenet model
# for dim in {3000,0}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarUntiedlenet_dir
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarUntiedlenet --fastfoodproj
# 	fi
# done



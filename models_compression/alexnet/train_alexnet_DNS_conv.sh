#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models_compression/alexnet/alexnet_solver_DNS_conv.prototxt \
    --weights=models_compression/alexnet/DNS_alexnet_train_iter_112500.caffemodel \
    --gpu=all $@ 
    

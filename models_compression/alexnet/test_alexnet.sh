#!/usr/bin/env sh
set -e

./build/tools/caffe test \
    --model=models_compression/alexnet/train_val.prototxt \
    --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel  $@ 
    

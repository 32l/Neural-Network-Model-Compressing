#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models_compression/alexnet/alexnet_solver_DNS.prototxt \
    --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel $@ 
    

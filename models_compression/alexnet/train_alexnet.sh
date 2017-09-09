#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models_compression/alexnet/alexnet_solver.prototxt $@ 
    

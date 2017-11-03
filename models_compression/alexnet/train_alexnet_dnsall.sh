#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=modelscomtest/alexnet/alexnet_solver_dnsall.prototxt \
    --weights=modelscomtest/alexnet/DNS_alexnet_train_iter_112500.caffemodel $@
    

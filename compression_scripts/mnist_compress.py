import os, sys, numpy as np

sys.path.append(os.getenv("CAFFE_ROOT")+"/python")
sys.path.append(os.getenv("CAFFE_ROOT")+"/python/caffe/proto")
import caffe
import caffe_pb2 as cp2

model_path = "/home/users/xieqikai/myGitRepo/caffe-0828-compress/examples/mnist/lenet_DNS_iter_10000.caffemodel"

model = cp2.NetParameter()

with open(model_path,'rb') as f:
    model.ParseFromString(f.read())

for layer in model.layer:
    if "conv" in layer.name or "ip" in layer.name or "fc" in layer.name:
        

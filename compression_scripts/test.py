
import os, sys
import numpy as np

try:
    caffe_root = os.environ["CAFFE_ROOT"]
except KeyError:
    print "Set system variable CAFFE_ROOT before running the script!"
    sys.exit(-1)

sys.path.append(caffe_root+"/python")
# sys.path.append(os.getenv("CAFFE_ROOT")+"/python/caffe/proto")
import caffe

os.chdir("/home/users/xieqikai/myGitRepo/caffe-0828-compress")
print "current dir: %s"%(os.getcwd())
prototxt = "examples/mnist/lenet_train_test_DNS.prototxt"
if not os.path.exists(prototxt):
    print "Error: %s does not exist!"%(prototxt)
    sys.exit()

caffe.set_mode_cpu()

net = caffe.Net(prototxt, caffe.TEST)

print





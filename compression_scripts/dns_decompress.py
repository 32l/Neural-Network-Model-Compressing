'''
reconstruct the model from compressed model.
'''

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

help = '''
Usage:
    dns_decompress.py <model.prototxt> <compressed_model> <target.caffemodel>
'''

if len(sys.argv) != 4:
    print help
    sys.exit(-1)
else:
    prototxt = sys.argv[1]  # 
    compressed_model = sys.argv[2]   # 
    target = sys.argv[3]    # 

if not os.path.exists(prototxt):
    print "Error: %s does not exist!"%(prototxt)
    sys.exit()
elif not os.path.exists(compressed_model):
    print "Error: %s does not exist!"%(compressed_model)
    sys.exit()

caffe.set_mode_cpu()

net = caffe.Net(prototxt, caffe.TEST)
layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
f_comp = open(compressed_model, 'rb')

def comp_to_normal(weights, spm_s, idx_s):
    data = np.zeros(weights.size, dtype=np.float32)
    data[idx_s] = spm_s
    data = data.reshape(weights.shape)
    np.copyto(weights, data)


for layer in layers:
    # sparse matrix size
    spm_size = np.fromfile(f_comp, dtype=np.int32, count=2)
    # sparse matrix of weights and indices
    w_spm_stream = np.fromfile(f_comp, dtype=np.float32, count=spm_size[0])
    w_idx_stream = np.fromfile(f_comp, dtype=np.int32, count=spm_size[0])
    print w_idx_stream
    # comp_to_normal(net.params[layer][0].data, w_spm_stream, w_idx_stream)

    # sparse matrix of bias and indices
    b_spm_stream = np.fromfile(f_comp, dtype=np.float32, count=spm_size[1])
    b_idx_stream = np.fromfile(f_comp, dtype=np.int32, count=spm_size[1])
    comp_to_normal(net.params[layer][1].data, b_spm_stream, b_idx_stream)

f_comp.close()
net.save(target)
print "Model has been decompressed as %s"%(target)



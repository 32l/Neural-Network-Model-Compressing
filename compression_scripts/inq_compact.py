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
    inq_compact.py <source_inq_net.prototxt> <source_inq.caffemodel> <output_filename> 

Converting the Caffe-output INQ model (twice the size of a normal model) to a 4-bit format, all parameters stored, including zeros.

Code book example: 
4-bit code:          0000  0001  0010  0011  0100  0101  0110  0111
corresponding param: -2^-9 -2^-8 -2^-7 -2^-6 -2^-5 -2^-4 -2^-3  0.0

4bit code:           1000  1001  1010  1011  1100  1101  1110  1111
corresponding param:  2^-9  2^-8  2^-7  2^-6  2^-5  2^-4  2^-3  N/A
'''
# corresponding to num_quantum_values in caffe.prototxt
num_kept_value = 7

if len(sys.argv) != 4:
    print help
    sys.exit(-1)
else:
    src_proto = sys.argv[1] # <source_inq_net.prototxt
    src_model = sys.argv[1]  # <source_inq.caffemodel>
    output = sys.argv[2]   # <output.compact>

if not os.path.exists(src_proto):
    print "Error: %s does not exist!"%(src_proto)
    sys.exit()
elif not os.path.exists(src_model):
    print "Error: %s does not exist!"%(src_model)
    sys.exit()

caffe.set_mode_cpu()
net = caffe.Net(src_proto, caffe.TEST, weights = src_model)
param_name_list = filter(lambda x: "conv" in x or "ip" in x or "fc" in x or "fire" in x, net.params.keys())

fout = open(output, 'wb')

for param_name in param_name_list:
    if len(net.params[param_name]) == 4:
        weights_inq = net.params[param_name][0].data.flatten()
        count = weights_inq.size
        fabs_params = np.fabs(weights_inq)
        # filter out the zeros
        fabs_params_gt0_exp = np.log2(fabs_params[fabs_params > 0])
        min_exp = np.round(np.min(fabs_params_gt0_exp))
        max_exp = np.round(np.max(fabs_params_gt0_exp))
        if (max_exp - min_exp +1) > num_kept_value:
            print "Error: in layer [%s] weight: max_exp - min_exp + 1 > %d, max_exp: %d, min_exp: %d!"%(param_name, num_kept_value, max_exp, min_exp)
            sys.exit()
        signs_params = np.sign(weights_inq)
        fabs_params[fabs_params > 0] = fabs_params_gt0_exp
        exp_params = np.int32(np.round(signs_params* fabs_params))

        bias_inq    = net.params[param_name][1].data.flatten()

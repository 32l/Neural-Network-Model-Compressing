import os, sys
import numpy as np
from enum import Enum

try:
    caffe_root = os.environ["CAFFE_ROOT"]
except KeyError:
    print "Set system variable CAFFE_ROOT before running the script!"
    sys.exit(-1)

sys.path.append(caffe_root+"/python")
sys.path.append(caffe_root+"/python/caffe/proto")
# sys.path.append(os.getenv("CAFFE_ROOT")+"/python/caffe/proto")
import caffe
import caffe_pb2

help = '''
Usage:
    inq_compact.py <source_inq_net.prototxt> <source_inq.caffemodel> <output_filename> 

Converting the Caffe-output INQ model (twice the size of a normal model) to a 4-bit format, all parameters stored, including zeros.

Code book example: 
4-bit code:           0000   0001   0010   0011   0100   0101   0110  0111
corresponding param: -2^-9  -2^-8  -2^-7  -2^-6  -2^-5  -2^-4  -2^-3  0.0

4-bit code:           1000   1001   1010   1011   1100   1101   1110  1111
corresponding param:  2^-9   2^-8   2^-7   2^-6   2^-5   2^-4   2^-3  N/A
'''
# corresponding to num_quantum_values in caffe.prototxt
num_kept_value = 7
# bits and num_kept_value should be in accordance with each other
bits = 4

# Fire_type = Enum(fire_t, (One_x_One, Three_x_Three))
Param_type = Enum('param_t', ('WEIGHT', 'BIAS'))


if len(sys.argv) != 4:
    print help
    sys.exit(-1)
else:
    src_proto = sys.argv[1] # <source_inq_net.prototxt
    src_model = sys.argv[2]  # <source_inq.caffemodel>
    output = sys.argv[3]   # <output.compact>

if not os.path.exists(src_model):
    print "Error: %s does not exist!"%(src_model)
    sys.exit()
# elif not os.path.exists(src_proto):
#     print "Error: %s does not exist!"%(src_proto)
#     sys.exit()

# caffe.set_mode_cpu()
# net = caffe.Net(src_proto, caffe.TEST, weights = src_model)

model = caffe_pb2.NetParameter()
with open(src_model, 'rb') as f:
    model.ParseFromString(f.read())

layers = model.layer
param_layer_ids = [i for i, layer in enumerate(layers) if len(layer.blobs) > 0]
# param_layer_id = [i for i, layer in enumerate(layers) if ("conv" in layer.name 
#                 or "ip" in layer.name or "fc" in layer.name or "fire" in 
#                 layer.name) and "INQ" in layer.type ]
# param_name_list = filter(lambda x: "conv" in x or "ip" in x or "fc" in x or "fire" in x, net.params.keys())

fout = open(output, 'wb')

def save_to_file(wb_inq, param_name, num_kept_value, bits, param_t):
    fabs_params = np.fabs(wb_inq)

    min_exp = np.round(np.log2(np.min(fabs_params[fabs_params > 0])))
    max_exp = np.round(np.log2(np.max(fabs_params)))
    if (max_exp - min_exp +1) > num_kept_value:
        print "Error: in layer [%s] weight: max_exp - min_exp + 1 > %d, max_exp: %d, min_exp: %d!"%(param_name, num_kept_value, max_exp, min_exp)
        sys.exit()
    positive_add_factor = num_kept_value * 2 - max_exp
    negative_add_factor = num_kept_value - 1 - max_exp

    # convert all params to low-bit value (e.g. 0000 ~ 1111)
    wb_inq[wb_inq > 0] = np.log2(wb_inq[wb_inq > 0]) + positive_add_factor
    wb_inq[wb_inq ==0] = num_kept_value
    wb_inq[wb_inq < 0] = np.log2(-wb_inq[wb_inq < 0]) + negative_add_factor

    # int value, float dtype
    wb_inq = np.around(wb_inq ).astype(np.uint16)

    if True in (wb_inq < 0):
        print "Error: weight in layer %s should not be negative!"%param_name
        sys.exit()
    elif True in (wb_inq > (2**bits-1)):
        print "Error: weight(max:%d) in layer %s should not be over 2**%d = %d!"%(wb_inq.max(), param_name, bits, 2**bits)
        sys.exit()
    # wb_inq = wb_inq.astype(np.uint8)
    # save info to file
    wb_size = wb_inq.size
    # save the minimum exp, uint8
    np.array([min_exp], dtype = np.uint8 ).tofile(fout)
    print "min exp = %d"%min_exp
    # save the params' size
    np.array([wb_size], dtype = np.int32 ).tofile(fout)
    # print "param count ="
    # save the params
    if param_t == Param_type.WEIGHT and ("3x3" in param_name or "conv" in param_name):
        # the param is weight and is in squeeze3x3 layer
        # 16 bits stores three 4-bit number

        num_append = ((wb_size - 1)/3 + 1)*3 - wb_size
        wb_inq = np.append(wb_inq, np.zeros(num_append, dtype = np.uint16))
        wb_size = wb_inq.size
        print "---- saving size: %d"%wb_size
        wb_to_store = wb_inq[np.arange(0, wb_size, 3)] + wb_inq[np.arange(1, wb_size, 3)]*2**bits + wb_inq[np.arange(2, wb_size, 3)]*2**(2*bits)
        print "==== stored size: %d"%wb_to_store.size
        if param_name == 'conv1':
          for i, val in enumerate(wb_to_store):
            print "%4x "%val,
            if (i+1) %10 ==0:
              print ''
        print ''
        wb_to_store.tofile(fout)
    else:
        # num_append = ((wb_size - 1)/2 + 1)*2 - wb_size
        num_append = ((wb_size - 1)/4 + 1)*4 - wb_size
        wb_inq = np.append(wb_inq, np.zeros(num_append, dtype = np.uint16))
        wb_size = wb_inq.size
        print "---- saving size: %d"%wb_size
        # wb_to_store = wb_inq[np.arange(0, wb_size, 2)] + wb_inq[np.arange(1, wb_size, 2)]*2**bits
        wb_to_store = wb_inq[np.arange(0, wb_size, 4)] + wb_inq[np.arange(1, wb_size, 4)]*2**bits + wb_inq[np.arange(2, wb_size, 4)]*2**(2*bits) + wb_inq[np.arange(3, wb_size, 4)]*2**(3*bits)
        print "==== stored size: %d"%wb_to_store.size
        if param_name == 'conv1':
          for i, val in enumerate(wb_to_store):
            print "%4x "%val,
            if (i+1) %10 ==0:
              print ''
        print ''
        wb_to_store.tofile(fout)

for layer_id in param_layer_ids:
    if len(layers[layer_id].blobs) == 4 or len(layers[layer_id].blobs) == 2:
        weights_inq = np.array(layers[layer_id].blobs[0].data).flatten()
        print "saving layer <%s>, weights_size = %d..."%(layers[layer_id].name, weights_inq.size)
        save_to_file(weights_inq, layers[layer_id].name, num_kept_value, bits, Param_type.WEIGHT)
        bias_inq = np.array(layers[layer_id].blobs[1].data).flatten()
        print "saving layer <%s>, bias_size =    %d..."%(layers[layer_id].name, bias_inq.size)
        save_to_file(bias_inq, layers[layer_id].name, num_kept_value, bits, Param_type.BIAS)

'''
for param_name in param_name_list:
    if len(net.params[param_name]) == 4 or len(net.params[param_name]) == 2:
        weights_inq = net.params[param_name][0].data.flatten()
        save_to_file(weights_inq, param_name, num_ketp_value, bits, Param_type.WEIGHT)
        bias_inq = net.params[param_name][1].data.flatten()
        save_to_file(bias_inq, param_name, num_ketp_value, bits, Param_type.BIAS)
    else:
        print "Error: layer [%s]'s param size is not 2 or 4!"%(param_name)
        sys.exit()
'''

fout.close()



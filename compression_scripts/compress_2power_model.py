'''
convert the .caffemodel to compressed format
'''

import sys
import os
import numpy as np
import caffe_pb2

help_ ='''
Usage:
    compress_model.py <source.caffemodel> <target.cmodel>
    Set the CAFFE_ROOT in the source file.
'''

if len(sys.argv) != 3:
    print help_
    sys.exit()
else:
    argv_list = list(sys.argv)
    # print "!!!!!list: ", argv_list
    caffemodel = argv_list[1]
    target = argv_list[2]

if not os.path.exists(caffemodel):
    print "Error: caffemodel does NOT exist!"
    sys.exit()
    
model = caffe_pb2.NetParameter()

f = open(caffemodel, 'rb')
model.ParseFromString(f.read())
f.close()

bits = 4
num_quantum = 7

fout = open(target, 'wb')


for layer in model.layer:
    if 'conv' not in layer.name and 'ip' not in layer.name and 'fc' not in layer.name:
        continue
    # count num of weight/bias and calculate codebook
    nz_num = np.zeros(2)
    nz_num[0] = len(layer.blobs[0].data)    # num of weight
    nz_num[1] =  len(layer.blobs[1].data)   # num of bias
    # To File
    nz_num.tofile(fout)
    
    # weight & bias codebook 
    codebook = np.array([np.zeros(2*num_quantum + 1)]*2)
    # codebook = np.zeros(2**bits)
    
    # codebook for weight
    weight = np.array(layer.blobs[0].data)
    abs_weight = abs(weight)
    max_weight_exp = np.log2(abs_weight.max()).astype(int)
    
    negative_part = [-2**x for x in range(max_weight_exp, max_weight_exp-num_quantum, -1)]  # -2^-1 ~ -2^-7
    positive_part = [2**x for x in range(max_weight_exp - num_quantum +1 , max_weight_exp +1, 1)]   # 2^-1 ~ 2^-1
    # codebook 
    temp = np.append(negative_part, [0])
    codebook[0] = np.append(temp, positive_part)
    
    # codebook for bias
    bias = np.array(layer.blobs[1].data)
    abs_bias = abs(bias)
    max_bias_exp = np.log2(abs_bias.max()).astype(int)
    
    negative_part = [-2**x for x in range(max_bias_exp, max_bias_exp-num_quantum, -1)]  # -2^-1 ~ -2^-7
    positive_part = [2**x for x in range(max_bias_exp - num_quantum +1 , max_bias_exp +1, 1)]   # 2^-1 ~ 2^-1
    # codebook 
    temp = np.append(negative_part, [0])
    codebook[1] = np.append(temp, positive_part)
    # CodeBook to File
    codebook.tofile(fout)
    
    '''weight and bias to file'''
    # Weight to store
    positive_add_factor = num_quantum * 2 - max_weight_exp
    negative_add_factor = num_quantum - 1 - max_weight_exp
    
    weight[weight > 0] = np.log2(weight[weight > 0]) + positive_add_factor
    weight[weight ==0] = num_quantum
    weight[weight < 0] = np.log2(-weight[weight < 0]) + negative_add_factor
    
    np.rint(weight,out=weight)
    
    if True in (weight < 0):
        print "Error: weight in layer %s should not be negative!"%layer.name
        sys.exit()
    elif True in (weight > (2**bits-1)):
        print "Error: weight in layer %s should not be over 2** bits!"%layer.name
        sys.exit()
    
    weight = weight.astype(np.uint8)
    if weight.size % 2 == 1:
        weight = np.append(weight, np.array([0],dtype=np.uint8))
        
    # weight_to_store = np.zeros((weigh.size / 2 , dtype = np.uint8)
    weight_to_store = weight[np.arange(0,weight.size,2)]*2**bits + weight[np.arange(1,weight.size,2)]
    weight_to_store.tofile(fout)
    
    # Bias to store
    positive_add_factor = num_quantum * 2 - max_bias_exp
    negative_add_factor = num_quantum - 1 - max_bias_exp
    
    bias[bias > 0] = np.log2(bias[bias > 0]) + positive_add_factor
    bias[bias ==0] = num_quantum
    bias[bias < 0] = np.log2(-bias[bias < 0]) + negative_add_factor
    
    np.rint(bias,out=bias)
    
    if True in (bias < 0):
        print "Error: bias in layer %s should not be negative!"%layer.name
        sys.exit()
    elif True in (bias > (2**bits-1)):
        print "Error: bias in layer %s should not be over 2**bits!"%layer.name
        sys.exit()
    
    bias = bias.astype(np.uint8)
    if bias.size % 2 == 1:
        bias = np.append(bias, np.array([0],dtype=np.uint8))
        
    # weight_to_store = np.zeros((weigh.size / 2 , dtype = np.uint8)
    bias_to_store = bias[np.arange(0,bias.size,2)]*2**bits + bias[np.arange(1,bias.size,2)]
    bias_to_store.tofile(fout)    
    
fout.close()



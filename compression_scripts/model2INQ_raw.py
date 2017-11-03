'''
Convert <normal model / dns pruned model> to INQ raw model.
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
    model2INQ_raw.py <source_prototxt.prototxt> <source_model.caffemodel> <INQ_net.prototxt> <INQ_raw_output_model.caffemodel>
'''

if len(sys.argv) != 5:
    print help
    sys.exit(-1)
else:
    source_prototxt = sys.argv[1]  #
    source_model = sys.argv[2]   #
    target_prototxt = sys.argv[3]
    target_model = sys.argv[4]    #

if not os.path.exists(source_prototxt):
    print "Error: %s does not exist!"%(source_prototxt)
    sys.exit()
elif not os.path.exists(source_model):
    print "Error: %s does not exist!"%(source_model)
    sys.exit()
elif not os.path.exists(target_prototxt):
    print "Error: %s does not exist!"%(target_prototxt)
    sys.exit()

caffe.set_mode_cpu()

net = caffe.Net(source_prototxt, caffe.TEST, weights=source_model)
net_target = caffe.Net(target_prototxt, caffe.TEST)
param_name_list = filter(lambda x: "conv" in x or "ip" in x or "fc" in x or "fire" in x, net.params.keys())

# number of decorative marks
num_mark = 55

# convert masked w/b to normal w/b
def dns_to_target(wb, wb_dns, wb_dns_mask):
    data = np.zeros(wb_dns.size)
    data[wb_dns_mask == 1] = wb_dns[wb_dns_mask == 1]
    data = data.reshape(wb.shape)
    np.copyto(wb, data)

def display_layer_info(lname, w_mask, b_mask, total_params, params_kept):
    w_kept = np.sum(w_mask)
    b_kept = np.sum(b_mask)
    print "="*num_mark
    print "%s layer w: %d/%d (%f %%) kept"%(lname, w_kept, w_mask.size, 100.0 * w_kept/ w_mask.size)
    print "%s       b: %d/%d (%f %%) kept"%(' '*len(lname), b_kept, b_mask.size, 100.0*b_kept/ b_mask.size)
    print "%s   total: %d/%d (%f %%) kept"%(' '*len(lname), w_kept+b_kept, w_mask.size+b_mask.size, 100*(w_kept+b_kept)/(w_mask.size+b_mask.size))
    return (total_params + w_mask.size + b_mask.size, params_kept + w_kept + b_kept)

# copy normal w/b from source to des
def normal_to_target(wb, wb_src):
    if(wb.size != wb_src.size):
        print "Error: wb.size does not equal to wb_src.size"
    np.copyto(wb, wb_src.reshape(wb.shape))

# Assuming the net and net_target have the same params, which should be the case.
total_params = 0
params_kept = 0
# print dir(net.params[param_name_list[0]])

for param_name in param_name_list:
    # source: DNS model, DNS mask transfered to INQ mask
    if len(net.params[param_name]) == 4:
        w_dns = net.params[param_name][0].data.astype(np.float32).flatten()
        b_dns = net.params[param_name][1].data.astype(np.float32).flatten()
        w_dns_mask = net.params[param_name][2].data.astype(np.float32).flatten()
        b_dns_mask = net.params[param_name][3].data.astype(np.float32).flatten()    

        total_params, params_kept = display_layer_info(param_name, w_dns_mask, b_dns_mask, total_params, params_kept ) 
        dns_to_target(net_target.params[param_name][0].data, w_dns, w_dns_mask)
        dns_to_target(net_target.params[param_name][1].data, b_dns, b_dns_mask)
        normal_to_target(net_target.params[param_name][2].data, w_dns_mask)
        normal_to_target(net_target.params[param_name][3].data, b_dns_mask)
        # dns_to_target(net_target.params[param_name][2].data, w_dns, w_dns_mask)
        # dns_to_target(net_target.params[param_name][3].data, b_dns, b_dns_mask)
    # source: normal model, INQ mask all ones
    elif len(net.params[param_name]) == 2:
        w_src = net.params[param_name][0].data.astype(np.float32).flatten()
        b_src = net.params[param_name][1].data.astype(np.float32).flatten()
        normal_to_target(net_target.params[param_name][0].data, w_src)
        normal_to_target(net_target.params[param_name][1].data, b_src)
        normal_to_target(net_target.params[param_name][2].data, np.ones(net_target.params[param_name][2].shape))
        normal_to_target(net_target.params[param_name][3].data, np.ones(net_target.params[param_name][3].shape))
    # source: Error
    else:
        print "Error: len of net.params[%s] is %d"%(param_name, len((net.params[param_name])))


net_target.save(target_model)

# display the final statistics
if total_params != 0 and params_kept != 0:
    print " "
    print "*"*num_mark
    print " "
    print "Final Statistics: %d/ %d (%f %%) kept"%(params_kept, total_params, 100.0*params_kept/ total_params)
    print "Compression Rate: %f"%(total_params / params_kept)
    print " "
    print "*"*num_mark

print "Model(%s) has been converted to INQ raw model, saved as %s"%(source_model, target_model)
print " "
print "Model %s is ready for INQ training."%(target_model)
print " "



'''
Convert dns pruned model to normal model directly.
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
    dns_to_normal.py <dns_model.prototxt> <dns_model.caffemodel> <target.prototxt> <target.caffemodel>
'''

if len(sys.argv) != 5:
    print help
    sys.exit(-1)
else:
    prototxt = sys.argv[1]  #
    dns_model = sys.argv[2]   #
    target_prototxt = sys.argv[3]
    target = sys.argv[4]    #

if not os.path.exists(prototxt):
    print "Error: %s does not exist!"%(prototxt)
    sys.exit()
elif not os.path.exists(dns_model):
    print "Error: %s does not exist!"%(dns_model)
    sys.exit()
elif not os.path.exists(target_prototxt):
    print "Error: %s does not exist!"%(target_prototxt)
    sys.exit()

caffe.set_mode_cpu()

net = caffe.Net(prototxt, caffe.TEST, weights=dns_model)
net_target = caffe.Net(target_prototxt, caffe.TEST)
param_name_list = filter(lambda x: "conv" in x or "ip" in x or "fc" in x, net.params.keys())

def dns_to_target(wb, wb_dns, wb_dns_mask):
    data = np.zeros(wb_dns.size)
    data[wb_dns_mask == 1] = wb_dns[wb_dns_mask == 1]
    data = data.reshape(wb.shape)
    np.copyto(wb, data)

def display_layer_info(lname, w_mask, b_mask, total_params, params_kept):
    w_kept = np.sum(w_mask)
    b_kept = np.sum(b_mask)
    print "%s layer w: %d/%d (%f %%) kept"%(lname, w_kept, w_mask.size, 100.0 * w_kept/ w_mask.size)
    print "%s       b: %d/%d (%f %%) kept"%(' '*len(lname), b_kept, b_mask.size, 100.0*b_kept/ b_mask.size)
    return (total_params + w_mask.size + b_mask.size, params_kept + w_kept + b_kept)
    
def normal_to_target(wb, wb_src):
    if(wb.size != wb_src.size):
        print "Error: wb.size does not equal wb_src"
    np.copyto(wb, wb_src.reshape(wb.shape))

# Assuming the net and net_target have the same params, and it should be so.
total_params = 0
params_kept = 0
print dir(net.params[param_name_list[0]])

for param_name in param_name_list:
    if len(net.params[param_name]) == 4:
        w_dns = net.params[param_name][0].data.astype(np.float32).flatten()
        b_dns = net.params[param_name][1].data.astype(np.float32).flatten()
        w_dns_mask = net.params[param_name][2].data.astype(np.float32).flatten()
        b_dns_mask = net.params[param_name][3].data.astype(np.float32).flatten()    

        total_params, params_kept = display_layer_info(param_name, w_dns_mask, b_dns_mask, total_params, params_kept ) 
        dns_to_target(net_target.params[param_name][0].data, w_dns, w_dns_mask)
        dns_to_target(net_target.params[param_name][1].data, b_dns, b_dns_mask)
    elif len(net.params[param_name]) == 2:
        w_src = net.params[param_name][0].data.astype(np.float32).flatten()
        b_src = net.params[param_name][1].data.astype(np.float32).flatten()
        normal_to_target(net_target.params[param_name][0].data, w_src)
        normal_to_target(net_target.params[param_name][1].data, b_src)

    else:
        print "Error: len of net.params[%s] is %d"%(param_name, len((net.params[param_name])) )


net_target.save(target)
if total_params != 0 and params_kept != 0:
    print "Statistics: %d/ %d (%f %%) kept"%(params_kept, total_params, 100.0*params_kept/ total_params)
print "Model has been converted from DNS to normal, saved as %s"%(target)



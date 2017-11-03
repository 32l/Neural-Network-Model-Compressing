
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
Convert quantized INQ raw model (size: A) to normal model (size: B), (A ~= 2B).

Usage:
    inq_to_normal.py <inq_net.prototxt> <inq_raw.caffemodel> <target_net.prototxt> <output_target_normal.caffemodel>

---- <inq_net.prototxt>:      the INQ net (e.g. train_val_inq.prototxt) used for training.
---- <inq_raw.caffemodel>:    the output caffe model of <inq_net.prototxt>.
---- <target_net.prototxt>:   the normal net (e.g. train_val.prototxt)  which was converted to <inq_net.prototxt> for INQ quantization.
---- <output_target_normal.caffemodel>: the output caffemodel which can be used for <target_net.prototxt>.

'''

if len(sys.argv) != 5:
    print help
    sys.exit(-1)
else:
    prototxt = sys.argv[1]  #
    inq_model = sys.argv[2]   #
    target_prototxt = sys.argv[3]
    target = sys.argv[4]    #

if not os.path.exists(prototxt):
    print "Error: %s does not exist!"%(prototxt)
    sys.exit()
elif not os.path.exists(inq_model):
    print "Error: %s does not exist!"%(inq_model)
    sys.exit()
elif not os.path.exists(target_prototxt):
    print "Error: %s does not exist!"%(target_prototxt)
    sys.exit()

caffe.set_mode_cpu()

net = caffe.Net(prototxt, caffe.TEST, weights=inq_model)
net_target = caffe.Net(target_prototxt, caffe.TEST)
param_name_list = filter(lambda x: "conv" in x or "ip" in x or "fc" in x or "fire" in x, net.params.keys())

# number of decorative marks
num_mark = 55

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
    
def normal_to_target(wb, wb_src):
    if(wb.size != wb_src.size):
        print "Error: wb.size does not equal wb_src"
    np.copyto(wb, wb_src.reshape(wb.shape))

# Assuming the net and net_target have the same params, and it should be so.
total_params = 0
params_kept = 0
# print dir(net.params[param_name_list[0]])
print ""
for param_name in param_name_list:
    print "converting layer [%s] ..."%param_name," "*max(0,(20-len(param_name))),

    if len(net.params[param_name]) == 4 or len(net.params[param_name]) == 2:
        w_src = net.params[param_name][0].data.astype(np.float32).flatten()
        b_src = net.params[param_name][1].data.astype(np.float32).flatten()
        normal_to_target(net_target.params[param_name][0].data, w_src)
        normal_to_target(net_target.params[param_name][1].data, b_src)
    else:
        print "Error: len of net.params[%s] is %d"%(param_name, len((net.params[param_name])) )
    print "[done]"
net_target.save(target)

# display the final statistics
print ""
print "Success: Model has been converted from INQ to normal, saved as %s"%(target)
print " "



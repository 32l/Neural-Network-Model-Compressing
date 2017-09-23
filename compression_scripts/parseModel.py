import os, sys

caffe_root = os.environ["CAFFE_ROOT"]
sys.path.append(caffe_root+"/python/caffe/proto")

import caffe_pb2

caffemodel_filename = "/home/users/xieqikai/myGitRepo/caffe-0828-compress/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"

model = caffe_pb2.NetParameter()

f = open(caffemodel_filename, 'rb')

model.ParseFromString(f.read())

f.close()

save_filename = "alexnet.dat"

f = open(save_filename, 'w')

# print 'model type: ', type(model)
print >> f,model
# print >> f,model.__str__

# print model.__str__

f.close()



# print model.__str__


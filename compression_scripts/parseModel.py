import caffe_pb2

caffemodel_filename = "/home/users/xieqikai/Dynamic-Network-Surgery-2power_v3_gpu/examples/mnist/tp100_iter_100.caffemodel"

model = caffe_pb2.NetParameter()

f = open(caffemodel_filename, 'rb')

model.ParseFromString(f.read())

f.close()

save_filename = "/home/users/xieqikai/Dynamic-Network-Surgery-2power_v3_gpu/tp_100.dat"

f = open(save_filename, 'w')

print 'model type: ', type(model)
print >> f,model
# print >> f,model.__str__

# print model.__str__

f.close()



# print model.__str__


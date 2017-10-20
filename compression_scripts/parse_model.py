'''
Convert binary caffemodel to human-readable text file.
'''

import os, sys

caffe_root = os.environ["CAFFE_ROOT"]
sys.path.append(caffe_root+"/python/caffe/proto")

import caffe_pb2

help = '''
Usage:
    parse_model.py <binary.caffemodel> <output.txt>
'''

if len(sys.argv) != 3:
    print help
    sys.exit(-1)
else:
    binary_model = sys.argv[1]  #
    output = sys.argv[2]   #

if not os.path.exists(binary_model):
    print "Error: %s does not exist!"%(binary_model)
    sys.exit()


model = caffe_pb2.NetParameter()
f = open(binary_model, 'rb')
print "Parsing binary caffemodel ..."
model.ParseFromString(f.read())
f.close()

f = open(output, 'w')
# print 'model type: ', type(model)
print "Saving file: %s ..."%(output)
print >> f,model
# print >> f,model.__str__

# print model.__str__

f.close()
print "Success: file saved as %s"%(output)


# print model.__str__


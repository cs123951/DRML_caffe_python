import caffe
import numpy as np
caffemodel_filename = 'snapshot_iter_2000.caffemodel'
deploy_file = 'prototxt/deploy_net.prototxt'



net = caffe.Net(deploy_file, caffemodel_filename, caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
#change the order of dimension
transformer.set_transpose('data',(2,0,1))
#if you didn't minus mean during training, this step can be ignored
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)    # scale to (0,255)
transformer.set_channel_swap('data', (2,1,0)) #change RGB to BGR
for img in ['test_pic/SN001_0014_LABEL_6.png','test_pic/SN011_4019_LABEL_2.png']:
    im=caffe.io.load_image(img)
    net.blobs['data'].data[...] = transformer.preprocess('data',im)

    out = net.forward()

    prob= net.blobs['score'].data[0].flatten()
    print(prob)

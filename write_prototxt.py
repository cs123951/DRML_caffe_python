# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L,params as P
# import os
import layers

def pre_layer(file_root,phase):  
    n = caffe.NetSpec()
    data_layer_params = dict(batch_size = 64,
                             im_shape = [170, 170],
                             split = phase,
                             num_labels = 12,
                             dataset_root = file_root)
    # module: the module name, usually the filename, needs to be in $PYTHONPATH
    # layer: the layer name, the class name in the module
    n.data, n.label = L.Python(module = 'layers.multilabel_datalayer', layer = "MultilabelDataLayer",
                               ntop = 2, param_str=str(data_layer_params))
    return str(n.to_proto())
    
def mid_layer():
    n = caffe.NetSpec()
    n.conv1 = L.Convolution(bottom='data',
                            kernel_size=11,
                            param=[{"lr_mult": 1,"decay_mult": 1},{"lr_mult": 2,"decay_mult": 0}],
                            weight_filler=dict(type="gaussian",std=0.01),
                            bias_filler=dict(type="constant",value=0),num_output=32)
    n.relu1 = L.ReLU(n.conv1, in_place=True)

    # clipping layer
    # exec_string = ''
    # for i in range(1,64):
    #    exec_string += 'n.out' + str(i) + ','
    # exec_string += 'n.out64 = L.Python(n.conv1,ntop=64,name="clipping",module = "layers.box_layer",layer="BoxLayer",\
    # param_str=str(dict(width=20,height=20)))'
    # exec(exec_string)
    #
    # for i in range(1,65):
    #     exec('n.bn1_{} = L.BatchNorm(n.out{},name="bn1_{}",\
    #    param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)])'.format(str(i),str(i),str(i)))
    #
    #
    # for i in range(1,65):
    #     exec('n.bn1_{} = L.ReLU(n.bn1_{},name="relu1_{}")'.format(str(i),str(i),str(i)))
    #
    # for i in range(1,65):
    #     exec('n.res1_{} = L.Convolution(n.bn1_{},name="conv1_{}",kernel_size=3,num_output=32,pad=1,\
    #    param=[dict(lr_mult=1,decay_mult=1),dict(lr_mult=2,decay_mult=0)],\
    #    weight_filler=dict(type="gaussian",std=0.01),bias_filler=dict(type="constant",value=1))'.format(str(i),str(i),str(i)))
    #
    # exec_string = 'n.concat = L.Python('
    # for i in range(1,65):
    #    exec_string += 'n.res1_' + str(i) + ','
    # exec_string += 'module="layers.splice_layer",layer="SpliceLayer",name="Splice",\
    #    param_str=str(dict(width=160,height=160)))'
    # exec(exec_string)
    # #    operation = 1表示sum
    # n.add = L.Eltwise(n.conv1,n.concat,name="add",operation=1)
    # n.add = L.ReLU(n.add,name="relu_res")

    n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.norm1 = L.LRN(n.pool1,local_size=5,alpha=1e-4,beta=0.75)
    n.conv2 = L.Convolution(n.norm1, kernel_size=8,num_output=16, \
                param=[{"lr_mult": 1,"decay_mult": 1},{"lr_mult": 2,"decay_mult": 0}],\
                weight_filler=dict(type="gaussian",std=0.01),bias_filler=dict(type="constant",value=1))
    n.relu2 = L.ReLU(n.conv2, in_place=True)

    n.conv3 = L.Convolution(n.relu2, kernel_size=8,num_output=16, \
                param=[{"lr_mult": 1,"decay_mult": 1},{"lr_mult": 2,"decay_mult": 0}],
                weight_filler=dict(type="gaussian",std=0.01),bias_filler=dict(type="constant",value=1))
    n.relu3 = L.ReLU(n.conv3,in_place=True)
    n.conv4 = L.Convolution(n.relu3, kernel_size=6,num_output=16,stride=2, \
                param=[{"lr_mult": 1,"decay_mult": 1},{"lr_mult": 2,"decay_mult": 0}],\
                weight_filler=dict(type="gaussian",std=0.01),bias_filler=dict(type="constant",value=1))
    n.relu4 = L.ReLU(n.conv4)
    n.conv5 = L.Convolution(n.relu4, kernel_size=5,num_output=16, \
                param=[{"lr_mult": 1,"decay_mult": 1},{"lr_mult": 2,"decay_mult": 0}],\
                weight_filler=dict(type="gaussian",std=0.01),bias_filler=dict(type="constant",value=1))
    n.relu5 = L.ReLU(n.conv5)
    n.fc6 = L.InnerProduct(n.relu5, num_output=4096, \
                param=[{"lr_mult": 1,"decay_mult": 1},{"lr_mult": 2,"decay_mult": 0}],\
                weight_filler=dict(type="gaussian",std=0.005),bias_filler=dict(type="constant",value=1))
    n.relu6 = L.ReLU(n.fc6)
    n.drop6 = L.Dropout(n.relu6, dropout_param=dict(dropout_ratio=0.5))
    
    n.fc7 = L.InnerProduct(n.drop6, num_output=2048, \
                param=[{"lr_mult": 1,"decay_mult": 1},{"lr_mult": 2,"decay_mult": 0}],\
                weight_filler=dict(type="gaussian",std=0.005),bias_filler=dict(type="constant",value=1))
    n.relu7 = L.ReLU(n.fc7)
    n.drop7 = L.Dropout(n.relu7, dropout_param=dict(dropout_ratio=0.5))

    n.score = L.InnerProduct(n.drop7, num_output=12, \
                param=[{"lr_mult": 1,"decay_mult": 1},{"lr_mult": 2,"decay_mult": 0}],\
                weight_filler=dict(type="gaussian",std=0.001),\
               bias_filler=dict(type="constant",value=0))

    return str(n.to_proto())
    
def post_layer():
    n = caffe.NetSpec()
    #    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    # n.loss = L.SigmoidCrossEntropyLoss(bottom=['score','label'])
    n.loss = L.Python(bottom=['score','label'],\
                      module='layers.multilabel_sigmoid_cross_entropy_loss_layer', \
                      layer="MultilabelSigmoidCrossEntropyLossLayer", )
#    n.loss = L.Python(n.score,n.label,loss_weight=1,module='multilabel_sigmoid_cross_entropy_loss_layer',layers='MultilabelSigmoidLossLayer')
#    n.loss = L.MultilabelSigmoidLoss(n.score, n.label)
    return str(n.to_proto())
    
def write_train_net(train_file, dataset_file):
    with open(train_file, 'w') as f:
        f.write(str(pre_layer(dataset_file,'train')))
        f.write(str(mid_layer()))
        f.write(str(post_layer()))
    f.close()
    
def write_test_net(test_file, dataset_file):
    with open(test_file, 'w') as f:
        f.write(str(pre_layer(dataset_file,'test')))
        f.write(str(mid_layer()))
        f.write(str(post_layer()))
    f.close()
        
def write_deploy(deploy_file): 
    with open(deploy_file, 'w') as f:
        f.write('name:"Lenet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:64\n')
        f.write('input_dim:3\n')
        f.write('input_dim:170\n')
        f.write('input_dim:170\n')
        f.write(str(mid_layer()))
    f.close()
 
if __name__ == "__main__":
    path = "prototxt/"
    train_file = path + "train_net.prototxt"
    test_file = path + "test_net.prototxt"
    train_dataset = 'disfa/disfa_train.txt';
    test_dataset = 'disfa/disfa_test.txt';

    # train_dataset = 'E:/Projects/DRML_caffe_python/data/au_train.txt'
    # test_dataset = 'E:/Projects/DRML_caffe_python/data/au_test.txt'
    write_train_net(train_file,train_dataset)
    write_test_net(test_file,test_dataset)
    
    deploy_file = path + "deploy_net.prototxt"
    write_deploy(deploy_file)

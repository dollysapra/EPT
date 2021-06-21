import numpy as np
import time
import os
import lmdb
import math
import time
from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    visualize,
    workspace,
    onnx,
)
import onnx.backend 
from caffe2.proto import caffe2_pb2
from caffe2.python.onnx.backend import Caffe2Backend
from caffe2.python.modeling import parameter_info



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train an onnx model with iterations')
    parser.add_argument('f', metavar='f', type=str, action='store', help='path to onnx model')
    args = parser.parse_args()

    data_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10')
    root_folder = os.path.join(os.path.expanduser('~'), 'python_scripts', 'aloha_test')

    onnxmodel = onnx.load(args.f)
    
    init_net_def, predict_net_def = Caffe2Backend.onnx_graph_to_caffe2_net(onnxmodel, device="CPU")

    if not os.path.isdir(root_folder):
        os.makedirs(root_folder)
    
    testing_lmdb_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'testing_lmdb')
   
    arg_scope = {"order": "NCHW"}
    test_model = model_helper.ModelHelper(name="test_model", arg_scope=arg_scope, init_params=False)
    
    #input_data,label = AddInput(val_model,validation_images,validation_lmdb_path)
    
    tdata_uint8,label = test_model.TensorProtosDBInput([],["data_uint8", "label"],batch_size=1000,db=testing_lmdb_path, db_type='lmdb')
    input_data = test_model.Cast(tdata_uint8, "input_data", to=core.DataType.FLOAT)
    input_data = test_model.Scale(input_data,input_data,scale=float(1./256))
    
    tmp_predict_net = core.Net(predict_net_def)
    test_model.net = test_model.net.AppendNet(tmp_predict_net)
    
    #Accuracy is added to validation model/ not train model
    accuracy = brew.accuracy(test_model, ['softmax_output', label ], "Accuracy")

    tmp_param_net = core.Net(init_net_def)
    test_model.param_init_net = test_model.param_init_net.AppendNet(tmp_param_net)
    
    workspace.ResetWorkspace(root_folder)
    
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)
    
    #workspace.RunNet(test_model.net)
   

    accuracy = np.zeros(10)

    for i in range(10):
        workspace.RunNet(test_model.net)
        accuracy[i] = workspace.FetchBlob('Accuracy')

    print(sum(accuracy)/10.0)
 
 
    #print(accuracy)
    
    
if __name__ == "__main__":
    main()

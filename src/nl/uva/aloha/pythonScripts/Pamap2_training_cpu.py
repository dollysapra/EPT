import numpy as np
import time
import os
import lmdb
import math
import time
import random
import onnx
from caffe2.python import (
    brew,
    core,
    model_helper,
    optimizer,
    workspace,
    onnx,
)
import onnx.backend 
import caffe2.python.onnx.frontend
import caffe2.python.onnx.helper

from caffe2.proto import caffe2_pb2
from caffe2.python import utils
from caffe2.python.onnx.backend import Caffe2Backend
from caffe2.python.modeling import parameter_info
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter


def AddInput(model, batch_size, db):
    data_uint8,label = model.TensorProtosDBInput([],["data_uint8", "label"],batch_size=batch_size,db=db, db_type='lmdb')
    input_data = model.Cast(data_uint8, "input_data", to=core.DataType.FLOAT)
    input_data = model.Scale(input_data,input_data,scale=float(1./256))
    input_data = model.StopGradient(input_data, input_data)
    return input_data,label

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train an onnx model with iterations')
    parser.add_argument('f', metavar='f', type=str, action='store', help='path to source onnx model')
    parser.add_argument('i', metavar='i', type=str, action='store', help='iteration or generation number')
    args = parser.parse_args()

    data_folder = os.path.join(os.path.expanduser('~'),'pamap2')
    root_folder = os.path.join(os.path.expanduser('~'), 'python_scripts', 'aloha_test')
    onnxmodel = onnx.load(args.f)
    iterations = int(args.i)
    
    init_net_def, predict_net_def = Caffe2Backend.onnx_graph_to_caffe2_net(onnxmodel, device="CPU")
    #TO SAVE LATER
    dup_init_net_def, dup_predict_net_def = Caffe2Backend.onnx_graph_to_caffe2_net(onnxmodel, device="CPU")

    if not os.path.isdir(root_folder):
        os.makedirs(root_folder)
    
    r = random.randint(0, 50);
    
    # Paths to LMDBs
    training_lmdb_path = os.path.join(data_folder,'db','sfl_train_lmdb'+str(r))
    validation_lmdb_path = os.path.join(data_folder,'pmap2sfl_val_lmdb')
    testing_lmdb_path = os.path.join(data_folder,'pmap2sfl_test_lmdb')
    
    
    # Training params
    training_iters = 400 #1epochs      # total training iterations
    training_net_batch_size = 50 # batch size for training
    validation_images = 3870       # total number of validation images
    validation_batch = 387   # validate every <validation_interval> training iterations
    
    arg_scope = {"order": "NHWC"}
    lr = 0.0001
    #if(iterations > 0):
        #lr = 0.0003 - 0.00001*iterations
    #if(iterations > 20):
        #lr = 0.0001
    #if(iterations == 25):
        #training_iters = 1200
        #training_lmdb_path = os.path.join(data_folder,'pmap2sfl_train_lmdb')
        #validation_lmdb_path = os.path.join(data_folder,'pmap2sfl_test_lmdb')

    device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)

    workspace.ResetWorkspace()
    deploy_model = model_helper.ModelHelper(name="deploy_model", arg_scope=arg_scope, init_params=False)
    tmp_predict_net = core.Net(predict_net_def)
    deploy_model.net = deploy_model.net.AppendNet(tmp_predict_net)


    ##################################################################3
    train_model = model_helper.ModelHelper(name="train_model", arg_scope=arg_scope)
    
    with core.DeviceScope(device_option):
        input_data,label = AddInput(train_model,training_net_batch_size,training_lmdb_path)
        
    tmp_param_net = core.Net(init_net_def)
    train_model.param_init_net = train_model.param_init_net.AppendNet(tmp_param_net, device_option=device_option)
    
    for op in train_model.param_init_net.Proto().op :
        op.device_option.CopyFrom(device_option)
        
    tmp_predict_net = core.Net(predict_net_def)
    train_model.net = train_model.net.AppendNet(tmp_predict_net, device_option=device_option)
     
        
    for op in train_model.net.Proto().op :
        op.device_option.CopyFrom(device_option)
        op.engine = "CUDNN"
        if(op.type == "SpatialBN"):
            op.arg[2].i = 0
            op.output.append(op.name+"_mean")
            op.output.append(op.name+"_var")
            op.output.append(op.name+"_s_mean")
            op.output.append(op.name+"_s_var")
        if(op.type == "Dropout"):
            op.arg[1].i = 0
        
    for blob_in in train_model.net.external_inputs :
        if(str(blob_in).find('weight') != -1):
            train_model.AddParameter(blob_in,parameter_info.ParameterTags.WEIGHT)
        elif(str(blob_in).find('bias') != -1):
            train_model.AddParameter(blob_in,parameter_info.ParameterTags.BIAS)
           
    with core.DeviceScope(device_option): 
        xent = train_model.LabelCrossEntropy(['softmax', label ], 'xent')
        loss = train_model.AveragedLoss(xent, "loss")
        train_model.AddGradientOperators([loss])   
        optimizer.build_adam(train_model,base_learning_rate= lr)
                #weight_decay=0.0001)
         
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)
   
    
    with core.DeviceScope(device_option):
        val_model = model_helper.ModelHelper(name="val_model", arg_scope=arg_scope, init_params=False)
    
        vdata_uint8,label = val_model.TensorProtosDBInput([],["data_uint8", "label"],batch_size=validation_batch,db=validation_lmdb_path, db_type='lmdb')
        input_data = val_model.Cast(vdata_uint8, "input_data", to=core.DataType.FLOAT)
        input_data = val_model.Scale(input_data,input_data,scale=float(1./256))
    
    tmp_predict_net = core.Net(predict_net_def)
    val_model.net = val_model.net.AppendNet(tmp_predict_net)
     
    for op in val_model.net.Proto().op :
        op.device_option.CopyFrom(device_option)
         
    #Accuracy is added to validation model/ not train model
    with core.DeviceScope(device_option):
        accuracy = brew.accuracy(val_model, ['softmax', label ], "Accuracy") 
    
    #########################################################################################################3

   
    import time
    start_time = time.time()
  
    # Now, we run the network (forward & backward pass)
    for i in range(training_iters):
        workspace.RunNet(train_model.net)
        if(i%100==1):
            print("-",workspace.FetchBlob('loss') )
    #print("-",workspace.FetchBlob('loss') )
    #####################################################
    val_iterations = int(math.ceil(validation_images/validation_batch))
    val_accuracy = np.zeros(val_iterations)
    workspace.RunNetOnce(val_model.param_init_net)
    workspace.CreateNet(val_model.net, overwrite=True)
    for j in range(val_iterations):
        workspace.RunNet(val_model.net)  
        val_accuracy[j] = workspace.FetchBlob('Accuracy')
    loss = workspace.FetchBlob('loss')
    
    print("..",str(r),"-Valaccuracy:")
    if(loss>5.0):
        print("0.0")
    else:
        #print(val_accuracy[0])
        print(sum(val_accuracy)/float(val_iterations))
    
    
    ##############################################EXPORT############################
    workspace.RunNetOnce(deploy_model.param_init_net)
    workspace.CreateNet(deploy_model.net, overwrite=True)
    

    ################################################################    
    init_net = caffe2_pb2.NetDef()
    for param in deploy_model.net.external_inputs:
        if(param == ("OC2_DUMMY_1")):   
            op = core.CreateOperator("GivenTensorIntFill", [], [param],arg=[ utils.MakeArgument("shape", workspace.FetchBlob(param).shape),utils.MakeArgument("values", workspace.FetchBlob(param))])
                 
        else:
            op = core.CreateOperator("GivenTensorFill", [], [param],arg=[ utils.MakeArgument("shape", workspace.FetchBlob(param).shape),utils.MakeArgument("values", workspace.FetchBlob(param))])
       
          
        init_net.op.extend([op])
        
    
    ######################################################################3  
    data_type = onnx.TensorProto.FLOAT
    data_shape = (1, 40, 100, 1)
    value_info = { 'input_data': (data_type, data_shape)}
    
    onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
        deploy_model.net.Proto(),
        init_net, 
        value_info )

    onnx.save(onnx_model, args.f )

    workspace.ResetWorkspace()
    
    
    print(":sec:")
    print(time.time() - start_time)

if __name__ == "__main__":
    main()

# EPT Evolutionary Piecemeal Training
Languages: Python and Java


Dependencies: 

Jar files: (included in "lib" folder)
1. commons-math3-3.6.1.jar
2. jenetics-4.2.1_modified.jar (Modified version based on Jenetics (jenetics.io) 4.2.1 library)
3. jenetics-4.2.1_modified.jar
4. onnx.jar (For onnx file creation and modification - based on protobuf)
5. espamjun21.jar published by Svetlana Minakova ()

Training for CIFAR-10 and PAMAP2 earlier versions were done using Caffe2 (in pytorch)
Training for VOC and CIFAR-10 resnet models is done using Keras/Tensoflow v2.0.0
For continued training during the evolutionary algorithm, it is important that weights can be correctly identified to be passed on from one generation to the next. 
getInitializerMap() in ONNXAlteration file was based on caffe2 earlier and support has been added for Keras2Onnx v1.7.0 and onnx2keras v0.0.23. 
To use different converter (or different version), this function can be updated.  



This work was first published in "Constrained evolutionary piecemeal training to design convolutional neural networks" https://link.springer.com/chapter/10.1007/978-3-030-55789-8_61

Best CIFAR-10 (92.5%) and PAMAP2(94.36%) can be downloaded from here- 

The code for PAMAP2 was used for paper "An evolutionary optimization algorithm for gradually saturating objective functions".
https://dl.acm.org/doi/abs/10.1145/3377930.3389834


For Scenario based run-time switching,
Scenarios for VOC are here -
Scenarios for PAMAP2 are here - 


For extended journal paper titled "Designing Convolutional Neural Networks withConstrained Evolutionary Piecemeal Training"
PAMAP2 pareto front is here - 


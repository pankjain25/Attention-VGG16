function graph = Attention_VGG16_Ablation4(classes, Nclass,inputSize)
lgraph = layerGraph();
classNames = classes;
tempLayers = [
    imageInputLayer(inputSize,"Name","input")
    convolution2dLayer([3 3],64,"Name","conv1_1","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu1_1")
    convolution2dLayer([3 3],64,"Name","conv1_2","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu1_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],64,"Name","Theta_X_1","Padding","same","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","pool1","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv2_1","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu2_1")
    convolution2dLayer([3 3],128,"Name","conv2_2","Padding","same","WeightL2Factor",0)
    reluLayer("Name","relu2_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],128,"Name","Theta_X_2","Padding","same","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],64,"Name","Phi_Gating_1","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],1,"Name","Psi_1","Padding","same")
    sigmoidLayer("Name","sigmoid_1")
    transposedConv2dLayer([3 3],64,"Name","transposed-conv_1","Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_1")
    maxPooling2dLayer([2 2],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv3_1","Padding","same")
    reluLayer("Name","relu3_1")
    convolution2dLayer([3 3],256,"Name","conv3_2","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu3_2")
    convolution2dLayer([3 3],256,"Name","conv3_3","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu3_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],256,"Name","Theta_X_3","Padding","same","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],128,"Name","Phi_Gating_2","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    reluLayer("Name","relu_2")
    convolution2dLayer([1 1],1,"Name","Psi_2","Padding","same")
    sigmoidLayer("Name","sigmoid_2")
    transposedConv2dLayer([3 3],128,"Name","transposed-conv_2","Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_2")
    maxPooling2dLayer([2 2],"Name","pool3","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv4_1","Padding",[1 1 1 1])
    reluLayer("Name","relu4_1")
    convolution2dLayer([3 3],512,"Name","conv4_2","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu4_2")
    convolution2dLayer([3 3],512,"Name","conv4_3","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu4_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],256,"Name","Theta_X_4","Padding","same","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([1 1],256,"Name","Phi_Gating_3","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    reluLayer("Name","relu_3")
    convolution2dLayer([1 1],1,"Name","Psi_3","Padding","same")
    sigmoidLayer("Name","sigmoid_3")
    transposedConv2dLayer([3 3],256,"Name","transposed-conv_3","Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_3")
    maxPooling2dLayer([2 2],"Name","pool4","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv5_1","Padding",[1 1 1 1])
    reluLayer("Name","relu5_1")
    convolution2dLayer([3 3],512,"Name","conv5_2","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu5_2")
    convolution2dLayer([3 3],512,"Name","conv5_3","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu5_3")
    convolution2dLayer([1 1],256,"Name","Phi_Gating_4","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    reluLayer("Name","relu_4")
    convolution2dLayer([1 1],1,"Name","Psi_4","Padding","same")
    sigmoidLayer("Name","sigmoid_4")
    transposedConv2dLayer([3 3],512,"Name","transposed-conv_4","Cropping","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_4")
    maxPooling2dLayer([2 2],"Name","maxpool_4","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_4")
    maxPooling2dLayer([2 2],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(2048,"Name","fc_1")
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(2048,"Name","fc_2")
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(Nclass,"Name","fc_3")
    softmaxLayer("Name","prob")
    focalLossLayer('Classes',classNames,'Name','focallosslayer')];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"relu1_2","Theta_X_1");
lgraph = connectLayers(lgraph,"relu1_2","pool1");
lgraph = connectLayers(lgraph,"relu1_2","multiplication_1/in2");
lgraph = connectLayers(lgraph,"Theta_X_1","addition_1/in2");
lgraph = connectLayers(lgraph,"relu2_2","Theta_X_2");
lgraph = connectLayers(lgraph,"relu2_2","Phi_Gating_1");
lgraph = connectLayers(lgraph,"relu2_2","depthcat_1/in2");
lgraph = connectLayers(lgraph,"relu2_2","multiplication_2/in2");
lgraph = connectLayers(lgraph,"Theta_X_2","addition_2/in2");
lgraph = connectLayers(lgraph,"Phi_Gating_1","addition_1/in1");
lgraph = connectLayers(lgraph,"transposed-conv_1","multiplication_1/in1");
lgraph = connectLayers(lgraph,"maxpool_1","depthcat_1/in1");
lgraph = connectLayers(lgraph,"relu3_3","Theta_X_3");
lgraph = connectLayers(lgraph,"relu3_3","Phi_Gating_2");
lgraph = connectLayers(lgraph,"relu3_3","depthcat_2/in2");
lgraph = connectLayers(lgraph,"relu3_3","multiplication_3/in2");
lgraph = connectLayers(lgraph,"Theta_X_3","addition_3/in2");
lgraph = connectLayers(lgraph,"Phi_Gating_2","addition_2/in1");
lgraph = connectLayers(lgraph,"transposed-conv_2","multiplication_2/in1");
lgraph = connectLayers(lgraph,"maxpool_2","depthcat_2/in1");
lgraph = connectLayers(lgraph,"relu4_3","Theta_X_4");
lgraph = connectLayers(lgraph,"relu4_3","Phi_Gating_3");
lgraph = connectLayers(lgraph,"relu4_3","depthcat_3/in2");
lgraph = connectLayers(lgraph,"relu4_3","multiplication_4/in2");
lgraph = connectLayers(lgraph,"Theta_X_4","addition_4/in2");
lgraph = connectLayers(lgraph,"Phi_Gating_3","addition_3/in1");
lgraph = connectLayers(lgraph,"transposed-conv_3","multiplication_3/in1");
lgraph = connectLayers(lgraph,"maxpool_3","depthcat_3/in1");
lgraph = connectLayers(lgraph,"Phi_Gating_4","addition_4/in1");
lgraph = connectLayers(lgraph,"transposed-conv_4","multiplication_4/in1");
lgraph = connectLayers(lgraph,"transposed-conv_4","depthcat_4/in2");
lgraph = connectLayers(lgraph,"maxpool_4","depthcat_4/in1");

graph = lgraph;
plot(lgraph);
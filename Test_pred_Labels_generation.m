clc; clear all; close all;

%%Load your ofline model weights
%%
model_path = "W:\(1)-CarotidClassification\Plaque-Classification\(5)-Xception\Ablation-Study";
for j = 1:5
    
model_name = sprintf('Xception_model_%d.mat',j);
    DL_model = load(fullfile(model_path, model_name));
%% Change the value of model number inside below line using value of j
%%
if j ==1 
        net = DL_model.Xception_model_1;
    elseif j==2
        net = DL_model.Xception_model_2;
    elseif j==3
        net = DL_model.Xception_model_3;
    elseif j==4
        net = DL_model.Xception_model_4;
    elseif j==5
        net = DL_model.Xception_model_5;
end

    YPred = classify(net,DL_model.imdsTest);
    
   [~,~, Ypred2] = unique(categorical(YPred));
% Get the known labels
    YTest = DL_model.imdsTest.Labels;
    [~,~, Ytest2] = unique(categorical(YTest));
    labels = [Ytest2  Ypred2];
    sheet = sprintf('Sheet%d',j);
    xlswrite('True-and-predicted-labels-Xception.xlsx',labels,sheet)
end
clc; clear all; close all;

%%Load your ofline model weights
%%
j = 5
model_path = pwd;
model_name = sprintf('Attention_VGG16_Ablation4_%d.mat',j);


DL_model = load(fullfile(model_path, model_name));

%% Change the value of model number inside below line using value of j
%%
net = DL_model.Attention_VGG16_Ablation4_5;
YPred = classify(net,DL_model.imdsTest);

% Get the known labels

YTest = DL_model.imdstest.Labels;

% accuracy = sum(YPred == YTest)/numel(YTest)
% % Tabulate the results using a confusion matrix.
% confMat = confusionmat(YTest, YPred);
% % Convert confusion matrix into percentage form
% confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
% show = confusionchart(YPred,YTest)

[GNX, ~, X] = unique(YTest);
% [GNY, ~, Y] = unique(YPred);
%% Class Names High = 1; Low = 2; Moderate =3;
%%
% [c_matrix,Result,RefereceResult]= confusion.getMatrix(X,Y);
% results_vector = [Result.Accuracy, Result.Error, Result.Sensitivity, Result.Specificity, Result.Precision,Result.FalsePositiveRate, Result.F1_score, Result.MatthewsCorrelationCoefficient, Result.Kappa]
sheetNo = sprintf('Sheet%d',j);
% column_heading = {'Accuracy', 'Error', 'Sensitivity', 'Specificity', 'Precision', 'FPR','F1', 'MCC', 'Kappa'};
% 
% xlswrite('ResultsAndMetrics.xlsx',column_heading,sheetNo,'A1')
% 
% xlswrite('ResultsAndMetrics.xlsx',results_vector,sheetNo,'A2')
% column_heading2 = {'ConfusionMatrix'};
% column_heading3 = {'ConfusionMatrix%'};
% 
% xlswrite('ResultsAndMetrics.xlsx',column_heading2,sheetNo,'A6')
% xlswrite('ResultsAndMetrics.xlsx',column_heading3,sheetNo,'E6')
% 
% xlswrite('ResultsAndMetrics.xlsx',c_matrix, sheetNo,'A7');
% xlswrite('ResultsAndMetrics.xlsx',confMat, sheetNo,'E7');
score = predict(net,DL_model.imdstest)
[X1,Y1,T,AUC] = perfcurve(X,score(:,2),2)

xlswrite('labels&scores.xlsx',score, sheetNo,'A1');
xlswrite('labels&scores.xlsx',X, sheetNo,'E1');
% xlswrite('labels&scores.xlsx',YTest, sheetNo,'A1');


% xlswrite('ROC_coordinates.xlsx',AUC, sheetNo,'D1');




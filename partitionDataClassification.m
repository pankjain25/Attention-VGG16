function [imdsTrain, imdsTest] = partitionDataClassification(imds,idx1,idx2 )

% rng(0); 
% numFiles = numel(imds.Files);
% shuffledIndices = randperm(numFiles);
% % 
% % % Use 60% of the images for training.
% numTrain = round(0.80 * numFiles);
trainingIdx = idx1;

% Use 20% of the images for validation
% numVal = round(0.20 * numFiles);
% valIdx = shuffledIndices(numTrain+1:numFiles);
valIdx=idx2;
% Use the rest for testing.

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
trainingLabels = imds.Labels(trainingIdx);

valImages = imds.Files(valIdx);
valLabels = imds.Labels(valIdx);

imdsTrain = imageDatastore(trainingImages);
imdsTrain.Labels = trainingLabels;

imdsTest = imageDatastore(valImages);
imdsTest.Labels =  valLabels;

end
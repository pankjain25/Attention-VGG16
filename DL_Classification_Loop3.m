clc; clear all; close all;

imageFolder = input('Enter the image Folder path', "s");

imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);


fold = 5;

Label_count = countEachLabel(imds);
size_Label_count = size(Label_count);
Nclass = size_Label_count(1);
imagesize = [224, 224, 3];
classes = table2array(Label_count(:,1));

model = Attention_VGG16_Ablation4(classes,Nclass,imagesize);
plot(model);
c = cvpartition(imds.Files,'KFold',fold)
for Mod = 5:fold
    
idx1 = training(c, Mod);
idx2 = test(c, Mod);

[imdstrain, imdstest] = partitionDataClassification(imds,idx1,idx2);

       if Mod == 1
            
            imdsTrain = imdstrain;
            imdsTest = imdstest;
            Test_files=  imdsTest.Files(1:end);
            Training_files=  imdsTrain.Files(1:end);
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            xlswrite('Training_Img_files.xlsx',Training_files,sprintf('Sheet%d',Mod),'A1');
            options = trainingOption(imdsTest);
            Attention_VGG16_Ablation4_1 = trainNetwork(imdsTrain,model,options)
            save Attention_VGG16_Ablation4_1
            clear Attention_VGG16_Ablation4_1;
        elseif Mod == 2
            
            imdsTrain = imdstrain;
            imdsTest = imdstest;
            Test_files=  imdsTest.Files(1:end);
            Training_files=  imdsTrain.Files(1:end);
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            xlswrite('Training_Img_files.xlsx',Training_files,sprintf('Sheet%d',Mod),'A1');
            options = trainingOption(imdsTest);
            Attention_VGG16_Ablation4_2 = trainNetwork(imdsTrain,model,options)
            save Attention_VGG16_Ablation4_2
            clear Attention_VGG16_Ablation4_2;
        elseif Mod == 3
            
            imdsTrain = imdstrain;
            imdsTest = imdstest;
            Test_files=  imdsTest.Files(1:end);
            Training_files=  imdsTrain.Files(1:end);
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            xlswrite('Training_Img_files.xlsx',Training_files,sprintf('Sheet%d',Mod),'A1');
            options = trainingOption(imdsTest);
            Attention_VGG16_Ablation4_3 = trainNetwork(imdsTrain,model,options);
            save Attention_VGG16_Ablation4_3
            clear Attention_VGG16_Ablation4_3;
        elseif Mod == 4      
            
            imdsTrain = imdstrain;
            imdsTest = imdstest;
            Test_files=  imdsTest.Files(1:end);
            Training_files=  imdsTrain.Files(1:end);
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            xlswrite('Training_Img_files.xlsx',Training_files,sprintf('Sheet%d',Mod),'A1');            
            options = trainingOption(imdsTest);
            Attention_VGG16_Ablation4_4 = trainNetwork(imdsTrain,model,options);
            save Attention_VGG16_Ablation4_4
            clear Attention_VGG16_Ablation4_4;
        elseif Mod == 5    
            
            imdsTrain = imdstrain;
            imdsTest = imdstest;
            Test_files=  imdsTest.Files(1:end);
            Training_files=  imdsTrain.Files(1:end);
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            xlswrite('Training_Img_files.xlsx',Training_files,sprintf('Sheet%d',Mod),'A1');
            options = trainingOption(imdsTest);
            Attention_VGG16_Ablation4_5 = trainNetwork(imdsTrain,model,options);
            save Attention_VGG16_Ablation4_5
            clear Attention_VGG16_Ablation4_5;
%        
       end
end


function options = trainingOption(imdsTest)

options = trainingOptions('adam', ...
    'MaxEpochs',100,...
    'MiniBatchSize',8,...
    'ValidationData',imdsTest, ...
    'ValidationPatience',30,...
    'InitialLearnRate',1e-4, ...
    'Plots','training-progress');
end 


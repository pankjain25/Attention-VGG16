clear all; clc; warning off;

m=1;
model_name = sprintf('Attention_VGG16_Ablation4_%d',m);
model_path = pwd;
model = load(fullfile(model_path,model_name));
outpath = strcat(pwd,'\heatmap');

network = model.Attention_VGG16_Ablation4_1;
% test_image = 'W:\(1)-CarotidClassification\Plaque-Classification\(0)-Database\two class images\HighRisk\HR_P59.png';
for im = 1:length(model.imdsTest.Files)
    test_image = string(model.imdsTest.Files(im));
    splitted_filename = split(test_image,'\');
    filename = splitted_filename(7,1)
    img = imread(test_image);
    label = string(table2cell(model.Label_count(:,1)));
    score = gradCAM(network,img,label);
    figure
    h = imshow(img)
    hold on

    h1 = imshow(score(:,:,1));
    colormap jet
    hold on
    H = getframe(gca)
    out_filename =fullfile(outpath,filename);
    imwrite(H.cdata, out_filename)
    close
end
% label = classify(network,img)

% map = jet;
% c= imagesc(score(:,:,1),'AlphaData',0.1);
% out_filename =fullfile(outpath,filename);
% imwrite(h1.CData,out_filename);
% imagesc(score(:,:,1),'AlphaData',0.1)
% h2 = imfuse(h.CData,h1.CData);
% h3 = imshow(h2,map)
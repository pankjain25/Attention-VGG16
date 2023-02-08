clear all; clc; warning off;
model_path = pwd;
heatmaps = strcat(pwd,'\heatmap');
outpath =  strcat(pwd,'\overlay');

for m=1:1
model_name = sprintf('Attention_VGG16_Ablation4_%d',m);
model = load(fullfile(model_path,model_name));

for im = 1:length(model.imdsTest.Files)

    test_image = string(model.imdsTest.Files(im));
    splitted_filename = split(test_image,'\');
    filename = splitted_filename(end,1)
    background_Raw = imread(test_image);

    Foreground_Heatmap = imread(fullfile(heatmaps,filename));

    imshow(background_Raw, 'InitialMag', 'fit')
    hold on
    h = imshow(Foreground_Heatmap);
    hold off
    set(h, 'AlphaData', background_Raw)
    H = getframe(gca);
    out_filename =fullfile(outpath,filename);
    imwrite(H.cdata, out_filename)
    close

end
clear model
end






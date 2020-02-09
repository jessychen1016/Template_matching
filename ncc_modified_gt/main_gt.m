clear all
close all
clc
tic;
img = imread('ortho_no_car.jpeg');
D='./ground_pics';
S = dir(fullfile(D,'*.jpg')); % pattern to match filenames.
matchinglist = zeros(3,numel(S));
for pic_num = 1:numel(S)
    F = fullfile(D,S(pic_num).name);
    temp = imread(F);
    
    img_g = rgb2gray(img);
    temp_g = rgb2gray(temp);
    
    temp_g = imresize(temp_g, 3.54);
    temp_r = imrotate(temp_g, -93.7);
    [img_H,img_W] = size(img_g);
    
    regionXmin=0;
    regionXmax=1000;
    regionYmin=439;
    regionYmax=1184;
    
    maxrotation=1;
    angle_resolution = 1;
    val_max = -1;
    xp = 0;
    yp = 0;
    number = 0;
    totalcomputation = 0;
    valmaxlist=zeros(1,maxrotation);
    coordinatelists=zeros(2,maxrotation);
    for rotationtest =1:maxrotation
        temp_r1 = imrotate(temp_r, -(rotationtest-1)*angle_resolution);
        [temp_H,temp_W] = size(temp_r1);
        totalcomputation = totalcomputation+(regionXmax-regionXmin-temp_W)*(regionYmax-regionYmin-temp_H);
        imshow(temp_r1);
    end
    
    for rotation=1:maxrotation
                
    
        temp_g = imrotate(temp_r, -(rotation-1)*angle_resolution);
        
        
        [temp_H,temp_W] = size(temp_g);
        
        dis = ones(regionYmax-temp_H-regionYmin,regionXmax-temp_W-regionXmin);
        
        for y=regionYmin:regionYmax-temp_H
            for x=regionXmin:regionXmax-temp_W
                val = NCC(img_g,temp_g,x,y);
                number = number + 1;
                progress = 100*number/totalcomputation;
                fprintf("progress of picture %d = ", pic_num);
                fprintf("%f\n", progress);
                dis(y-regionYmin+1,x-regionXmin+1)=val;
                if val > val_max
                    val_max = val;
                    xp = x;
                    yp = y;
                    angle=rotation;
                end   
            end
        end
        valmaxlist(1,rotation)=val_max;
        coordinatelists(1,rotation)=xp;
        coordinatelists(2,rotation)=yp;
    end
    
    
    figure
    hold on
    
    imshow(img)
    
    line([xp xp+temp_W], [yp yp],'Color','g','LineWidth',0.5);
    line([xp xp], [yp yp+temp_H],'Color','g','LineWidth',0.5);
    line([xp+temp_W xp+temp_W], [yp yp+temp_H],'Color','g','LineWidth',0.5);
    line([xp xp+temp_W], [yp+temp_H yp+temp_H],'Color','g','LineWidth',0.5);
    matchinglist(1, pic_num) = pic_num;
    matchinglist(2, pic_num) = xp+(temp_W-1)/2;
    matchinglist(3, pic_num) = yp+(temp_H-1)/2;
    
end

time=toc;
% hold off
% for rotation = 1: maxrotation
%     figure
%     heatmap(dis)
% end

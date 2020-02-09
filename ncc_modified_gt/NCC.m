function val = NCC(img,temp,x,y)
    [H,W] = size(temp);
	
    sum_img = 0.0;
    sum_temp = 0.0;
    sum_2 = 0.0;
        
    for i=1:W
        for j=1:H
            if(temp(j,i) ~= 255 && img(y+j, x+i) ~=0)
                sum_img = sum_img + double(img(y+j,x+i))^2;
                sum_temp = sum_temp + double(temp(j,i))^2;
                sum_2 = sum_2 + (2-abs((W/2)-i)/(W/2)*abs((H/2)-j)/(H/2))*double(img(y+j,x+i))*double(temp(j,i));
%                 sum_2 = sum_2 + double(img(y+j,x+i))*double(temp(j,i));
            end
        end
    end

    val = sum_2/sqrt(double(sum_img)*double(sum_temp));
function x_num = str2num_(x)
    if ischar(x)
        x_num = str2num(x);
    else
        x_num = x;
    end
end
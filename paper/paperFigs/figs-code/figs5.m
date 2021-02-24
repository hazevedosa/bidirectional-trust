close all; clear all; clc;


dx = 0.001;

x = 0:dx:1;
y = zeros(1, length(x));
l = 0.2;
u = 0.8;

for i = 1:length(x)
    if(x(i) >= l && x(i) < u)
        y(i) = 1 / (u - l);
    end
end


plot(x, y, 'LineWidth', 1.25)
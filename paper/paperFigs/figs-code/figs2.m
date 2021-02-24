close all; clear all; clc;


x = 0:0.001:1;
y = 0:0.001:1;

beta = [1 100000000];
zeta = [0.1 100000000];


figure(1)
set(gcf, 'Position', [400 250 500 500])

bbeta = beta(1);
zzeta = zeta(1);


t = 1 ./ (1 + exp(bbeta * (x-0.5))).^zzeta;

subplot(2, 2, 1)
plot(x, t, 'LineWidth', 1.5)
axis equal
axis([0 1 0 1])
set(gca,'XTick',[0:0.5:1])
set(gca,'YTick',[0, 1])
grid on;

title('$\beta_i = 1 \wedge \zeta_i = 0.1$', 'Interpreter', 'Latex', 'FontSize', 15);

xlabel('$\bar{\lambda}_i$', 'Interpreter', 'Latex', 'FontSize', 15);
ylabel('$\tau_i$', 'Interpreter', 'Latex', 'FontSize', 15);

hold on


bbeta = beta(1);
zzeta = zeta(2);


t = 1 ./ (1 + exp(bbeta * (x-0.5))).^zzeta;

subplot(2, 2, 2)
plot(x, t, 'LineWidth', 1.5)
axis equal
axis([0 1 0 1])
set(gca,'XTick',[0:0.5:1])
set(gca,'YTick',[0, 1])
grid on;

title('$\beta_i = 1 \wedge \zeta_i = 10$', 'Interpreter', 'Latex', 'FontSize', 15);

xlabel('$\bar{\lambda}_i$', 'Interpreter', 'Latex', 'FontSize', 15);
ylabel('$\tau_i$', 'Interpreter', 'Latex', 'FontSize', 15);

hold on


bbeta = beta(2);
zzeta = zeta(1);


t = 1 ./ (1 + exp(bbeta * (x-0.5))).^zzeta;

subplot(2, 2, 3)
plot(x, t, 'LineWidth', 1.5)
axis equal
axis([0 1 0 1])
set(gca,'XTick',[0:0.5:1])
set(gca,'YTick',[0, 1])
grid on;

title('$\beta_i = 100 \wedge \zeta_i = 0.1$', 'Interpreter', 'Latex', 'FontSize', 15);

xlabel('$\bar{\lambda}_i$', 'Interpreter', 'Latex', 'FontSize', 15);
ylabel('$\tau_i$', 'Interpreter', 'Latex', 'FontSize', 15);

hold on


bbeta = beta(2);
zzeta = zeta(2);


t = 1 ./ (1 + exp(bbeta * (x-0.5))).^zzeta;

subplot(2, 2, 4)
plot(x, t, 'LineWidth', 1.5)
axis equal
axis([0 1 0 1])
set(gca,'XTick',[0:0.5:1])
set(gca,'YTick',[0, 1])
grid on;

title('$\beta_i = 100 \wedge \zeta_i = 10$', 'Interpreter', 'Latex', 'FontSize', 15);

xlabel('$\bar{\lambda}_i$', 'Interpreter', 'Latex', 'FontSize', 15);
ylabel('$\tau_i$', 'Interpreter', 'Latex', 'FontSize', 15);

hold on


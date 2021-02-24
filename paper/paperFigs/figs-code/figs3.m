close all; clear all; clc;

b = 500;

l1 = 0.0;
u1 = 1.0;
l2 = 0.0;
u2 = 1.0;

dl = 0.02;

vv = 0.0:dl:1.0;

[L1, L2] = meshgrid(vv, vv);

T = trust_(l1, u1, b, L1) .* trust_(l2, u2, b, L2);

figure()
set(gcf, 'Position', [10 10 300 250])
surf(L1, L2, T, 'EdgeColor',[0 0 0], 'FaceAlpha', 0.3, 'EdgeAlpha', 0.2)
view([45 45])
hold on

poly_Xs1 = [-0.1 1.1 1.1 -0.1];
poly_Ys1 = [l2 l2 u2 u2];

pgon1 = polyshape(poly_Xs1, poly_Ys1);
plot(pgon1, 'FaceColor', [1 1 1], 'FaceAlpha', 0.0, 'LineStyle', '--', 'EdgeColor', [0.5 0.5 0.5])


poly_Xs2 = [l1 u1 u1 l1];
poly_Ys2 = [-0.1 -0.1 1.1 1.1];

pgon2 = polyshape(poly_Xs2, poly_Ys2);
plot(pgon2, 'FaceColor', [1 1 1], 'FaceAlpha', 0.0, 'LineStyle', '--', 'EdgeColor', [0.5 0.5 0.5])

poly_Xs3 = [l1 u1 u1 l1];
poly_Ys3 = [l2 l2 u2 u2];

pgon3 = polyshape(poly_Xs3, poly_Ys3);
plot(pgon3, 'FaceColor', [0.55 0.55 0.55], 'FaceAlpha', 1.0, 'EdgeColor', [0.55 0.55 0.55])

axis([0 1 0 1])

zlabel('$\tau$','Interpreter','latex', 'FontSize',20)
xlabel('$\bar{\lambda}_1$','Interpreter','latex', 'FontSize',20)
ylabel('$\bar{\lambda}_2$','Interpreter','latex', 'FontSize',20)

set(gcf, 'Renderer', 'painters');
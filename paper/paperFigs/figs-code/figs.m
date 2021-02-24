close all; clear all; clc;

results_gp  = load('results_mat_gp');
results_btm = load('results_mat_btm');
results_opt = load('results_mat_lineargaussian.mat');

lcs_gp  = results_gp.allresults(:, 4);
lcs_btm = results_btm.allresults(:, 4);
lcs_opt = results_opt.allresults(:, 4);

plot_fig1 = true;
plot_fig2 = true;


if (plot_fig1 == true)
    y_min = 0.5;
    y_max = 1.0;


    m = 1;

    for i=1:size(lcs_gp, 1)
        m = max(m, size(lcs_gp{i}, 2));
    end

    data = zeros(3, m, 10);
    data_avg_gp = zeros(3, m);

    for i=1:size(lcs_gp, 1)
        data_ = lcs_gp{i};
        data(1, 1:m, i) = 1:m;
        data(2:end, 1:size(data_, 2), i) = data_(2:end, :);
        data(2, size(data_, 2):end, i) = data_(2, end);
        data(3, size(data_, 2):end, i) = data_(3, end);
        data_avg_gp = data_avg_gp + data(:, :, i)/10;
    end

    sdevs_gp = zeros(2, m);

    for i=1:2
        for j = 1:m
            sdevs_gp(i, j) = std(data(i+1, j, :));
        end
    end

    figure(1)
    set(gcf, 'Position', [10 10 1800 300])
    subplot(1, 4, 2)
    curve = plot(data_avg_gp(1, :), data_avg_gp(2, :), 'LineWidth', 1);
    hold on

    poly_Xs = [data_avg_gp(1, :), flip(data_avg_gp(1, :))];
    poly_Ys = [data_avg_gp(2, :) - sdevs_gp(1, :), flip(data_avg_gp(2, :) + sdevs_gp(1, :))];

    pgon = polyshape(poly_Xs, poly_Ys);
    shadows = plot(pgon, 'FaceColor','black','FaceAlpha',0.05);
    shadows.LineStyle = 'none';
    axis([1 m y_min y_max])



    %--------------

    m=1;

    for i=1:size(lcs_opt, 1)
        m = max(m, size(lcs_opt{i}, 2));
    end

    data = zeros(3, m, 10);
    data_avg_opt = zeros(3, m);

    for i=1:size(lcs_opt, 1)
        data_ = lcs_opt{i};
        data(1, 1:m, i) = 1:m;
        data(2:end, 1:size(data_, 2), i) = data_(2:end, :);
        data(2, size(data_, 2):end, i) = data_(2, end);
        data(3, size(data_, 2):end, i) = data_(3, end);
        data_avg_opt = data_avg_opt + data(:, :, i)/10;
    end

    sdevs_opt = zeros(2, m);

    for i=1:2
        for j = 1:m
            sdevs_opt(i, j) = std(data(i+1, j, :));
        end
    end

    subplot(1, 4, 3)
    curve = plot(data_avg_opt(1, :), data_avg_opt(2, :), 'LineWidth', 1);
    hold on

    poly_Xs = [data_avg_opt(1, :), flip(data_avg_opt(1, :))];
    poly_Ys = [data_avg_opt(2, :) - sdevs_opt(1, :), flip(data_avg_opt(2, :) + sdevs_opt(1, :))];

    pgon = polyshape(poly_Xs, poly_Ys);
    shadows = plot(pgon, 'FaceColor','black','FaceAlpha',0.05);
    shadows.LineStyle = 'none';
    axis([1 m y_min y_max])



    %--------------
    m=1;

    for i=1:size(lcs_btm, 1)
        m = max(m, size(lcs_btm{i}, 2));
    end

    data = zeros(3, m, 10);
    data_avg_btm = zeros(3, m);

    for i=1:size(lcs_btm, 1)
        data_ = lcs_btm{i};
        data(1, 1:m, i) = 1:m;
        data(2:end, 1:size(data_, 2), i) = data_(2:end, :);
        data(2, size(data_, 2):end, i) = data_(2, end);
        data(3, size(data_, 2):end, i) = data_(3, end);
        data_avg_btm = data_avg_btm + data(:, :, i)/10;
    end

    sdevs_btm = zeros(2, m);

    for i=1:2
        for j = 1:m
            sdevs_btm(i, j) = std(data(i+1, j, :));
        end
    end

    subplot(1, 4, 1)
    curve = plot(data_avg_btm(1, :), data_avg_btm(2, :), 'LineWidth', 1);
    hold on

    poly_Xs = [data_avg_btm(1, :), flip(data_avg_btm(1, :))];
    poly_Ys = [data_avg_btm(2, :) - sdevs_btm(1, :), flip(data_avg_btm(2, :) + sdevs_btm(1, :))];

    pgon = polyshape(poly_Xs, poly_Ys);
    shadows = plot(pgon, 'FaceColor','black','FaceAlpha',0.05);
    shadows.LineStyle = 'none';
    axis([1 m y_min y_max])




    subplot(1, 4, 4)
    axis([1 m y_min y_max])

    bars_data = [data_avg_btm(2, end), data_avg_gp(2, end), data_avg_opt(2, end)];
    X = categorical({'BTM','GP','OPT'});

    bar(X, bars_data)

    ylim([y_min y_max])

end


%% Figure 2



if(plot_fig2 == true)
    
    y_min = 0.15;
    y_max = 0.4;


    m = 1;

    for i=1:size(lcs_gp, 1)
        m = max(m, size(lcs_gp{i}, 2));
    end

    data = zeros(3, m, 10);
    data_avg_gp = zeros(3, m);

    for i=1:size(lcs_gp, 1)
        data_ = lcs_gp{i};
        data(1, 1:m, i) = 1:m;
        data(2:end, 1:size(data_, 2), i) = data_(2:end, :);
        data(2, size(data_, 2):end, i) = data_(2, end);
        data(3, size(data_, 2):end, i) = data_(3, end);
        data_avg_gp = data_avg_gp + data(:, :, i)/10;
    end

    sdevs_gp = zeros(2, m);

    for i=1:2
        for j = 1:m
            sdevs_gp(i, j) = std(data(i+1, j, :));
        end
    end

    figure(2)
    set(gcf, 'Position', [10 10 1800 300])
    subplot(1, 4, 2)
    curve = plot(data_avg_gp(1, :), data_avg_gp(3, :), 'LineWidth', 1);
    hold on

    poly_Xs = [data_avg_gp(1, :), flip(data_avg_gp(1, :))];
    poly_Ys = [data_avg_gp(3, :) - sdevs_gp(2, :), flip(data_avg_gp(3, :) + sdevs_gp(2, :))];

    pgon = polyshape(poly_Xs, poly_Ys);
    shadows = plot(pgon, 'FaceColor','black','FaceAlpha',0.05);
    shadows.LineStyle = 'none';
    axis([1 m y_min y_max])



    %--------------

    m=1;

    for i=1:size(lcs_opt, 1)
        m = max(m, size(lcs_opt{i}, 2));
    end

    data = zeros(3, m, 10);
    data_avg_opt = zeros(3, m);

    for i=1:size(lcs_opt, 1)
        data_ = lcs_opt{i};
        data(1, 1:m, i) = 1:m;
        data(2:end, 1:size(data_, 2), i) = data_(2:end, :);
        data(2, size(data_, 2):end, i) = data_(2, end);
        data(3, size(data_, 2):end, i) = data_(3, end);
        data_avg_opt = data_avg_opt + data(:, :, i)/10;
    end

    sdevs_opt = zeros(2, m);

    for i=1:2
        for j = 1:m
            sdevs_opt(i, j) = std(data(i+1, j, :));
        end
    end

    subplot(1, 4, 3)
    curve = plot(data_avg_opt(1, :), data_avg_opt(3, :), 'LineWidth', 1);
    hold on

    poly_Xs = [data_avg_opt(1, :), flip(data_avg_opt(1, :))];
    poly_Ys = [data_avg_opt(3, :) - sdevs_opt(2, :), flip(data_avg_opt(3, :) + sdevs_opt(2, :))];

    pgon = polyshape(poly_Xs, poly_Ys);
    shadows = plot(pgon, 'FaceColor','black','FaceAlpha',0.05);
    shadows.LineStyle = 'none';
    axis([1 m y_min y_max])



    %--------------
    m=1;

    for i=1:size(lcs_btm, 1)
        m = max(m, size(lcs_btm{i}, 2));
    end

    data = zeros(3, m, 10);
    data_avg_btm = zeros(3, m);

    for i=1:size(lcs_btm, 1)
        data_ = lcs_btm{i};
        data(1, 1:m, i) = 1:m;
        data(2:end, 1:size(data_, 2), i) = data_(2:end, :);
        data(2, size(data_, 2):end, i) = data_(2, end);
        data(3, size(data_, 2):end, i) = data_(3, end);
        data_avg_btm = data_avg_btm + data(:, :, i)/10;
    end

    sdevs_btm = zeros(2, m);

    for i=1:2
        for j = 1:m
            sdevs_btm(i, j) = std(data(i+1, j, :));
        end
    end

    subplot(1, 4, 1)
    curve = plot(data_avg_btm(1, :), data_avg_btm(3, :), 'LineWidth', 1);
    hold on

    poly_Xs = [data_avg_btm(1, :), flip(data_avg_btm(1, :))];
    poly_Ys = [data_avg_btm(3, :) - sdevs_btm(2, :), flip(data_avg_btm(3, :) + sdevs_btm(2, :))];

    pgon = polyshape(poly_Xs, poly_Ys);
    shadows = plot(pgon, 'FaceColor','black','FaceAlpha',0.05);
    shadows.LineStyle = 'none';
    axis([1 m y_min y_max])




    subplot(1, 4, 4)
    axis([1 m y_min y_max])

    bars_data = [data_avg_btm(3, end), data_avg_gp(3, end), data_avg_opt(3, end)];
    X = categorical({'BTM','GP','OPT'});

    bar(X, bars_data)

    ylim([y_min y_max]) 
end

% figure(1)
% subplot(1, 3, 1)
% plot(data_avg_gp(1, :), data_avg_gp(2, :))
% axis([0 size(data_avg_gp(1, :), 2) 0.59 0.85])
% 
% subplot(1, 3, 2)
% plot(data_avg_btm(1, :), data_avg_btm(2, :))
% axis([0 size(data_avg_btm(1, :), 2) 0.59 0.85])
% 
% subplot(1, 3, 3)
% plot(data_avg_opt(1, :), data_avg_opt(2, :))
% axis([0 size(data_avg_opt(1, :), 2) 0.59 0.85])

close all; clear all; clc;

% Parameters

b = 1000;

lims = [[0.54 0.56];
        [0.74 0.76]];

l1 = min(lims(1, :));
u1 = max(lims(1, :));

l2 = min(lims(2, :));
u2 = max(lims(2, :));    

%% Generate new data or not?


generatingNew = false; % true for generating a new set of tasks; false for using the same tasks;

if generatingNew == true

    num_tasks = 10000;

    p = rand(2, num_tasks);
    perfs = zeros(1, num_tasks);

    trust_p = trust_(l1, u1, b, p(1,:)) .* trust_(l2, u2, b, p(2,:));

    for i = 1 : length(p)
        trust_pi = trust_p(i);
        tester = rand;
        if tester <= trust_pi
            perfs(i) = 1.0;
        end
    end

else
    
    load('./fixed_tasks_robotTrust.mat')
    num_tasks = 800; %5, 15, 400, 800. Max = 10000

    p = p_from_mat(:, 1:num_tasks);
    perfs = perfs_from_mat(:, 1:num_tasks);

end

%% Put the observed probabilities together

nbins = 10;
bin_lims = linspace(1/nbins, 1.0, nbins);
bin_lims_ = [0, bin_lims];
bin_centers = linspace(0.5/nbins, 1.0 - 0.5/nbins, nbins);

total_obs = zeros(nbins, nbins);
total_successes = zeros(nbins, nbins);

for i = 1 : length(p)
    for j = 1:nbins
        for k = 1:nbins
            if p(1, i) > bin_lims_(j) && p(1, i) <= bin_lims_(j+1)
                if p(2, i) > bin_lims_(k) && p(2, i) <= bin_lims_(k+1)
                    total_obs(j, k) = total_obs(j, k) + 1;
                    if perfs(i) == 1
                        total_successes(j, k) = total_successes(j, k) + 1;
                    end
                end
            end
        end
    end
end

observed_probs = total_successes ./ total_obs


% Save observed probabilities.

saving = true;

if saving
    save('./robotTrust_ObsProbs.mat', 'observed_probs', 'num_tasks');
end


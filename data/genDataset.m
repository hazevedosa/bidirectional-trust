clear all; close all;  clc;

raw_table = readtable('RawData.xlsx');

raw_table(1,:) = []; % if using macOS

% Text embeddings for the tasks

tasks_embeddings = zeros(4, 50);


% 'park, moving forward, in an empty space.'
tasks_embeddings(1, :) = ...
[ 3.2414117e-01  4.1447020e-01  4.3304004e-02  4.8868597e-02 ...
  4.9976879e-01 -5.7396363e-02 -6.7488801e-01 -3.1776690e-01 ...
 -6.3055411e-02 -2.8014469e-01  9.0402362e-05 -1.7512281e-01 ...
 -3.8491130e-01  8.3770128e-03  9.5973402e-02  1.6910079e-01 ...
  2.3971979e-01  1.4714999e-01 -6.8228096e-01 -2.3195300e-01 ...
  1.4912410e-01  1.8444458e-01 -1.6219600e-01 -6.7747109e-02 ...
  3.2402601e-02 -1.3294082e+00  1.4793441e-01  3.2838890e-01 ...
  2.6602936e-01 -2.4549556e-01  3.2139697e+00 -5.5527158e-02 ...
 -1.0209189e-01 -4.6102899e-01  6.3027114e-02  5.5950999e-02 ...
  1.4906329e-01  2.7490652e-01  2.2315459e-01  8.4912494e-02 ...
 -4.0771864e-02  9.7326681e-02  4.7370501e-02 -8.5684709e-02 ...
 -1.6837761e-01  1.7530830e-01 -5.9744291e-02 -4.0996504e-01 ...
  4.9381936e-03  5.7684919e-03];

% 'park, parallel to curb, in a space between cars.'
tasks_embeddings(2, :) = ...
[ 0.4099835   0.36084476  0.13827     0.09126333  0.16032267  0.16862215 ...
 -0.44250187 -0.46024978  0.03336776 -0.24009109 -0.05584915 -0.2033288 ...
 -0.40363052 -0.14415582  0.17571335  0.19522808  0.14247169  0.20761 ...
 -0.5584566  -0.368355    0.39317107 -0.05244559 -0.14226581  0.04407408 ...
 -0.01021872 -1.3762116   0.0656625   0.13323     0.2594942  -0.10531122 ...
  3.2551334   0.00811487 -0.23325968 -0.32749584  0.12381525 -0.05831984 ...
  0.02808992  0.1111024  -0.01662975  0.19176108 -0.19204605  0.04687556 ...
  0.12709801 -0.10007099 -0.10491458  0.06570274  0.01011558 -0.46474496 ...
  0.00325942 -0.18478699];

% 'when reaching a roundabout, check left for oncoming traffic and complete the right turn when safe.'
tasks_embeddings(3, :) = ...
[ 2.80031085e-01  6.97480589e-02  1.44717649e-01 -2.12731719e-01 ...
  1.92829311e-01  1.22776270e-01 -4.26211476e-01  1.03697889e-01 ...
  1.11459456e-01 -1.44328937e-01 -7.08094891e-03 -7.00216070e-02 ...
 -4.72601265e-01  5.89743331e-02  2.25088537e-01  1.13537565e-01 ...
  2.26994418e-02 -1.04376741e-01 -3.94146889e-01 -3.49649340e-01 ...
  1.25912726e-01  1.15610994e-01  2.78388127e-03  9.53572839e-02 ...
  2.14079663e-01 -1.35917842e+00  3.54975462e-02  2.48316079e-01 ...
  5.51716030e-01 -2.17567533e-01  2.85751581e+00  1.86755180e-01 ...
 -1.86410010e-01 -1.98157772e-01  5.02493754e-02  6.98378310e-02 ...
  2.16202080e-01  9.10104886e-02  1.05642103e-01  8.74429569e-02 ...
 -1.20017372e-01  1.13179259e-01  2.14564446e-02 -2.86201164e-02 ...
 -2.29960039e-01 -4.90290076e-02  6.38552234e-02 -2.58705944e-01 ...
  1.50818259e-01 -9.31316018e-02];

% 'when navigating on a two-way road behind a vehicle in foggy weather, check for oncoming traffic and pass when safe.'
tasks_embeddings(4, :) = ...
[ 2.5635621e-01  1.2729196e-01  1.5596174e-01 -5.8086503e-02 ...
  1.8803999e-03  2.9977530e-02 -6.2349457e-01 -4.6262216e-02 ...
  1.7178625e-01 -2.9206380e-01 -1.2491322e-01 -1.5295058e-01 ...
 -4.7805548e-01  7.3632821e-02  1.3588281e-01  9.1785319e-02 ...
 -6.2709562e-02 -7.1686752e-02 -5.3967685e-01 -4.6171650e-01 ...
  5.3738415e-02  2.0808376e-01 -6.6608265e-03  1.2337481e-01 ...
  2.2615390e-01 -1.4730545e+00 -7.6600066e-03  3.5499397e-01 ...
  5.9533715e-01 -1.1706673e-01  3.0174828e+00  1.7022167e-01 ...
 -2.1489652e-01 -1.8710987e-01  2.8041434e-01  1.0629254e-01 ...
  9.7192131e-02  1.1425366e-01  8.7957762e-02  1.4076504e-01 ...
 -2.0862235e-01  2.5998649e-01  1.3246487e-01 -6.0317264e-04 ...
 -9.5666647e-02 -5.7021160e-02  1.5941440e-01 -2.9321033e-01 ...
  3.6136732e-01 -1.2350861e-01];



num_responses = size(raw_table, 1);

obs_task_seq = [];
pred_task = [];
obs_task_sens_cap_seq = [];
obs_task_proc_cap_seq = [];
obs_task_perf_seq = [];
pred_task_sens_cap = [];
pred_task_proc_cap = [];
trust_pred = [];




for i = 1:num_responses
    
    participant_warning = false;
    
    participant_data = raw_table(i, :);
    participant_mturk_code = str2num(participant_data.mTurkCode{1});
    participant_prediction_task = str2num(participant_data.randNumber{1});
    participant_videos_order_raw = participant_data.FL_15_DO{1};
    participant_videos_order = [str2num(participant_videos_order_raw(19)), ...
                                str2num(participant_videos_order_raw(39)), ...
                                str2num(participant_videos_order_raw(59)), ...
                                str2num(participant_videos_order_raw(79))];
                            
    participant_videos_order = setdiff(participant_videos_order, participant_prediction_task, 'stable');
    
    participant_fail_succ = [str2num(participant_data.fail_succ_1{1}), ...
                             str2num(participant_data.fail_succ_2{1}), ...
                             str2num(participant_data.fail_succ_3{1}), ...
                             str2num(participant_data.fail_succ_4{1})];
    
    %% sensing check
    sensing_diff_order = [str2num(participant_data.Sens_1_1{1}), ...
                          str2num(participant_data.Sens_1_2{1}), ...
                          str2num(participant_data.Sens_1_3{1}), ...
                          str2num(participant_data.Sens_1_4{1})];
    
    sensing_capabilities = 0.1 * [str2num(participant_data.Sens_2_1{1}), ...
                                  str2num(participant_data.Sens_2_2{1}), ...
                                  str2num(participant_data.Sens_2_3{1}), ...
                                  str2num(participant_data.Sens_2_4{1})];
                        
    [~, sensing_diff_order_from_capabilities] = sort(sensing_capabilities);
    
    
    if any(sensing_diff_order - sensing_diff_order_from_capabilities) ~= 0
%         disp(i);
%         disp('Warning in Sensing -- possibly need to remove participant.');
%         participant_warning = true;
        if length(sensing_capabilities) ~= length(unique(sensing_capabilities))
%             disp('But there are equal sensing capabilities.');
        end        
    end    
    
    %% processing check
    processing_diff_order = [str2num(participant_data.Proc_1_1{1}), ...
                             str2num(participant_data.Proc_1_2{1}), ...
                             str2num(participant_data.Proc_1_3{1}), ...
                             str2num(participant_data.Proc_1_4{1})];
    
    processing_capabilities = 0.1 * [str2num(participant_data.Proc_2_1{1}), ...
                                     str2num(participant_data.Proc_2_2{1}), ...
                                     str2num(participant_data.Proc_2_3{1}), ...
                                     str2num(participant_data.Proc_2_4{1})];
                        
    [~, processing_diff_order_from_capabilities] = sort(processing_capabilities);
    
    
    if any(processing_diff_order - processing_diff_order_from_capabilities) ~= 0
%         disp(i);
%         disp('Warning in Processing -- possibly need to remove participant.');
%         participant_warning = true;
        if length(processing_capabilities) ~= length(unique(processing_capabilities))
%             disp('But there are equal processing capabilities.');
        end
    end

    participant_observed_task = [[0, 0, participant_videos_order(1)];
                                 [0, participant_videos_order(1), participant_videos_order(2)];
                                 [participant_videos_order(1), participant_videos_order(2), participant_videos_order(3)]];
                        
    participant_sensing_capabilities = [[0, 0, sensing_capabilities(participant_videos_order(1))];
                                        [0, sensing_capabilities(participant_videos_order(1)), sensing_capabilities(participant_videos_order(2))];
                                        [sensing_capabilities(participant_videos_order(1)), sensing_capabilities(participant_videos_order(2)), sensing_capabilities(participant_videos_order(3))]];
    
    participant_sens_cap_pred_task = [sensing_capabilities(participant_prediction_task);
                                      sensing_capabilities(participant_prediction_task);
                                      sensing_capabilities(participant_prediction_task)];
                                    
                                    
    participant_processing_capabilities = [[0, 0, processing_capabilities(participant_videos_order(1))];
                                           [0, processing_capabilities(participant_videos_order(1)), processing_capabilities(participant_videos_order(2))];
                                           [processing_capabilities(participant_videos_order(1)), processing_capabilities(participant_videos_order(2)), processing_capabilities(participant_videos_order(3))]];

    participant_proc_cap_pred_task = [processing_capabilities(participant_prediction_task);
                                      processing_capabilities(participant_prediction_task);
                                      processing_capabilities(participant_prediction_task)];
                                       
    participant_performances = zeros(3,3,2);
                                       
    participant_performances(:, :, 1) = [[0, 0, not(participant_fail_succ(participant_videos_order(1)))];
                                         [0, not(participant_fail_succ(participant_videos_order(1))), not(participant_fail_succ(participant_videos_order(2)))];
                                         [not(participant_fail_succ(participant_videos_order(1))), not(participant_fail_succ(participant_videos_order(2))), not(participant_fail_succ(participant_videos_order(3)))]];

    participant_performances(:, :, 2) = [[0, 0, participant_fail_succ(participant_videos_order(1))];
                                         [0, participant_fail_succ(participant_videos_order(1)), participant_fail_succ(participant_videos_order(2))];
                                         [participant_fail_succ(participant_videos_order(1)), participant_fail_succ(participant_videos_order(2)), participant_fail_succ(participant_videos_order(3))]];
    
    participant_obs_perf_order = [participant_fail_succ(participant_videos_order(1)), participant_fail_succ(participant_videos_order(2)), participant_fail_succ(participant_videos_order(3))];

    participant_trust_raw = {raw_table(i, 46:50);
                             raw_table(i, 51:55);
                             raw_table(i, 56:60);
                             raw_table(i, 61:65)};

    participant_prediction_task = [participant_prediction_task;
                                   participant_prediction_task;
                                   participant_prediction_task];    


    participant_trust_1 = participant_trust_raw{participant_videos_order(1)};
    
    for j = 1:2
        if strcmp(participant_trust_1{1, j}{1}, 'Yes')
            attChk_1 = 1;
        elseif strcmp(participant_trust_1{1, j}{1}, 'No')
            attChk_1 = 0;
        end
    end
    
    if attChk_1 == participant_fail_succ(participant_videos_order(1))
%         disp(i);
%         disp('Att Chk 1 OK');
    else
%         disp(i);
        disp('Att Chk 1 NOK');
        participant_warning = true;
    end
    
    for j = 3:5
        if ~strcmp(participant_trust_1{1, j}{1}, '')
            trust_prediction_1 = str2num(participant_trust_1{1, j}{1});
        end
    end
    
    
    
    participant_trust_2 = participant_trust_raw{participant_videos_order(2)};
    
    for j = 1:2
        if strcmp(participant_trust_2{1, j}{1}, 'Yes')
            attChk_2 = 1;
        elseif strcmp(participant_trust_2{1, j}{1}, 'No')
            attChk_2 = 0;
        end
    end
    
    if attChk_2 == participant_fail_succ(participant_videos_order(2))
%         disp(i);
%         disp('Att Chk 2 OK');
    else
%         disp(i);
        disp('Att Chk 2 NOK');
        participant_warning = true;
    end

    
    for j = 3:5
        if ~strcmp(participant_trust_2{1, j}{1}, '')
            trust_prediction_2 = str2num(participant_trust_2{1, j}{1});
        end
    end

    
    participant_trust_3 = participant_trust_raw{participant_videos_order(3)};

    
    for j = 1:2
        if strcmp(participant_trust_3{1, j}{1}, 'Yes')
            attChk_3 = 1;
        elseif strcmp(participant_trust_3{1, j}{1}, 'No')
            attChk_3 = 0;
        end
    end
    
    
    for j = 3:5
        if ~strcmp(participant_trust_2{1, j}{1}, '')
            trust_prediction_2 = str2num(participant_trust_2{1, j}{1});
        end
    end

    
    participant_trust_3 = participant_trust_raw{participant_videos_order(3)};

    
    for j = 1:2
        if strcmp(participant_trust_3{1, j}{1}, 'Yes')
            attChk_3 = 1;
        elseif strcmp(participant_trust_3{1, j}{1}, 'No')
            attChk_3 = 0;
        end
    end
    

    if attChk_3 == participant_fail_succ(participant_videos_order(3))
%         disp(i);
%         disp('Att Chk 3 OK');
    else
%         disp(i);
        disp('Att Chk 3 NOK');
        participant_warning = true;
    end
    
    for j = 3:5
        if ~strcmp(participant_trust_3{1, j}{1}, '')
            trust_prediction_3 = str2num(participant_trust_3{1, j}{1});
        end
    end
    
                                     
    participant_trust_predictions = [trust_prediction_1;
                                     trust_prediction_2;
                                     trust_prediction_3];
                                 
    participant_trust_predictions_sig = 1 ./ (1.0 + exp(-(participant_trust_predictions - 4.0)));
                                 
    if participant_warning == true
        disp('Participant NOK. MTurk Code:');
        disp(participant_mturk_code);
        pause;
        
    else
        disp('Participant OK. Mturk Code:');
        disp(participant_mturk_code);
    end
    

                                 
%% data concatenation...


    obs_task_seq = [obs_task_seq, participant_observed_task];
    pred_task = [pred_task; participant_prediction_task];
    obs_task_sens_cap_seq = [obs_task_sens_cap_seq, participant_sensing_capabilities];
    obs_task_proc_cap_seq = [obs_task_proc_cap_seq, participant_processing_capabilities];
    obs_task_perf_seq = [obs_task_perf_seq, participant_performances];
    pred_task_sens_cap = [pred_task_sens_cap; participant_sens_cap_pred_task];
    pred_task_proc_cap = [pred_task_proc_cap; participant_proc_cap_pred_task];
    trust_pred = [trust_pred; participant_trust_predictions_sig];
    
end

obs_task_feats = zeros(size(obs_task_seq, 1), size(obs_task_seq, 2), 50);

for m = 1:size(obs_task_seq, 1)
    for n = 1:size(obs_task_seq, 2)
        for p = 1:50
            if obs_task_seq(m, n) ~= 0
                obs_task_feats(m, n, p) = tasks_embeddings(obs_task_seq(m, n), p);
            end
        end
    end
end

pred_task_feats = zeros(size(pred_task, 1), size(pred_task, 2), 50);

for m = 1:size(pred_task, 1)
    for n = 1:size(pred_task, 2)
        for p = 1:50
            if pred_task(m, n) ~= 0
                pred_task_feats(m, n, p) = tasks_embeddings(pred_task(m, n), p);
            end
        end
    end
end


% Save the thing...

saving = true;

if saving
    save(...
        '../code/MatDataset.mat',...
                                              'obs_task_feats',...
                                           'obs_task_perf_seq',...
                                             'pred_task_feats',...
                                                  'trust_pred',...
                                                'obs_task_seq',...
                                                   'pred_task',...
                                                  'trust_pred',...
                                       'obs_task_sens_cap_seq',...
                                       'obs_task_proc_cap_seq',...
                                          'pred_task_sens_cap',...
                                          'pred_task_proc_cap' ...
         )
end




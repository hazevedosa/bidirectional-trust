# bidirectional-trust

Dataset and software for the paper "A Unified Bi-Directional Model for Natural and Artificial Trust in Human-Robot Collaboration"

trustmodels.py are from Soh and Chen's paper "multi-task transfer..."

## Instructions

### Humans' Natural Trust

1. run data/genDataset.m on MATLAB.
That will generate the MatDataset.mat file in the code folder.

2. 'cd code'; then run 'python3 trustExperiment.py <model>'; <model> = btm; gpMod; or opt

3. explore the results file "results_mat_<model>.mat" on MATLAB.


### Robots' Artificial Trust

1. Generate the sythentic data with robotTrust_dataGen.m

2. run python3 RobotTrustModel_2Dim.py

3. explore the results file 'results/resultsRobotTrust_2Dim.mat' on MATLAB

## TODO
Everything

Raw Dataset: file qualtricsRawData_8.xlsx

All videos used in the experiment are available at https://bit.ly/37gXXkI.

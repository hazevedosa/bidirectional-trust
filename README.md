# bidirectional-trust

Dataset and software for the paper "A Unified Bi-Directional Model for Natural and Artificial Trust in Human-Robot Collaboration".

## Dataset

The Raw dataset is found in data/RawData.xlsx. This dataset is processed with the genDataset.m MATLAB script, as in the Instructions below.


## Software

The Bi-directional trust model is implemented in two files:

* code/BidirectionalTrustModel.py, which contains the class BidirectoinalTrustModel; and
* code/RobotTrustModel_2Dim.py, which contains the class RobotTrustModel. This file also has a main function to test our Bi-directional trust model in the Artificial Trust mode.

Both classes extend PyTorch's nn.Module class.

The other models evaluated in the paper are implemented in the file trustmodels.py. This file is adapted from the models described in the paper "Multi-task trust transfer for human–robot interaction. The International Journal of Robotics Research 39.2-3 (2020): 233-249." (Soh, Harold, et al. 2020), available at 
https://github.com/crslab/human-trust-transfer.

## Use Instructions

### Humans' Natural Trust

1. run 'data/genDataset.m' on MATLAB.
That will generate the MatDataset.mat file in the code folder.

2. Run 'python3 trustExperiment.py <model>' from the 'code' dir, with <model> = btm; gpMod; or opt.
That will generate the file 'results/results_mat_<model>.mat'.

3. explore the results file 'results/results_mat_<model>.mat' on MATLAB.


### Robots' Artificial Trust

1. Run 'code/robotTrust_dataGen.m' on MATLAB.
That will generate the sythentic data for the Artificial Trust Mode simulation. Here line 41 can be changed to represent the value of N.

2. run 'python3 RobotTrustModel_2Dim.py' from the 'code' dir.
That will generate the file 'results/resultsRobotTrust_2Dim.mat'

3. explore the results file 'results/resultsRobotTrust_2Dim.mat' on MATLAB

### Paper Figures

With the results, data can be explored and figures can be generated with the scripts in paper/paperFigs/figs-code

### Experiment Videos

All videos used in the experiment are available at https://bit.ly/37gXXkI.

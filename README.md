# bidirectional-trust

Dataset and software for the paper "A Unified Bi-Directional Model for Natural and Artificial Trust in Human-Robot Collaboration".

## Dataset

The raw dataset is found in `data/RawData.xlsx`. This dataset is processed with the `genDataset.m` MATLAB script, as in the **Instructions** below.


## Software

### Dependencies

All implementations were tested with Python 3.8.5 and PyTorch v1.5.0.
The following packages are needed (please install with `python3 -m pip install --user <package name>`):

* `numpy`
* `torch`
* `pickle`
* `scipy`
* `sklearn`

### Model Implementation

The bi-directional trust model is implemented in two files:

* `code/BidirectionalTrustModel.py`, which contains the class BidirectionalTrustModel; and
* `code/RobotTrustModel_2Dim.py`, which contains the class RobotTrustModel. This file also has a main function to test our bi-directional trust model in the Artificial Trust mode.

Both classes extend PyTorch's `nn.Module` class.

The other models evaluated in the paper are implemented in the file `trustmodels.py`. This file is adapted from the models described in the paper "Multi-task trust transfer for human–robot interaction. The International Journal of Robotics Research 39.2-3 (2020): 233-249." (Soh, Harold, et al. 2020), available at 
https://github.com/crslab/human-trust-transfer.

## Use Instructions

### Humans' Natural Trust

1. Run `data/genDataset.m` on MATLAB.
  (That will generate the `MatDataset.mat` file in the `code` directory.)

2. Run `python3 trustExperiment.py <model>` from the `code` directory, with `<model> = btm`; `gpMod`; or `opt`.
  (That will generate the file `results/results_mat_<model>.mat`.)

3. Explore the results file `results/results_mat_<model>.mat` on MATLAB.


### Robots' Artificial Trust

1. Run `data/robotTrust_dataGen.m` on MATLAB.
  (That will generate the sythentic data for the Artificial Trust Mode simulation. Here, line _41_ can be changed to represent the value of _N_.)

2. Run `python3 RobotTrustModel_2Dim.py` from the `code` directory.
  (That will generate the file `results/resultsRobotTrust_2Dim.mat`.)

3. Explore the results file `results/resultsRobotTrust_2Dim.mat` on MATLAB

## Paper Figures

With the results, data can be explored and figures can be generated with the scripts in `paper/paperFigs/figs-code`.

## Experiment Videos

All videos used in the experiment are available at https://bit.ly/37gXXkI.

## Misc

The `paper` directory also contains the LaTeX source files for the paper. (TO BE CHANGED)

`survey/Qualtrics Survey.pdf` presents the Qualtrics survey taken by the Amazon Mechanical Turk workers.

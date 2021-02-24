# imports
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter

import numpy as np
from numpy.linalg import norm

import scipy.io as sio

import pickle

usecuda = True
usecuda = usecuda and torch.cuda.is_available()

dtype = torch.DoubleTensor

if usecuda:
    dtype = torch.cuda.FloatTensor



class RobotTrustModel(torch.nn.Module):

    def __init__(self):
        super(RobotTrustModel, self).__init__()

        # self.lambda_l = Parameter(dtype(np.zeros(1)))
        # self.lambda_u = Parameter(dtype(np.ones(1)))
        # self.beta = Parameter(dtype(20.0 * np.random.rand(1)))
        # self.beta = dtype([1000.0])

        self.pre_beta_1 = Parameter(dtype(4.0 * np.ones(1)), requires_grad=True)
        self.pre_beta_2 = Parameter(dtype(4.0 * np.ones(1)), requires_grad=True)

        self.pre_l_1 = Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True)
        self.pre_u_1 = Parameter(dtype( 10.0 * np.ones(1)), requires_grad=True)
        self.pre_l_2 = Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True)
        self.pre_u_2 = Parameter(dtype( 10.0 * np.ones(1)), requires_grad=True)



    def forward(self, bin_centers, obs_probs_idxs):


        n_diffs = obs_probs_idxs.shape[0]
        trust = torch.zeros(n_diffs)

        if(self.pre_l_1 > self.pre_u_1):
            buf = self.pre_l_1
            self.pre_l_1 = self.pre_u_1
            self.pre_u_1 = buf

        if(self.pre_l_2 > self.pre_u_2):
            buf = self.pre_l_2
            self.pre_l_2 = self.pre_u_2
            self.pre_u_2 = buf

        l_1 = self.sigm(self.pre_l_1)
        u_1 = self.sigm(self.pre_u_1)
        beta_1 = self.pre_beta_1 * self.pre_beta_1

        l_2 = self.sigm(self.pre_l_2)
        u_2 = self.sigm(self.pre_u_2)
        beta_2 = self.pre_beta_2 * self.pre_beta_2


        for i in range(n_diffs):
            bin_center_idx_1 = obs_probs_idxs[i, 0]
            bin_center_idx_2 = obs_probs_idxs[i, 1]
            trust[i] = self.compute_trust(l_1, u_1, beta_1, bin_centers[bin_center_idx_1]) * self.compute_trust(l_2, u_2, beta_2, bin_centers[bin_center_idx_2])


        return trust.cuda()

    def compute_trust(self, l, u, b, p):

        if b < -50:
            trust = 1.0 - 1.0 / (b * (u - l)) * torch.log( (1.0 + torch.exp(b * (p - l))) / (1.0 + torch.exp(b * (p - u))) )
        else:
            if p <= l:
                trust = torch.tensor([1.0])
            elif p > u:
                trust = torch.tensor([0.0])
            else:
                trust = (u - p) / (u - l + 0.0001)

        return trust.cuda()

    def sigm(self, x):
        return 1 / (1 + torch.exp(-x))




if __name__ == "__main__":
    

    model = RobotTrustModel()
    model.cuda()

    bin_c = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bin_c = dtype(bin_c)

    obs_probs_mat = sio.loadmat('../data/robotTrust_ObsProbs.mat')
    obs_probs = obs_probs_mat["observed_probs"]
    total_num_tasks = obs_probs_mat["num_tasks"]

    obs_probs_idxs = []
    for i in range(obs_probs.shape[0]):
        for j in range(obs_probs.shape[1]):
            if np.isnan(obs_probs[i, j]) == False:
                obs_probs_idxs += [[i, j]]

    obs_probs_idxs = np.array(obs_probs_idxs)


    obs_probs_vect = []
    for i in range(obs_probs_idxs.shape[0]):
        obs_probs_vect += [obs_probs[obs_probs_idxs[i, 0], obs_probs_idxs[i, 1]]]



    obs_probs = dtype(obs_probs)
    obs_probs_vect = dtype(obs_probs_vect)

    learning_rate = 0.01
    weight_decay = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_tolerance = 0.0005

    t = 0
    report_period = 100

    l_1 = []
    u_1 = []
    l_2 = []
    u_2 = []
    tt = []
    loss_to_save = []


    while t < 1520:

        def closure():
            diff = model(bin_c, obs_probs_idxs) - obs_probs_vect
            loss = torch.mean( torch.pow(diff, 2.0) )

            optimizer.zero_grad()

            loss.backward()

            return loss

        optimizer.step(closure)

        l1 = model.sigm(model.pre_l_1)
        u1 = model.sigm(model.pre_u_1)
        l2 = model.sigm(model.pre_l_2)
        u2 = model.sigm(model.pre_u_2)
        ll = torch.mean( torch.pow( (model(bin_c, obs_probs_idxs) - obs_probs_vect), 2.0 ) )

        l_1 += [l1.item()]
        u_1 += [u1.item()]
        l_2 += [l2.item()]
        u_2 += [u2.item()]

        tt += [t]

        loss_to_save += [ll.item()]

        if loss_to_save[-1] < loss_tolerance:
            break


        if t % report_period == 0:
            print("\nt =", tt[-1])

            print("l_1 =", l_1[-1])
            print("u_1 =", u_1[-1])

            print("l_2 =", l_2[-1])
            print("u_2 =", u_2[-1])

            print("\nloss", loss_to_save[-1])

        t = t + 1



    res_dict = {"l_1": l_1, "u_1": u_1, "l_2": l_2, "u_2": u_2, "tt": tt, "loss": loss_to_save, "total_num_tasks": total_num_tasks[0][0]}
    res_mat_file_name = "results/resultsRobotTrust_2Dim.mat"
    sio.savemat(res_mat_file_name, res_dict)
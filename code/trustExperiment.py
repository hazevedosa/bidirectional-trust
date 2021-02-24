#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter

import csv
import os 

import numpy as np
from numpy.linalg import norm
from numpy import pi, sign, fabs, genfromtxt


from scipy.special import gamma
import scipy.stats as stats
from sklearn.manifold import TSNE
import sklearn.metrics as metrics
from sklearn.decomposition import PCA

import scipy.io as sio

import spacy
from spacy.language import Language

import time
import sys

import pickle

from trustmodels import *
from BidirectionalTrustModel import *

import matplotlib.pyplot as plt

# some globals
usecuda = True # change this to True if you want cuda
usecuda = usecuda and torch.cuda.is_available()
dtype = torch.FloatTensor
if usecuda:
    dtype = torch.cuda.FloatTensor


npcafeats = 4
pca = PCA(n_components=npcafeats)
dirname = os.path.realpath('..')


def createDataset_fromMatFile(mat_file_name):
    
    mat_contents = sio.loadmat(mat_file_name)

    tasksobsfeats   = mat_contents["obs_task_feats"]
    tasksobsperf    = mat_contents["obs_task_perf_seq"]
    taskspredfeats  = mat_contents["pred_task_feats"]
    trustpred       = mat_contents["trust_pred"]
    tasksobsids     = mat_contents["obs_task_seq"]
    taskpredids     = mat_contents["pred_task"]
    taskpredtrust   = mat_contents["trust_pred"]
    matTasks        = mat_contents["obs_task_seq"]
    matTaskPredIDs  = mat_contents["pred_task"]
    data_labels     = ['Gamma-0', 'Gamma-1', 'Gamma-2', 'Gamma-3', 'Gamma-4']
    obs_task_sens_cap_seq = mat_contents["obs_task_sens_cap_seq"]
    obs_task_proc_cap_seq = mat_contents["obs_task_proc_cap_seq"]
    pred_task_sens_cap = mat_contents["pred_task_sens_cap"]
    pred_task_proc_cap = mat_contents["pred_task_proc_cap"]


    trustpred = np.squeeze(trustpred)
    taskspredfeats = np.squeeze(taskspredfeats)
    matTasks = matTasks.T
    tasksobsids = np.expand_dims(tasksobsids, axis=2)
    
    dataset = (
               tasksobsfeats,           # (3, 63, 50)   [numpy.ndarray]
               tasksobsperf,            # (3, 63, 2)    [numpy.ndarray]
               taskspredfeats,          # (63, 50)      [numpy.ndarray]
               trustpred,               # (63,)         [numpy.ndarray]
               tasksobsids,             # (3, 63, 1)    [numpy.ndarray]
               taskpredids,             # (63, 1)       [numpy.ndarray]
               taskpredtrust,           # (63, 1)       [numpy.ndarray]
               matTasks,                # (63, 3)       [numpy.ndarray]
               matTaskPredIDs,          # (63, 1)       [numpy.ndarray]
               data_labels,             # ???????       [list]
               obs_task_sens_cap_seq,   # (3, 63)       [numpy.ndarray]
               obs_task_proc_cap_seq,   # (3, 63)       [numpy.ndarray]
               pred_task_sens_cap,      # (63, 1)       [numpy.ndarray]
               pred_task_proc_cap,      # (63, 1)       [numpy.ndarray]
    )

    return dataset



def getTrainTestValSplit_fromMatFile(dataset, excludeid=None, pval=0.1, nfolds=10):
    tasksobsfeats, tasksobsperf, taskspredfeats, trustpred, tasksobsids, taskpredids, taskpredtrust, tasks_obs, tasks_pred, labels, \
    obs_task_sens_cap_seq, obs_task_proc_cap_seq, pred_task_sens_cap, pred_task_proc_cap = dataset


    obsseqlen = 3  # length of observation sequence
    predseqlen = 1  # length of prediction sequence

    # tasks_obs -- matrix of tasks that were observed
    # tasks_pred -- matrix of tasks that were predicted
    pretasks_pred = tasks_pred # matrix of tasks that were predicted
    nparts = tasksobsfeats.shape[1] # number of participants


    ntotalpred = trustpred.shape[0] # number of total predictions = 63
    ntotalprepred = int(np.prod(pretasks_pred.shape)) # 5 predicted trusts x 50 predictions = 250

    tasks_pred_T = tasks_pred.transpose().reshape([int(np.prod(tasks_pred.shape)), 1]) # rearrange in a column
    pretasks_pred_T = pretasks_pred.transpose().reshape([int(np.prod(pretasks_pred.shape)), 1]) # rearrange in a column


    trainids = []
    testids = []
    valids = []

        
    ntestparts = int(nparts/nfolds)
    partid = excludeid*ntestparts

    print("Num Test Participants: ", ntestparts)
    partids = [] #[partid, partid+1, partid+2]
    

    for i in range(ntestparts):
        partids += [partid + i]


    sidx = 0
    eidx = predseqlen 

    for partid in partids:
        for i in range(sidx, eidx):
            testids += [i * nparts + partid]



    trainids = np.setdiff1d(range(ntotalpred), testids)


    ntrain = len(trainids)
    nval = int(np.floor(pval * ntrain))       

    arr = np.arange(ntrain) # array range
    rids = np.random.permutation(arr)
    valids = trainids[rids[0:nval]]
    trainids = np.setdiff1d(trainids, valids)



    tasksobsfeats_train = tasksobsfeats[:, trainids, :]
    tasksobsperf_train = tasksobsperf[:, trainids, :]
    taskspredfeats_train = taskspredfeats[trainids, :]
    trustpred_train = trustpred[trainids]
    
    tasksobsids_train = tasksobsids[:, trainids, :]
    taskpredids_train = taskpredids[trainids, :]

    obs_task_sens_cap_seq_train = obs_task_sens_cap_seq[:, trainids]
    obs_task_proc_cap_seq_train = obs_task_proc_cap_seq[:, trainids]
    pred_task_sens_cap_train = pred_task_sens_cap[trainids, :]
    pred_task_proc_cap_train = pred_task_proc_cap[trainids, :]

    tasksobsfeats_val = tasksobsfeats[:, valids, :]
    tasksobsperf_val = tasksobsperf[:, valids, :]
    taskspredfeats_val = taskspredfeats[valids, :]
    trustpred_val = trustpred[valids]

    tasksobsids_val = tasksobsids[:, valids, :]
    taskpredids_val = taskpredids[valids, :]

    obs_task_sens_cap_seq_val = obs_task_sens_cap_seq[:, valids]
    obs_task_proc_cap_seq_val = obs_task_proc_cap_seq[:, valids]
    pred_task_sens_cap_val = pred_task_sens_cap[valids, :]
    pred_task_proc_cap_val = pred_task_proc_cap[valids, :]

    tasksobsfeats_test = tasksobsfeats[:, testids, :]
    tasksobsperf_test = tasksobsperf[:, testids, :]
    taskspredfeats_test = taskspredfeats[testids, :]
    trustpred_test = trustpred[testids]

    tasksobsids_test = tasksobsids[:, testids, :]
    taskpredids_test = taskpredids[testids, :]

    obs_task_sens_cap_seq_test = obs_task_sens_cap_seq[:, testids]
    obs_task_proc_cap_seq_test = obs_task_proc_cap_seq[:, testids]
    pred_task_sens_cap_test = pred_task_sens_cap[testids, :]
    pred_task_proc_cap_test = pred_task_proc_cap[testids, :]    



    expdata = {
        "tasksobsfeats_train": tasksobsfeats_train,
        "tasksobsperf_train": tasksobsperf_train,
        "taskspredfeats_train": taskspredfeats_train,
        "trustpred_train": trustpred_train,
        "tasksobsids_train": tasksobsids_train,
        "taskpredids_train": taskpredids_train,
        "obs_task_sens_cap_seq_train": obs_task_sens_cap_seq_train,
        "obs_task_proc_cap_seq_train": obs_task_proc_cap_seq_train,
        "pred_task_sens_cap_train": pred_task_sens_cap_train,
        "pred_task_proc_cap_train": pred_task_proc_cap_train,
        "tasksobsfeats_val": tasksobsfeats_val,
        "tasksobsperf_val": tasksobsperf_val,
        "taskspredfeats_val": taskspredfeats_val,
        "trustpred_val": trustpred_val,
        "tasksobsids_val": tasksobsids_val,
        "taskpredids_val": taskpredids_val,
        "obs_task_sens_cap_seq_val": obs_task_sens_cap_seq_val,
        "obs_task_proc_cap_seq_val": obs_task_proc_cap_seq_val,
        "pred_task_sens_cap_val": pred_task_sens_cap_val,
        "pred_task_proc_cap_val": pred_task_proc_cap_val,
        "tasksobsfeats_test": tasksobsfeats_test,
        "tasksobsperf_test": tasksobsperf_test,
        "taskspredfeats_test": taskspredfeats_test,
        "trustpred_test": trustpred_test,
        "tasksobsids_test": tasksobsids_test,
        "taskpredids_test": taskpredids_test,
        "obs_task_sens_cap_seq_test": obs_task_sens_cap_seq_test,
        "obs_task_proc_cap_seq_test": obs_task_proc_cap_seq_test,
        "pred_task_sens_cap_test": pred_task_sens_cap_test,
        "pred_task_proc_cap_test": pred_task_proc_cap_test,
        "labels": labels
    }

    return expdata



def main(
        reptype="wordfeat",
        excludeid=2,
        taskrepsize=50,
        modeltype="btm",
        pval=0.1,
        seed=0,
        nfolds=10
    ):
    
    
    modelname = modeltype + "_" + str(excludeid)

    usepriormean = False
    usepriorpoints = False

    verbose = False

    torch.manual_seed(seed)  # set up our seed for reproducibility
    np.random.seed(seed)

    mat_file_name = 'MatDataset.mat'
    dataset = createDataset_fromMatFile(mat_file_name)

    # create dataset splits
    expdata = getTrainTestValSplit_fromMatFile(dataset, excludeid=excludeid, pval=pval, nfolds=nfolds)
    

    nfeats = 50
    Ainit = None 


    inptasksobs = Variable(dtype(expdata["tasksobsfeats_train"]), requires_grad=False)
    inptasksperf = Variable(dtype(expdata["tasksobsperf_train"]), requires_grad=False)
    inptaskspred = Variable(dtype(expdata["taskspredfeats_train"]), requires_grad=False)
    outtrustpred = Variable(dtype(expdata["trustpred_train"]), requires_grad=False)

    tasksobsids = Variable(dtype(expdata["tasksobsids_train"]), requires_grad=False)
    taskspredids = Variable(dtype(expdata["taskpredids_train"]), requires_grad=False)

    obs_task_sens_cap_seq = Variable(dtype(expdata["obs_task_sens_cap_seq_train"]), requires_grad=False)
    pred_task_sens_cap = Variable(dtype(expdata["pred_task_sens_cap_train"]), requires_grad=False)
    obs_task_proc_cap_seq = Variable(dtype(expdata["obs_task_proc_cap_seq_train"]), requires_grad=False)
    pred_task_proc_cap = Variable(dtype(expdata["pred_task_proc_cap_train"]), requires_grad=False)

    inptasksobs_val = Variable(dtype(expdata["tasksobsfeats_val"]), requires_grad=False)
    inptasksperf_val = Variable(dtype(expdata["tasksobsperf_val"]), requires_grad=False)
    inptaskspred_val = Variable(dtype(expdata["taskspredfeats_val"]), requires_grad=False)
    outtrustpred_val = Variable(dtype(expdata["trustpred_val"]), requires_grad=False)

    tasksobsids_val = Variable(dtype(expdata["tasksobsids_val"]), requires_grad=False)
    taskspredids_val = Variable(dtype(expdata["taskpredids_val"]), requires_grad=False)

    obs_task_sens_cap_seq_val = Variable(dtype(expdata["obs_task_sens_cap_seq_val"]), requires_grad=False)
    pred_task_sens_cap_val = Variable(dtype(expdata["pred_task_sens_cap_val"]), requires_grad=False)
    obs_task_proc_cap_seq_val = Variable(dtype(expdata["obs_task_proc_cap_seq_val"]), requires_grad=False)
    pred_task_proc_cap_val = Variable(dtype(expdata["pred_task_proc_cap_val"]), requires_grad=False)


    inptasksobs_test = Variable(dtype(expdata["tasksobsfeats_test"]), requires_grad=False)
    inptasksperf_test = Variable(dtype(expdata["tasksobsperf_test"]), requires_grad=False)
    inptaskspred_test = Variable(dtype(expdata["taskspredfeats_test"]), requires_grad=False)
    outtrustpred_test = Variable(dtype(expdata["trustpred_test"]), requires_grad=False)

    tasksobsids_test = Variable(dtype(expdata["tasksobsids_test"]), requires_grad=False)
    taskspredids_test = Variable(dtype(expdata["taskpredids_test"]), requires_grad=False)

    obs_task_sens_cap_seq_test = Variable(dtype(expdata["obs_task_sens_cap_seq_test"]), requires_grad=False)
    pred_task_sens_cap_test = Variable(dtype(expdata["pred_task_sens_cap_test"]), requires_grad=False)
    obs_task_proc_cap_seq_test = Variable(dtype(expdata["obs_task_proc_cap_seq_test"]), requires_grad=False)
    pred_task_proc_cap_test = Variable(dtype(expdata["pred_task_proc_cap_test"]), requires_grad=False)

    learning_rate = 1e-2

    if modeltype == "gp":
        learning_rate = 1e-2
        usepriormean = usepriormean

        obsseqlen = 8

        phiinit = 1.0
        weight_decay = 0.01 #0.01
        modelparams = {
            "inputsize": inptasksobs.shape[2],
            "reptype": reptype,
            "taskrepsize": taskrepsize,
            "phiinit": phiinit,
            "Ainit": None,# np.array(Ainit),
            "obsseqlen": obsseqlen,
            "verbose": verbose,
            "usepriormean":usepriormean,
            "usepriorpoints":usepriorpoints
        }
    elif modeltype == "gpMod":
        learning_rate = 1e-2
        usepriormean = usepriormean

        obsseqlen = 3

        phiinit = 1.0
        weight_decay = 0.01 #0.01
        modelparams = {
            "inputsize": inptasksobs.shape[2],
            "reptype": reptype,
            "taskrepsize": taskrepsize,
            "phiinit": phiinit,
            "Ainit": None,# np.array(Ainit),
            "obsseqlen": obsseqlen,
            "verbose": verbose,
            "usepriormean":usepriormean,
            "usepriorpoints":usepriorpoints
        }
    elif modeltype == "btm":
        
        learning_rate = 1e-2
        obsseqlen = 3
        weight_decay = 0.01

        modelparams = {
            "inputsize": inptasksobs.shape[2],
            "taskrepsize": taskrepsize,
            "obsseqlen": obsseqlen,
            "verbose": verbose,
        }
    elif modeltype == "neural":
        perfrepsize = taskrepsize
        numGRUlayers = 2
        nperf = 2
        weight_decay = 0.00
        modelparams = {
            "perfrepsize": perfrepsize,
            "numGRUlayers": numGRUlayers,
            "nperf": nperf,
            "verbose": verbose,
            "taskrepsize": taskrepsize,
            "Ainit": None, #np.array(Ainit), 
            "nfeats": inptasksobs.shape[2]
        }
    elif modeltype == "opt":
        obsseqlen = 2
        weight_decay = 0.01
        modelparams = {
            "inputsize": inptasksobs.shape[2],
            "obsseqlen": obsseqlen,
        }
    elif modeltype == "constant":
        obsseqlen = 2
        weight_decay = 0.01
        modelparams = {
            "inputsize": inptasksobs.shape[2],
            "obsseqlen": obsseqlen,
        }
    else:
        raise ValueError("No such model")

    verbose = False
    reportperiod = 1
    
    # these two parameters control the early stopping
    # we save the stopcount-th model after the best validation is achived
    # but keep the model running for burnin longer in case a better
    # model is attained
    stopcount = 3
    burnin = 50
    
    t0 = time.time()
    bestvalloss = 1e10
    
    modeldir = "savedmodels"
    
    runOptimization = True

    if runOptimization:

        curve_data = []

        for rep in range(1):
            print("REP", rep)

            model = initModel(modeltype, modelname, parameters=modelparams)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            counter = 0

            torch.save(model, os.path.join(modeldir, model.modelname + ".pth"))
            restartopt = False
            t = 1

            t_vec = []
            loss_vec = []
            mae_vec = []

            while t < 5000:

                def closure():
                    N = inptaskspred.shape[0]

                    if modeltype == "btm":
                        predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0], tasksobsids, taskspredids, \
                             obs_task_sens_cap_seq, pred_task_sens_cap, obs_task_proc_cap_seq, pred_task_proc_cap)
                    else:
                        predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0])
                    predtrust = torch.squeeze(predtrust)


                    loss = torch.mean(torch.pow(predtrust - outtrustpred, 2.0))


                    optimizer.zero_grad()

                    if usepriorpoints:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()
                    return loss


                optimizer.step(closure)


                if t % reportperiod == 0:
                    # compute training loss
                    if modeltype == "btm":
                        predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0], tasksobsids, taskspredids, \
                                            obs_task_sens_cap_seq, pred_task_sens_cap, obs_task_proc_cap_seq, pred_task_proc_cap)
                    else:
                        predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0])

                    predtrust = torch.squeeze(predtrust)
                    loss = -(torch.dot(outtrustpred, torch.log(predtrust)) +
                            torch.dot((1 - outtrustpred), torch.log(1.0 - predtrust))) / inptaskspred.shape[0]

                    # compute validation loss
                    if modeltype == "btm":
                        predtrust_val = model(inptasksobs_val, inptasksperf_val, inptaskspred_val, inptasksobs_val.shape[0], tasksobsids_val, taskspredids_val, \
                        obs_task_sens_cap_seq_val, pred_task_sens_cap_val, obs_task_proc_cap_seq_val, pred_task_proc_cap_val)
                    else:
                        predtrust_val = model(inptasksobs_val, inptasksperf_val, inptaskspred_val, inptasksobs_val.shape[0])
                    predtrust_val = torch.squeeze(predtrust_val)
                    valloss = -(torch.dot(outtrustpred_val, torch.log(predtrust_val)) +
                                torch.dot((1 - outtrustpred_val), torch.log(1.0 - predtrust_val))) / predtrust_val.shape[0]

                    # compute prediction loss
                    if modeltype == "btm":
                        predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test, inptasksobs_test.shape[0], tasksobsids_test, taskspredids_test, \
                             obs_task_sens_cap_seq_test, pred_task_sens_cap_test, obs_task_proc_cap_seq_test, pred_task_proc_cap_test))
                        
                    else:
                        predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test, inptasksobs_test.shape[0]))
                    
                    predloss = -(torch.dot(outtrustpred_test, torch.log(predtrust_test)) +
                                torch.dot((1 - outtrustpred_test), torch.log(1.0 - predtrust_test))) / predtrust_test.shape[0]




                    #check for nans
                    checkval = np.sum(np.array(predtrust_test.cpu().data))
                    if np.isnan(checkval) or np.isinf(checkval):
                        # check if we have already restarted once
                        if restartopt:
                            #we've already done this, fail out.
                            #break out.
                            print("Already restarted once. Quitting")
                            break

                        # reinitialize model and switch optimizer
                        print("NaN value encountered. Restarting opt")
                        model = initModel(modeltype, modelname, parameters=modelparams)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                        t = 1
                        counter = 0
                        restartopt = True
                    else:
                        
                        mae = metrics.mean_absolute_error(predtrust_test.cpu().data, outtrustpred_test.cpu().data)

                        print("\nepoch: ", t, "loss: ", loss.cpu().data.item(), "valloss: ", valloss.cpu().data.item(),"predloss: ", predloss.cpu().data.item(),"mae: ", mae)
                        optimizer.zero_grad()
                        
                        # if validation loss has increased for stopcount iterations

                        augname = model.modelname + "_" + str(excludeid) + ".pth"
                        if valloss.cpu().data.item() <= bestvalloss:
                            torch.save(model, os.path.join(modeldir, augname) )
                            print("\nvalloss: ", valloss.cpu().data.item(), "bestvalloss: ", bestvalloss, "Model saved")
                            bestvalloss = valloss.cpu().data.item()
                            counter = 0
                        else:
                            if counter < stopcount and (valloss.cpu().data.item()-bestvalloss) <= 0.1:
                                torch.save(model, os.path.join(modeldir, augname))
                                print(valloss.cpu().data.item(), bestvalloss, "Model saved : POST", counter)
                            counter += 1

                t_vec += [t]
                loss_vec += [predloss.cpu().data.item()]
                mae_vec += [mae]

                if counter >= stopcount and t > burnin:
                    #torch.save(model, modeldir+ model.modelname + ".pth")
                    break

                t = t + 1

        curve_data = [t_vec, loss_vec, mae_vec]

    t1 = time.time()
    print("Total time: ", t1 - t0)


    model = torch.load(os.path.join(modeldir,  modelname + "_" + str(excludeid) + ".pth"))

    print("model params", list(model.parameters()))


    if modeltype == "btm":
        predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test, inptasksobs_test.shape[0], \
            tasksobsids_test, taskspredids_test, \
            obs_task_sens_cap_seq_test, pred_task_sens_cap_test, obs_task_proc_cap_seq_test, pred_task_proc_cap_test))
    else:
        predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test, inptasksobs_test.shape[0]))

    res = np.zeros((predtrust_test.shape[0], 2))
    res[:, 0] = predtrust_test.cpu().data[:]
    res[:, 1] = outtrustpred_test.cpu().data[:]


    mae = metrics.mean_absolute_error(predtrust_test.cpu().data, outtrustpred_test.cpu().data)
    predloss = -(torch.dot(outtrustpred_test, torch.log(predtrust_test)) + 
                    torch.dot((1 - outtrustpred_test), torch.log(1.0 - predtrust_test))) / predtrust_test.shape[0]
    predloss = predloss.cpu().data.item()

    return (mae, predloss, res, curve_data)



if __name__ == "__main__":

    reptype = "wordfeat"
    modeltype = sys.argv[1] # "btm" or "gp" or "gpMod" or "opt"
    taskrepsize = 50
    
    start = 0
    nfolds = 10
    end = nfolds
    pval = 0.15  # validation proportion


    allresults = []
    print(start, end)
    for excludeid in range(start, end):
        print("Test id:", excludeid)
        result = main(
            reptype=reptype,
            excludeid=excludeid,
            taskrepsize=taskrepsize,
            modeltype=modeltype,
            pval=pval,
            seed=0,
            nfolds=nfolds
        )
        allresults += [result]


        # save to disk after each run
        print("printing results: mae, predloss, predtrust_test --- outtrustpred_test\n\n")
        print(result)

        print("\n\n")
        
        print("MAEs")
        for i in range(len(allresults)):
            print(allresults[i][0])

        print("\n\nPredLosses")
        for i in range(len(allresults)):
            print(allresults[i][1])

        res_dict = {"allresults": allresults}
        res_mat_file_name = "results/results_mat_" + modeltype + ".mat"

        sio.savemat(res_mat_file_name, res_dict)
        


        resultsdir = "results"
        filename =  modeltype + ".pkl"
        with open(os.path.join(resultsdir, filename), 'wb') as output:
            pickle.dump(allresults, output, pickle.HIGHEST_PROTOCOL)

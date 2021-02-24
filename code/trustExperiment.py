#!/usr/bin/env python
# -*- coding: utf-8 -*-

# our imports
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

# include_prepreds = True             # THIS might change some stuff......................................
include_prepreds = False


npcafeats = 4
pca = PCA(n_components=npcafeats)
dirname = os.path.realpath('..')


def loadData(dom="household", bound01=True):
    # constants
    
    data_filenames = {
        # 'driving': os.path.join(dirname, 'data', 'trust_transfer_driving_cleaned.csv'), 
        'driving': os.path.join(dirname, 'data', 'trust_transfer_driving_cleaned__Mod.csv'), 
        'household': os.path.join(dirname, 'data', 'trust_transfer_household_cleaned.csv'),
    }

    # the task labels match those in the paper. 
    task_labels = {
        'driving': ['0-0', 'C-5', 'C-2', 'C-4', 'C-1', 'C-6', 'C-3', 'D-6', 'D-1', 'D-4', 'D-3', 'D-5', 'D-2'],
        'household': ['0-0', 'A-5', 'A-3', 'A-6', 'A-1', 'A-4', 'A-2', 'B-4', 'B-2', 'B-5', 'B-1', 'B-6', 'B-3'],
    }

    labels = task_labels[dom]
    nc = 2  # number of classes
    neg = 0  # negative class
    pos = 1  # positive class

    # task parameters
    ntasks = 12
    m = 2  # latent dimensionality

    # observations
    maxobs = 2  # maximum number of observations per person

    maxnpred = 3
    J = 7
    N = 0

    with open(data_filenames[dom]) as csvfile:

        reader = csv.DictReader(csvfile)

        i = 0

        pretasks_pred = []
        tasks_pred = []

        pretrust_scores = []
        trust_scores = []

        SSFF = []

        nobs = []
        tasks_obs = []
        tasks_obs_perf = []
        nprepred = []
        npred = []

        difficulties = []

        for row in reader:
            # these are the predicted tasks (before and observations)
            pretasks_pred += [[int(row['C_ID']), int(row['D_ID']), int(row['E_ID'])]]
            tasks_pred += [[int(row['C_ID']), int(row['D_ID']), int(row['E_ID'])]]

            # these are the scores before observations
            pretrust_scores += [[int(row['C1_rating']), int(row['D1_rating']), int(row['E1_rating'])]]
            trust_scores += [[int(row['C2_rating']), int(row['D2_rating']), int(row['E2_rating'])]]  

            difficulties += [[float(row['AD_e']), float(row['BD_e']), float(row['CD_e']), float(row['DD_e']), float(row['ED_e'])]]


            # this tracks if the robot suceeds or fails
            SSFF += row['B_SF']

            if row['B_SF'] == '1':  # success scenario (robot succeeds)
                # nobs += [[0,2]]
                tasks_obs += [[int(row['A_ID']), int(row['B_ID'])]]
                tasks_obs_perf += [[1, 1]]
            else:  # failure (robot fails)
                # nobs += [[2,0]]
                tasks_obs += [[int(row['A_ID']), int(row['B_ID'])]]
                tasks_obs_perf += [[0, 0]]

            nprepred += [3]
            npred += [3]
            N += 1
            i += 1

    # N=32 # number of participants
    tasks_obs_perf = np.array(tasks_obs_perf)
    tasks_obs = np.array(tasks_obs)
    tasks_pred = np.array(tasks_pred)
    trust_scores = np.array(trust_scores)

    difficulties = np.array(difficulties)

    pretasks_pred = np.array(pretasks_pred)
    pretrust_scores = np.array(pretrust_scores)
    nparts = N  # number of participants
    print("Num participants: %d" % (nparts))
    
    def sigmoid(x):
          return 1 / (1 + np.exp(-x))
    
    if bound01: 
        pretrust_scores = sigmoid(pretrust_scores-4.0)
        trust_scores = sigmoid(trust_scores-4.0)#(trust_scores - 1) / 6


    data = {
        "tasks_obs_perf": tasks_obs_perf,
        "tasks_obs": tasks_obs,
        "tasks_pred": tasks_pred,
        "trust_scores": trust_scores,
        "pretasks_pred": pretasks_pred,
        "pretrust_scores": pretrust_scores,
        "nparts": nparts,
        "labels": labels,
        "difficulties": difficulties
    }

    return data, nparts


def recreateWordVectors(vectors_loc="wordfeats/glove.6B/glove.6B.50d.txt", save_loc="wordfeats"):

    lang = "en"
    if lang is None:
        nlp = Language()
    else:
        nlp = spacy.blank(lang)
    with open(vectors_loc, 'rb') as file_:
        header = file_.readline()
        # nr_row, nr_dim = header.split()
        nr_dim = 50
        nlp.vocab.reset_vectors(width=int(nr_dim))
        for line in file_:
            line = line.rstrip().decode('utf8')
            pieces = line.rsplit(' ', int(nr_dim))
            word = pieces[0]
            vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
            nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab

    nlp.to_disk(save_loc)
    return


def loadWordFeatures(dom, loc="wordfeats", loadpickle=False, savepickle=False):
    if loadpickle:
        # we load saved word features from our pickle file
        # the word features were generated as below
        with open(os.path.join(dirname, 'data', 'wordfeatures.pkl'),'rb') as f:
            featdict = pickle.load(f)
        return featdict[dom]


    nlp = Language().from_disk(loc)


    taskwords = {
        'household': 
        [
            [' '],
            ['Pick and Place glass'],
            ['Pick and Place plastic can'],
            ['Pick and Place lemon'],
            ['Pick and Place plastic bottle'],
            ['Pick and Place apple'],
            ['Pick and Place plastic cup'],
            ['Navigate while avoiding moving people'],
            ['Navigate to the main room door'],
            ['Navigate while following a person'],
            ['Navigate to the dining table'],
            ['Navigate while avoiding obstacles'],
            ['Navigate to the living room']
        ],
        'driving':
        [
            [' '],
            ['Parking backwards cars and people around, misaligned'],
            ['Parking backwards empty lot, misaligned'],
            ['Parking backwards cars and people around, aligned'],
            ['Parking forwards empty lot, aligned'],
            ['Parking forwards cars and people around, misaligned'],
            ['Parking forwards empty lot, misaligned'],
            ['Navigating lane merge with other moving vehicles'],
            ['Navigating lane merge on a clear road'],
            ['Navigating traffic-circle with other moving vehicles'],
            ['Navigating traffic-circle on a clear road'],
            ['Navigating T-junction with other moving vehicles'],
            ['Navigating T-junction on a clear road'],
        ]
    }

    featdict = {}

    for d,task_word_list in taskwords.items():
        wordfeatures = []
        for i in range(len(task_word_list)):
            # print("yeah thats mee", task_word_list[i][0])
            wordfeatures.append(nlp(task_word_list[i][0]).vector)

        wordfeatures = np.array(wordfeatures)
        featdict[d] = wordfeatures

    wordfeatures = featdict[dom]


    # save the data
    if savepickle:
        with open(os.path.join(dirname, 'data', 'wordfeatures.pkl'),'wb') as f:
            pickle.dump(featdict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return wordfeatures


def getInputRep(taskid, nfeats, reptype="1hot", feats=None):
    taskfeat = []
    if reptype == "1hot":
        taskfeat = np.zeros((nfeats))
        taskfeat[taskid] = 1
    elif reptype == "wordfeat":
        taskfeat = np.zeros((1, feats.shape[1]))
        taskfeat[:] = feats[taskid, :]
    elif reptype == "taskid":
        taskfeat = np.zeros((1, 1))
        taskfeat[:] = taskid
    elif reptype == "tsne":
        taskfeat = np.zeros((1, feats.shape[1]))
        taskfeat[:] = feats[taskid, :]
    elif reptype == "pca":
        taskfeat = np.zeros((1, npcafeats))
        taskfeat[:] = feats[taskid, :]
    else:
        print("ERROR!")
        raise NameError('no such reptype')
    return taskfeat


# transforms raw data into dataset usable by our models
# def createDataset(data, reptype, allfeatures):
#     # create dataset suitable for model
#     nperf = 2  # number of performance outcomes (e.g., 2 - success, failure)
#     obsseqlen = 2  # length of observation sequence
#     predseqlen = 3  # length of prediction sequence

#     tasks_obs_perf = data["tasks_obs_perf"]
#     tasks_obs = data["tasks_obs"]
#     tasks_pred = data["tasks_pred"]
#     trust_scores = data["trust_scores"]
#     pretasks_pred = data["pretasks_pred"]
#     pretrust_scores = data["pretrust_scores"]
#     nparts = data["nparts"]

#     difficulties = data["difficulties"]

#     if reptype == "1hot":
#         nfeats = 13
#     elif reptype == "taskid":
#         nfeats = 1
#     elif reptype == "wordfeat":
#         taskfeatures = allfeatures["wordfeat"]
#         nfeats = taskfeatures.shape[1]
#     elif reptype == "tsne":
#         nfeats = 3
#         taskfeatures = allfeatures["tsne"]
#     elif reptype == "pca":
#         nfeats = npcafeats
#         taskfeatures = allfeatures["pca"]
#     else:
#         raise ValueError("no such reptype")
    
#     ntasks = taskfeatures.shape[0]
#     print("num features:", nfeats)


#     # create 1-hot representation for tasks observed
#     N = nparts
#     tasksobsfeats = np.zeros((obsseqlen, nparts, nfeats))
#     tasksobsids = np.zeros((obsseqlen, nparts, 1))

#     for i in range(N):
#         for t in range(obsseqlen):
#             tasksobsids[t, i, :] = tasks_obs[i, t]
#             tasksobsfeats[t, i, :] = getInputRep(tasks_obs[i, t], nfeats, reptype=reptype, feats=taskfeatures)


#     tasksobsfeats = np.tile(tasksobsfeats, [1, predseqlen, 1])
#     tasksobsids = np.tile(tasksobsids, [1, predseqlen, 1])

#     tasksobsperf = np.zeros((obsseqlen, nparts, nperf))

#     for i in range(N):
#         for t in range(obsseqlen):
#             tasksobsperf[t, i, tasks_obs_perf[i, t]] = 1
#     tasksobsperf = np.tile(tasksobsperf, [1, predseqlen, 1])


#     difficulties_obs = np.zeros((obsseqlen, nparts, 1))

#     for i in range(N):
#         for t in range(obsseqlen):
#             difficulties_obs[t, i, :] = difficulties[i, t]
#     difficulties_obs = np.tile(difficulties_obs, [1, predseqlen, 1])



#     # create 1-hot representation for tasks to predict
#     ntotalpred = int(np.prod(tasks_pred.shape))
#     tasks_pred_T = tasks_pred.transpose().reshape([ntotalpred, 1])

#     difficulties_pred = difficulties[:, 2:].transpose().reshape([ntotalpred, 1])


#     taskspredfeats = np.zeros((ntotalpred, nfeats))
#     for t in range(ntotalpred):
#         taskspredfeats[t, :] = getInputRep(tasks_pred_T[t][0], nfeats, reptype=reptype, feats=taskfeatures)

#     trust_scores_T = trust_scores.transpose().reshape([ntotalpred, 1])
#     trustpred = np.zeros(ntotalpred)
#     for t in range(ntotalpred):
#         trustpred[t] = trust_scores_T[t][0]

#     taskpredids = tasks_pred_T
#     taskpredtrust = trust_scores_T


#     if include_prepreds:
#         pretasksobsids = np.zeros((obsseqlen, N, 1))
#         pretasksobsfeats = np.zeros((obsseqlen, N, nfeats))

#         pre_difficulties_obs = np.zeros((obsseqlen, N, 1))

#         pretasksobsids = np.tile(pretasksobsids, [1, predseqlen, 1])
#         pretasksobsfeats = np.tile(pretasksobsfeats, [1, predseqlen, 1])

#         pre_difficulties_obs = np.tile(pre_difficulties_obs, [1, predseqlen, 1])

#         pretasksobsperf = np.zeros((obsseqlen, N, nperf))
#         pretasksobsperf = np.tile(pretasksobsperf, [1, predseqlen, 1])

#         # create 1-hot representation for pre-observation tasks to predict
#         ntotalprepred = int(np.prod(pretasks_pred.shape))
#         pretasks_pred_T = pretasks_pred.transpose().reshape([ntotalprepred, 1])
#         pretaskspredfeats = np.zeros((ntotalprepred, nfeats))
#         for t in range(ntotalprepred):
#             pretaskspredfeats[t, :] = getInputRep(pretasks_pred_T[t][0], nfeats, reptype=reptype, feats=taskfeatures)

#         pretrust_scores_T = pretrust_scores.transpose().reshape([ntotalprepred, 1])
#         pretrustpred = np.zeros(ntotalprepred)
#         for t in range(ntotalprepred):
#             pretrustpred[t] = pretrust_scores_T[t][0]

#         # create merged dataset
#         tasksobsfeats = np.column_stack([pretasksobsfeats, tasksobsfeats])
#         tasksobsperf = np.column_stack([pretasksobsperf, tasksobsperf])
#         taskspredfeats = np.concatenate([pretaskspredfeats, taskspredfeats])
#         trustpred = np.concatenate([pretrustpred, trustpred])

#         tasksobsids = np.column_stack([pretasksobsids, tasksobsids])
#         taskpredids = np.concatenate([pretasks_pred_T, tasks_pred_T])
#         taskpredtrust = np.concatenate([pretrust_scores_T, trust_scores_T])

#         difficulties_obs = np.column_stack([pre_difficulties_obs, difficulties_obs])
#         difficulties_pred = np.concatenate([difficulties_pred, difficulties_pred])



#     # ok, I got too lazy to create a dict, using a tuple for now.

#     # So, here we have 192 or 186 dimensions long datasets. Basically they have stacked up the 1st ratings, 
#     # without observed tasks and 2nd ratings, with the observed tasks. The observed tasks in the first place are 0s --- there were no observed tasks at all...
#     # the observed tasks in the second half are those named A and B in the original dataset.
#     # I dont know why there is trustpred and taskpredtrust. they have basically the same data...


#     dataset = (
#                tasksobsfeats,       # [0] (2, 192, 50) or (2, 186, 50)  [numpy.ndarray]
#                tasksobsperf,        # [1] (2, 192, 2) or (2, 186, 2)    [numpy.ndarray]
#                taskspredfeats,      # [2] (192, 50) or (186, 50)        [numpy.ndarray]
#                trustpred,           # [3] (192,) or (186,)              [numpy.ndarray]
#                tasksobsids,         # [4] (2, 192, 1) or (2, 186, 1)    [numpy.ndarray]
#                taskpredids,         # [5] (192, 1) or (186, 1)          [numpy.ndarray]
#                taskpredtrust,       # [6] (192, 1) or (186, 1)          [numpy.ndarray]
#                data["labels"],      # [7] ['0-0', 'A-5', 'A-3', 'A-6', 'A-1', 'A-4', 'A-2', 'B-4', 'B-2', 'B-5', 'B-1', 'B-6', 'B-3'] for household [list]
#                difficulties_obs,    # [8] difficulties of observed tasks
#                difficulties_pred    # [9] difficulties of tasks to be predicted
#               )



#     return dataset


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

# def createDataset_fromMatFile_(mat_file_name):
    
#     mat_contents = sio.loadmat(mat_file_name)

#     tasksobsfeats   = mat_contents["tasksobsfeats"]
#     tasksobsperf    = mat_contents["tasksobsperf"]
#     taskspredfeats  = mat_contents["taskspredfeats"]
#     trustpred       = mat_contents["trustpred"]
#     tasksobsids     = mat_contents["tasksobsids"]
#     taskpredids     = mat_contents["taskpredids"]
#     taskpredtrust   = mat_contents["taskpredtrust"]
#     matTasks        = mat_contents["matTasks"]
#     matTaskPredIDs  = mat_contents["matTaskPredIDs"]
#     data_labels     = ['0-0', 'H-1', 'H-2', 'H-3', 'H-4', 'H-5']

#     trustpred = np.squeeze(trustpred)
#     tasksobsids = np.expand_dims(tasksobsids, axis=2)
    
#     dataset = (
#                tasksobsfeats,   # (51, 255, 50) [numpy.ndarray]
#                tasksobsperf,    # (51, 255, 2)  [numpy.ndarray]
#                taskspredfeats,  # (255, 50)     [numpy.ndarray]
#                trustpred,       # (255,)        [numpy.ndarray]
#                tasksobsids,     # (51, 255, 1)  [numpy.ndarray]
#                taskpredids,     # (255, 1)      [numpy.ndarray]
#                taskpredtrust,   # (255, 1)      [numpy.ndarray]
#                matTasks,        # (51, 51)      [numpy.ndarray]
#                matTaskPredIDs,  # (55,  5)      [numpy.ndarray]
#                data_labels      # ????????????  [list]
#     )

#     return dataset

# def computeTSNEFeatures(wordfeatures):
#     tsnefeatures = TSNE(n_components=3, perplexity=5).fit_transform(np.array(wordfeatures))
#     tsnefeatures = tsnefeatures / np.max(tsnefeatures) * 3
#     return tsnefeatures


# def computePCAFeatures(wordfeatures):
#     global npcafeats
#     pca = PCA(n_components=npcafeats)
#     pca.fit(wordfeatures)
#     return pca.transform(wordfeatures)


# pval is the validation proportion
# def getTrainTestValSplit(data, dataset, splittype, excludeid=None, pval=0.1, nfolds=10):
#     tasksobsfeats, tasksobsperf, taskspredfeats, trustpred, tasksobsids, taskpredids, taskpredtrust, labels, difficulties_obs, difficulties_pred = dataset

#     obsseqlen = 2  # length of observation sequence
#     predseqlen = 3  # length of prediction sequence

#     # tasks_obs_perf = data["tasks_obs_perf"]
#     tasks_obs = data["tasks_obs"]   # matrix of tasks that were observed (32, 2)
#     tasks_pred = data["tasks_pred"] # matrix of tasks that were predicted (32, 3)
#     # trust_scores = data["trust_scores"]
#     pretasks_pred = data["pretasks_pred"] # matrix of tasks that were predicted (32, 3)
#     # pretrust_scores = data["pretrust_scores"]
#     nparts = data["nparts"] # number of participants

#     ntotalpred = trustpred.shape[0]
#     ntotalprepred = int(np.prod(pretasks_pred.shape))
#     tasks_pred_T = tasks_pred.transpose().reshape([int(np.prod(tasks_pred.shape)), 1])

#     # trust_scores_T = trust_scores.transpose().reshape([ntotalpred, 1])
#     # pretrust_scores_T = pretrust_scores.transpose().reshape([ntotalprepred, 1])
#     pretasks_pred_T = pretasks_pred.transpose().reshape([int(np.prod(pretasks_pred.shape)), 1])

#     trainids = []
#     testids = []
#     valids = []

#     if splittype == "random":
#         # Random splits
#         # split into test and train set
#         ntrain = int(np.floor(0.9 * ntotalpred))
#         rids = np.random.permutation(ntotalpred)
#         trainids = rids[0:ntrain]
#         testids = rids[ntrain + 1:]

#         nval = int(np.floor(pval * ntrain))
#         valids = trainids[0:nval]
#         trainids = np.setdiff1d(trainids, valids)

#     elif splittype == "3participant":
        
#         ntestparts = int(nparts/nfolds)
#         partid = excludeid*ntestparts
#         print("Num Test Participants: ", ntestparts)
#         partids = [] #[partid, partid+1, partid+2]
        
#         for i in range(ntestparts):
#             partids += [partid + i]
        
#         # ridx = np.random.permutation(nparts)
#         # for i in range(ntestparts):
#         #     partids += [ridx[i]]        
    
#         if include_prepreds:
#             sidx = 0
#             eidx = predseqlen * 2
#         else:
#             sidx = 0
#             eidx = predseqlen 
            
#         for partid in partids:
#             for i in range(sidx, eidx):
#                 testids += [i * nparts + partid]

#         trainids = np.setdiff1d(range(ntotalpred), testids)

#         ntrain = len(trainids)
#         nval = int(np.floor(pval * ntrain))
#         arr = np.arange(ntrain)
#         rids = np.random.permutation(arr)
#         valids = trainids[rids[0:nval]]
#         # print("valids", valids)
#         trainids = np.setdiff1d(trainids, valids)
#         # print(trainids)        
#     elif splittype == "LOOtask":
#         # note that task ids range from 1 to nparts-1
#         # remove all participants who observed the task

#         taskid = excludeid
#         print(labels[excludeid])
#         partids = []
#         testids = []
#         for i in range(nparts):
#             for t in range(obsseqlen):
#                 if tasks_obs[i, t] == taskid:
#                     partids += [i]

#         preshapesize = 0
#         if include_prepreds:
#             preshapesize = pretasks_pred_T.shape[0]
                    
#         if include_prepreds:
#             for partid in partids:
#                 for i in range(predseqlen):
#                     testids += [i * nparts + partid + preshapesize]

#         # remove all training samples where the prediction (pre and post were the task)
#         if include_prepreds:
#             for i in range(pretasks_pred_T.shape[0]):
#                 if pretasks_pred_T[i] == taskid:
#                     testids += [i]

        
#         for i in range(tasks_pred_T.shape[0]):
#             if tasks_pred_T[i] == taskid:
#                 testids += [preshapesize + i]  # adding the size of pretasks because we concatenate the vectors
                    
        
#         testids = np.sort(np.unique(testids))
#         trainids = np.setdiff1d(range(ntotalpred), testids)
#         ntrain = len(trainids)
#         nval = int(np.floor(pval * ntrain))
#         rids = np.random.permutation(ntrain)
#         valids = trainids[rids[0:nval]]
        
        
#         trainids = np.setdiff1d(trainids, valids)



#     tasksobsfeats_train = tasksobsfeats[:, trainids, :]
#     tasksobsperf_train = tasksobsperf[:, trainids, :]
#     taskspredfeats_train = taskspredfeats[trainids, :]
#     trustpred_train = trustpred[trainids]
    
#     tasksobsids_train = tasksobsids[:, trainids, :]
#     taskpredids_train = taskpredids[trainids, :]

#     difficulties_obs_train = difficulties_obs[:, trainids, :]
#     difficulties_pred_train = difficulties_pred[trainids, :]

#     tasksobsfeats_val = tasksobsfeats[:, valids, :]
#     tasksobsperf_val = tasksobsperf[:, valids, :]
#     taskspredfeats_val = taskspredfeats[valids, :]
#     trustpred_val = trustpred[valids]

#     tasksobsids_val = tasksobsids[:, valids, :]
#     taskpredids_val = taskpredids[valids, :]

#     difficulties_obs_val = difficulties_obs[:, valids, :]
#     difficulties_pred_val = difficulties_pred[valids, :]

#     tasksobsfeats_test = tasksobsfeats[:, testids, :]
#     tasksobsperf_test = tasksobsperf[:, testids, :]
#     taskspredfeats_test = taskspredfeats[testids, :]
#     trustpred_test = trustpred[testids]

#     tasksobsids_test = tasksobsids[:, testids, :]
#     taskpredids_test = taskpredids[testids, :]

#     difficulties_obs_test = difficulties_obs[:, testids, :]
#     difficulties_pred_test = difficulties_pred[testids, :]
    

#     expdata = {
#         "tasksobsfeats_train": tasksobsfeats_train,
#         "tasksobsperf_train": tasksobsperf_train,
#         "taskspredfeats_train": taskspredfeats_train,
#         "trustpred_train": trustpred_train,
#         "tasksobsids_train": tasksobsids_train,
#         "taskpredids_train": taskpredids_train,
#         "difficulties_obs_train": difficulties_obs_train,
#         "difficulties_pred_train": difficulties_pred_train,
#         "tasksobsfeats_val": tasksobsfeats_val,
#         "tasksobsperf_val": tasksobsperf_val,
#         "taskspredfeats_val": taskspredfeats_val,
#         "trustpred_val": trustpred_val,
#         "tasksobsids_val": tasksobsids_val,
#         "taskpredids_val": taskpredids_val,
#         "difficulties_obs_val": difficulties_obs_val,
#         "difficulties_pred_val": difficulties_pred_val,
#         "tasksobsfeats_test": tasksobsfeats_test,
#         "tasksobsperf_test": tasksobsperf_test,
#         "taskspredfeats_test": taskspredfeats_test,
#         "trustpred_test": trustpred_test,
#         "tasksobsids_test": tasksobsids_test,
#         "taskpredids_test": taskpredids_test,
#         "difficulties_obs_test": difficulties_obs_test,
#         "difficulties_pred_test": difficulties_pred_test,
#         "labels": data["labels"]
#     }

#     return expdata


def getTrainTestValSplit_fromMatFile(dataset, splittype, excludeid=None, pval=0.1, nfolds=10):
    tasksobsfeats, tasksobsperf, taskspredfeats, trustpred, tasksobsids, taskpredids, taskpredtrust, tasks_obs, tasks_pred, labels, \
    obs_task_sens_cap_seq, obs_task_proc_cap_seq, pred_task_sens_cap, pred_task_proc_cap = dataset


    obsseqlen = 3  # length of observation sequence
    predseqlen = 1  # length of prediction sequence

    # tasks_obs -- matrix of tasks that were observed (32, 2) // (63, 3)
    # tasks_pred -- matrix of tasks that were predicted (32, 3) // (63, 1)
    pretasks_pred = tasks_pred # matrix of tasks that were predicted (32, 3) // (63, 1)
    nparts = tasksobsfeats.shape[1] # number of participants?????????


    ntotalpred = trustpred.shape[0] # number of total predictions = 63
    ntotalprepred = int(np.prod(pretasks_pred.shape)) # 5 predicted trusts x 50 predictions = 250

    tasks_pred_T = tasks_pred.transpose().reshape([int(np.prod(tasks_pred.shape)), 1]) # rearrange in a column
    pretasks_pred_T = pretasks_pred.transpose().reshape([int(np.prod(pretasks_pred.shape)), 1]) # rearrange in a column


    trainids = []
    testids = []
    valids = []

    if splittype == "random":
        # Random splits
        # split into test and train set
        ntrain = int(np.floor(0.9 * ntotalpred))
        rids = np.random.permutation(ntotalpred)
        trainids = rids[0:ntrain]
        testids = rids[ntrain + 1:]

        nval = int(np.floor(pval * ntrain))
        valids = trainids[0:nval]
        trainids = np.setdiff1d(trainids, valids)

    elif splittype == "3participant":
        
        ntestparts = int(nparts/nfolds)
        partid = excludeid*ntestparts

        print("Num Test Participants: ", ntestparts)
        partids = [] #[partid, partid+1, partid+2]
        

        for i in range(ntestparts):
            partids += [partid + i]


        # ridx = np.random.permutation(nparts)
        # for i in range(ntestparts):
        #     partids += [ridx[i]]        



        if include_prepreds: # i am not sure this is needed...
            sidx = 0
            eidx = predseqlen * 2
        else:
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

    elif splittype == "LOOtask":
        # note that task ids range from 1 to nparts-1
        # remove all participants who observed the task

        taskid = excludeid
        print(labels[excludeid])
        partids = []
        testids = []

        for i in range(nparts):
            for t in range(obsseqlen):
                if tasks_obs[i, t] == taskid:
                    partids += [i]

        preshapesize = 0
        if include_prepreds:
            preshapesize = pretasks_pred_T.shape[0]
                    
        if include_prepreds:
            for partid in partids:
                for i in range(predseqlen):
                    testids += [i * nparts + partid + preshapesize]

        # remove all training samples where the prediction (pre and post were the task)
        if include_prepreds:
            for i in range(pretasks_pred_T.shape[0]):
                if pretasks_pred_T[i] == taskid:
                    testids += [i]

        
        for i in range(tasks_pred_T.shape[0]):
            if tasks_pred_T[i] == taskid:
                testids += [preshapesize + i]  # adding the size of pretasks because we concatenate the vectors
                    
        
        testids = np.sort(np.unique(testids))
        trainids = np.setdiff1d(range(ntotalpred), testids)
        ntrain = len(trainids)
        nval = int(np.floor(pval * ntrain))
        rids = np.random.permutation(ntrain)
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


# Utility function: to plot the task representations
def plotEmbeddings(m, labels, reptype, feats, use_tsne=True):
    ntasks = len(labels)
    nfeats = feats.shape[0]
    taskreps = m.getTaskEmbeddings(nfeats, reptype=reptype, feats=feats)
    # print(taskreps)
    # print(taskreps.shape)
    taskreps = taskreps[1:, :]
    use_tsne = use_tsne
    if use_tsne:
        taskreps2d = TSNE(n_components=2, perplexity=2).fit_transform(taskreps)
    else:
        taskreps2d = taskreps

    fig, ax = plt.subplots()
    plt.scatter(taskreps2d[:, 0], taskreps2d[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')

    for i, txt in enumerate(labels[1:]):
        ax.annotate(txt, (taskreps2d[i, 0], taskreps2d[i, 1]))

    return taskreps2d


def getInitialProjectionMatrix(taskfeatures, reptype, taskrepsize, doplot=False, labels=None):
    # use dists to compute pre latent positions

    ntasks, nfeats = taskfeatures.shape

    class ProjMDS(torch.nn.Module):
        def __init__(self, nfeats, repsize=2):
            super(ProjMDS, self).__init__()
            # linearly transform one hot / features into a latent representation
            self.zrep = nn.Linear(nfeats, repsize, bias=False)
            # print(self.zrep.weight.shape)
            # self.zrep.weight = Parameter(torch.Tensor(np.random.rand(repsize,nfeats)*(1/50.)))
            # initz = np.random.rand(repsize,nfeats)
            # self.zrep.weight = Parameter(torch.Tensor(initz))

        def forward(self, pairs, taskfeats):
            N = pairs.shape[0]
            preddist = Variable(torch.FloatTensor(np.zeros((N, 1))))
            k = 0
            for pair in pairs:
                i = int(pair.cpu().data[0])
                j = int(pair.cpu().data[1])
                xi = taskfeats[i]
                xj = taskfeats[j]
                # print(i, j)
                # print(xi)
                zi = self.zrep(xi)
                zj = self.zrep(xj)
                # print('zi', zi)
                # print('zj', zj)
                dij = torch.norm(zi - zj)
                # print(dij)
                preddist[k] = dij
                k += 1

            return preddist

    # reduction to 2D
    # taskreps2d = TSNE(n_components=2, perplexity=2).fit_transform(np.array(taskreps.data))
    taskreps2d = TSNE(n_components=taskrepsize, perplexity=5).fit_transform(np.array(taskfeatures))
    dists = metrics.pairwise.pairwise_distances(taskreps2d / np.max(taskreps2d))
    # dists = metrics.pairwise.pairwise_distances(taskfeatures)

    featpairs = []
    featdists = []
    for i in range(ntasks):
        for j in range(i, ntasks):
            featpairs += [[i, j]]
            featdists += [dists[i, j]]

    featpairs = np.array(featpairs)
    featdists = np.array(featdists)
    featdists = np.power(featdists, 2.0)

    # first match locally to get the initial transformation matrix
    inppairs = Variable(torch.FloatTensor(featpairs), requires_grad=False)
    inpdistlist = Variable(torch.FloatTensor(featdists), requires_grad=False)

    alltasks1hot = np.zeros((ntasks, ntasks))
    for i in range(ntasks):
        alltasks1hot[i, i] = 1

    inpalltasks = None
    if reptype == "1hot":
        inpalltasks = Variable(torch.FloatTensor(alltasks1hot), requires_grad=False)
    elif reptype == "wordfeat" or reptype == "tsne":
        inpalltasks = Variable(torch.FloatTensor(taskfeatures), requires_grad=False)

    mds = ProjMDS(nfeats, taskrepsize)

    learning_rate = 1e-1
    optimizer = torch.optim.LBFGS(mds.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(gpmodel.parameters(), lr=learning_rate)

    t0 = time.time()

    for t in range(50):

        # logloss = model(inptasksobs, inptasksperf, inptaskspred, outtrustpred)
        def closure():
            optimizer.zero_grad()
            preddists = mds(inppairs, inpalltasks)
            loss = torch.mean(torch.pow(preddists - inpdistlist, 2.0))
            if usepriorpoints:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            return loss


        optimizer.step(closure)


        if t % 10 == 0:
            preddists = mds(inppairs, inpalltasks)
            loss = torch.mean(torch.pow(preddists - inpdistlist, 2.0))
            optimizer.zero_grad()
            print(t, loss.cpu().data.item())
            optimizer.zero_grad()

    t1 = time.time()
    print("Total time: ", t1 - t0)
    if doplot:

        taskreps = mds.zrep(inpalltasks)
        fig, ax = plt.subplots()
        plt.scatter(taskreps[:, 0], taskreps[:, 1])
        plt.gca().set_aspect('equal', adjustable='box')

        if labels is None:
            labels = [str(i) for i in range(13)]
        for i, txt in enumerate(labels):
            ax.annotate(txt, (taskreps[i, 0], taskreps[i, 1]))
        plt.show(block=True)

    return mds.zrep.weight.cpu().data

    
def getGPParams(mode):
    usepriormean = False
    usepriorpoints = False
    if mode == 1:
        usepriormean = True
    elif mode == 2:
        usepriorpoints = True
    elif mode == 3:
        usepriorpoints = True
        usepriormean = True
    else:
        print("No such mode. defaulting to regular gp")
    return (usepriormean,usepriorpoints)

def main(
        dom="driving",
        reptype="wordfeat",
        splittype="LOOtask",  
        excludeid=2,
        taskrepsize=2,
        modeltype="neural",
        gpmode = 0,
        pval=0.1,
        seed=0,
        nfolds=10
    ):
    
    
    modelname = modeltype + "_" + str(taskrepsize) + "_" + str(gpmode) + "_" + dom + "_" + splittype + "_" + str(excludeid)
    
    # check what kind of modifications to the GP we are using
    print("Modelname: ", modelname)
    usepriormean, usepriorpoints = getGPParams(gpmode)


    verbose = False

    torch.manual_seed(seed)  # set up our seed for reproducibility
    np.random.seed(seed)

    # load the data
    # data, nparts = loadData(dom)



    # recreate word vectors if needed
    # e.g., when you download new word features from glove. ----- To do it, must download "glove.6B.50d.txt" which is about 163.41 MB
    recreate_word_vectors = False
    # recreate_word_vectors = True
    if recreate_word_vectors:
        recreateWordVectors()

    # load word features 
    # wordfeatures = loadWordFeatures(dom, loadpickle=True)
    # wordfeatures = loadWordFeatures(dom, loadpickle=False)


  
    # in the experiments in the paper, we use the word features directly. However, 
    # you can also use tsne or pca dim-reduced features. 
    # tsnefeatures = computeTSNEFeatures(wordfeatures)
    # pcafeatures = computePCAFeatures(wordfeatures)
    
    # allfeatures = {"wordfeat": wordfeatures, "tsne": tsnefeatures, "pca": pcafeatures}


    # create primary dataset
    # dataset = createDataset(data, reptype, allfeatures)

    # --> Pay attention here <--

    mat_file_name = 'MatDataset.mat'
    dataset = createDataset_fromMatFile(mat_file_name)

    # mat_file_name = 'RawDataset.mat'
    # dataset = createDataset_fromMatFile_(mat_file_name)


    # create dataset splits
    # expdata = getTrainTestValSplit(data, dataset, splittype, excludeid=excludeid, pval=pval, nfolds=nfolds)
    expdata = getTrainTestValSplit_fromMatFile(dataset, splittype, excludeid=excludeid, pval=pval, nfolds=nfolds)
    

    # nfeats = allfeatures[reptype].shape[1]
    nfeats = 50


    # we don't use an initial projection matrix. You can substitute one here if you like
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
    elif modeltype == "lineargaussian":
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
    if splittype=="3participant":
        stopcount = 3
        burnin = 50
    elif splittype == "LOOtask":
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
            # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            #if modeltype == "neural"
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            #optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=10, max_eval=20)
            #optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
            counter = 0

            torch.save(model, os.path.join(modeldir, model.modelname + ".pth"))
            restartopt = False
            t = 1
            #l2comp = nn.L2Loss()

            t_vec = []
            loss_vec = []
            mae_vec = []

            while t < 5000:

                def closure():
                    N = inptaskspred.shape[0]

                    if modeltype == "btm":
                        # predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0], tasksobsids, taskspredids, difficulties_obs, difficulties_pred)
                        predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0], tasksobsids, taskspredids, \
                             obs_task_sens_cap_seq, pred_task_sens_cap, obs_task_proc_cap_seq, pred_task_proc_cap)
                    else:
                        predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0])
                    predtrust = torch.squeeze(predtrust)

                    # print(predtrust)

                    # logloss = torch.mean(torch.pow(predtrust - outtrustpred, 2.0)) # / 2*torch.exp(obsnoise))

                    loss = torch.mean(torch.pow(predtrust - outtrustpred, 2.0))

                    # loss = -(torch.dot(outtrustpred, torch.log(predtrust)) +
                    #         torch.dot((1 - outtrustpred), torch.log(1.0 - predtrust))) / N


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
                        # predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0], tasksobsids, taskspredids, difficulties_obs, difficulties_pred)
                        predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0], tasksobsids, taskspredids, \
                                            obs_task_sens_cap_seq, pred_task_sens_cap, obs_task_proc_cap_seq, pred_task_proc_cap)
                    else:
                        predtrust = model(inptasksobs, inptasksperf, inptaskspred, inptasksobs.shape[0])

                    predtrust = torch.squeeze(predtrust)
                    loss = -(torch.dot(outtrustpred, torch.log(predtrust)) +
                            torch.dot((1 - outtrustpred), torch.log(1.0 - predtrust))) / inptaskspred.shape[0]

                    # compute validation loss
                    if modeltype == "btm":
                        # predtrust_val = model(inptasksobs_val, inptasksperf_val, inptaskspred_val, inptasksobs_val.shape[0], tasksobsids_val, taskspredids_val, difficulties_obs_val, difficulties_pred_val)
                        predtrust_val = model(inptasksobs_val, inptasksperf_val, inptaskspred_val, inptasksobs_val.shape[0], tasksobsids_val, taskspredids_val, \
                        obs_task_sens_cap_seq_val, pred_task_sens_cap_val, obs_task_proc_cap_seq_val, pred_task_proc_cap_val)
                    else:
                        predtrust_val = model(inptasksobs_val, inptasksperf_val, inptaskspred_val, inptasksobs_val.shape[0])
                    predtrust_val = torch.squeeze(predtrust_val)
                    valloss = -(torch.dot(outtrustpred_val, torch.log(predtrust_val)) +
                                torch.dot((1 - outtrustpred_val), torch.log(1.0 - predtrust_val))) / predtrust_val.shape[0]

                    # compute prediction loss
                    if modeltype == "btm":
                        # predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test, inptasksobs_test.shape[0], tasksobsids_test, taskspredids_test, difficulties_obs_test, difficulties_pred_test))
                        predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test, inptasksobs_test.shape[0], tasksobsids_test, taskspredids_test, \
                             obs_task_sens_cap_seq_test, pred_task_sens_cap_test, obs_task_proc_cap_seq_test, pred_task_proc_cap_test))
                        
                    else:
                        predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test, inptasksobs_test.shape[0]))
                    
                    predloss = -(torch.dot(outtrustpred_test, torch.log(predtrust_test)) +
                                torch.dot((1 - outtrustpred_test), torch.log(1.0 - predtrust_test))) / predtrust_test.shape[0]




                    #print(model.wb, model.wtp, model.trust0, model.sigma0)

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



    azvTesting = False

    if azvTesting:
        # read the observed tasks
        inptasksobs_azv = genfromtxt('inptasksobs.csv', delimiter=',')
        inptasksobs_azv = torch.tensor(inptasksobs_azv)
        inptasksobs_azv = torch.unsqueeze(inptasksobs_azv, 1)
        inptasksobs_azv = inptasksobs_azv.type(torch.FloatTensor)


        # read the observed tasks performances
        inptasksperf_azv = genfromtxt('inptasksperf.csv', delimiter=',')
        inptasksperf_azv = torch.tensor(inptasksperf_azv)
        inptasksperf_azv = torch.unsqueeze(inptasksperf_azv, 1)
        inptasksperf_azv = inptasksperf_azv.type(torch.FloatTensor)

        
        # the number of observed tasks goes here... but if only 1 is desired, its better to hack the model and hardcode num_obs_tasks = 1
        num_obs_tasks = inptasksperf_azv.shape[0]


        # read the observed tasks performances
        inptaskspred_azv = genfromtxt('inptaskspred.csv', delimiter=',')
        inptaskspred_azv = torch.tensor(inptaskspred_azv)
        inptaskspred_azv = torch.unsqueeze(inptaskspred_azv, 0)
        inptaskspred_azv = inptaskspred_azv.type(torch.FloatTensor)

        inptasksobs_azv = inptasksobs_azv.cuda()
        inptasksperf_azv = inptasksperf_azv.cuda()
        inptaskspred_azv = inptaskspred_azv.cuda()


        # print(type(inptasksobs_azv))
        # print(type(inptasksperf_azv))
        # print(type(inptaskspred_azv))

        # make predictions using trained model and compute metrics
        if modeltype == "btm":
            # predtrust_test = torch.squeeze(model(inptasksobs_azv, inptasksperf_azv, inptaskspred_azv, inptasksobs_azv.shape[0], tasksobsids_azv, taskspredids_azv, difficulties_obs_azv, difficulties_pred_azv))
            predtrust_test = torch.squeeze(model(inptasksobs_azv, inptasksperf_azv, inptaskspred_azv, inptasksobs_azv.shape[0], tasksobsids_azv, taskspredids_azv, \
                 obs_task_sens_cap_seq_azv, pred_task_sens_cap_azv, obs_task_proc_cap_seq_azv, pred_task_proc_cap_azv))
        else:
            predtrust_test = torch.squeeze(model(inptasksobs_azv, inptasksperf_azv, inptaskspred_azv, inptasksobs_azv.shape[0]))

        print("predtrust_test", predtrust_test)
        stop()


    if not(azvTesting):
        if modeltype == "btm":
            # predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test, inptasksobs_test.shape[0], tasksobsids_test, taskspredids_test, difficulties_obs_test, difficulties_pred_test))
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

    dom = "driving" # sys.argv[1] #"household" or "driving"
    reptype = "wordfeat"
    splittype = "3participant" #"3participant" or "LOOtask"
    modeltype = sys.argv[1] #"neural" or "gp"
    gpmode = 0
    taskrepsize = 50
    
    print(dom, modeltype, splittype, gpmode)
    
    # if dom == "household" or dom == "driving":
    #     _, nparts = loadData(dom)
    # else:
    #     raise ValueError("No such domain")

    # ntasks = 13
    # if splittype == "3participant":
    #     start = 0
    #     nfolds = 10
    #     end = nfolds
    #     pval = 0.15  # validation proportion
    # elif splittype == "LOOtask":
    #     start = 1
    #     end = ntasks
    #     nfolds = ntasks
    #     pval = 0.15  # validation proportion
    # elif splittype == "random":
    #     start = 0
    #     end = 10
    #     nfolds = 10
    # else:
    #     raise ValueError("No such splittype")


    start = 0
    nfolds = 10
    end = nfolds
    pval = 0.15  # validation proportion


    allresults = []
    print(start, end)
    for excludeid in range(start, end):
    # for excludeid in range(1):
        print("Test id:", excludeid)
        result = main(
            dom=dom,
            reptype=reptype,
            splittype=splittype,
            excludeid=excludeid,
            taskrepsize=taskrepsize,
            modeltype=modeltype,
            gpmode=gpmode,
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
        res_mat_file_name = "results_mat_" + modeltype + "_minMAE.mat"

        sio.savemat(res_mat_file_name, res_dict)
        


        resultsdir = "results"
        filename =  dom + "_" + modeltype + "_" + str(taskrepsize) + "_" + splittype + "_" + str(gpmode) + ".pkl"
        with open(os.path.join(resultsdir, filename), 'wb') as output:
            pickle.dump(allresults, output, pickle.HIGHEST_PROTOCOL)

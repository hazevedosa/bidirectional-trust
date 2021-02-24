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

dtype = torch.FloatTensor

if usecuda:
    dtype = torch.cuda.FloatTensor

class BidirectionalTrustModel(torch.nn.Module):

    # Init Method (define parameters)
    def __init__(
                self, 
                modelname, 
                inpsize, 
                obsseqlen,
                taskrepsize,
                capabilityRepresentationSize
                ):
        super(BidirectionalTrustModel, self).__init__()


        self.modelname = modelname # modelname

        self.capabilityRepresentationSize = capabilityRepresentationSize # how many capabilities are represented
        self.capabilityEdges = Variable(dtype(np.ones((self.capabilityRepresentationSize,2)) * [0.0, 1.0]), requires_grad=False) # initialized as zeros and ones

        self.discretizationBins = 10 # how many bins in each dimension
        self.updateProbabilityDistribution() # probability distribution tensor


        self.betas = Parameter(dtype(20.0 * np.random.rand( self.capabilityRepresentationSize ))) # parameters to be optimized
        self.zetas = Parameter(dtype(np.random.rand( self.capabilityRepresentationSize ))) # parameters to be optimized
        # self.zetas = dtype(np.ones( self.capabilityRepresentationSize )) # or only ones

        self.optimizedCapabilitiesMatrix = Parameter(dtype(np.random.rand(1, 12))) # parameters to be optimized

        self.counter = 0



    # Forward Method (model process)
    def forward(self, inptasksobs, inptasksperf, inptaskspred, num_obs_tasks, tasksobsids, taskspredids, \
                obs_task_sens_cap_seq, pred_task_sens_cap, obs_task_proc_cap_seq, pred_task_proc_cap):

        # parameters

        tasksPerObservationSequence = inptasksobs.shape[0]  # 3 for our dataset // 2 for Soh's
        observationSequencesNumber  = inptasksobs.shape[1]  # 49 or 63 or N for our dataset // 192 or 186 for Soh's
        trustPredictionsNumber      = 1                     # adequate to the dataset format... // (both)
        predictedTrust              = Variable(dtype(np.zeros((observationSequencesNumber, trustPredictionsNumber))), requires_grad=False) 
                                                                                                      # (49, 1) for our dataset // (both)


        # for each (of the 49 * 3) observations sequence prior to trust predictions
        for i in range(observationSequencesNumber):
            
            # re-initialize the capability edges
            self.capabilityEdges = Variable(dtype(np.ones((self.capabilityRepresentationSize,2)) * [0.0, 1.0]), requires_grad=False)
            self.updateProbabilityDistribution()


            ## Capabilities estimation loop
            # checks each task on the observation sequence
            for j in range(tasksPerObservationSequence):
                self.capabilityUpdate(inptasksobs[j,i,:], inptasksperf[j,i,:], tasksobsids[j,i,0], \
                    obs_task_sens_cap_seq[j, i], obs_task_proc_cap_seq[j, i])
                    #  difficulties_obs[j, i, 0])


            ## Trust computation loop
            # computes trust for each input task... But in our dataset we consider only 1
            for j in range(trustPredictionsNumber):
                predictedTrust[i, j] = self.computeTrust(taskspredids[i, 0], \
                    pred_task_sens_cap[i, 0], pred_task_proc_cap[i, 0])
                    # difficulties_pred[i, 0])


        trust = predictedTrust

        return dtype(trust)



    # Auxiliary Methods
    def capabilityUpdate(self, observedTask, observedTaskPerformance, observedTaskID,
                            observedTaskSensingCap, observedTaskProcessingCap):

        observedCapability = dtype((observedTaskSensingCap, observedTaskProcessingCap))

        taskIsNonZero, taskSuccess = self.getSuccessOrFailBools(observedTaskPerformance)

        capabilityEdgesChanged = False

        if taskIsNonZero:
            if taskSuccess:
                for i in range(self.capabilityRepresentationSize):
                    if observedCapability[i] > self.capabilityEdges[i, 1]:
                        self.capabilityEdges[i, 1] = observedCapability[i]
                        capabilityEdgesChanged = True
                    elif observedCapability[i] > self.capabilityEdges[i, 0]:
                        self.capabilityEdges[i, 0] = observedCapability[i]
                        capabilityEdgesChanged = True

            else:
                for i in range(self.capabilityRepresentationSize):
                    if  observedCapability[i] < self.capabilityEdges[i, 0]:
                        self.capabilityEdges[i, 0] = observedCapability[i]
                        capabilityEdgesChanged = True
                    elif observedCapability[i] < self.capabilityEdges[i, 1]:
                        self.capabilityEdges[i, 1] = observedCapability[i]
                        capabilityEdgesChanged = True

        for i in range(self.capabilityRepresentationSize):
            if self.capabilityEdges[i, 0] == self.capabilityEdges[i, 1]:
                if self.capabilityEdges[i, 1] == 0.0:
                    self.capabilityEdges[i, 1] = 1 / self.discretizationBins
                else:
                    self.capabilityEdges[i, 0] = self.capabilityEdges[i, 1] - 1 / self.discretizationBins

        if capabilityEdgesChanged == True:
            self.updateProbabilityDistribution()

        return


    def getSuccessOrFailBools(self, observedTaskPerformance):
        
        if not(observedTaskPerformance[0]) and not(observedTaskPerformance[1]):
            taskIsNonZero = False
            taskSuccess = False
        elif not(observedTaskPerformance[0]) and observedTaskPerformance[1]:
            taskIsNonZero = True
            taskSuccess = True
        elif observedTaskPerformance[0] and not(observedTaskPerformance[1]):
            taskIsNonZero = True
            taskSuccess = False
        else:
            print("Error: performance indicators = [1, 1]")
            raise SystemExit(0)

        return taskIsNonZero, taskSuccess


    def sigm(self, x):
        return 1 / (1 + torch.exp(-x))

    def computeTrust(self, inptaskspredID, predictionTaskSensingCap, predictionTaskProcessingCap):

        requiredCapability = dtype((predictionTaskSensingCap, predictionTaskProcessingCap))

        trust = 0.0

        if self.capabilityRepresentationSize == 1:
            for j in range(self.discretizationBins):
                stepInDim_j = (j + 0.5) / self.discretizationBins
                trust = trust + self.trustGivenCapability([stepInDim_j], requiredCapability) * self.probabilityDistribution[j]

        elif self.capabilityRepresentationSize == 2:
            for k in range(self.discretizationBins):
                stepInDim_k = (k + 0.5) / self.discretizationBins
                for j in range(self.discretizationBins):
                    stepInDim_j = (j + 0.5) / self.discretizationBins
                    trust = trust + self.trustGivenCapability([stepInDim_j, stepInDim_k], 
                                                                requiredCapability) * self.probabilityDistribution[j, k]

        elif self.capabilityRepresentationSize == 3:
            for l in range(self.discretizationBins):
                stepInDim_l = (l + 0.5) / self.discretizationBins
                for k in range(self.discretizationBins):
                    stepInDim_k = (k + 0.5) / self.discretizationBins
                    for j in range(self.discretizationBins):
                        stepInDim_j = (j + 0.5) / self.discretizationBins
                        trust = trust + self.trustGivenCapability([stepInDim_j, stepInDim_k, stepInDim_l], 
                                                                    requiredCapability) * self.probabilityDistribution[j, k, l]

        # print("capEdges: ", self.capabilityEdges)
        # print("reqCap: ", requiredCapability)
        # print("Trust: ", trust)
        # print("------")

        return trust


    def trustGivenCapability(self, capability, requiredCapability):

        trust = 1.0
        for i in range(self.capabilityRepresentationSize):

            p_i = self.betas[i] * (requiredCapability[i] - capability[i])
            d_i = ( 1 + torch.exp(p_i) ) ** ( - self.zetas[i] * self.zetas[i] )

            trust = trust * d_i

        return trust


    def updateProbabilityDistribution(self):

        # Tuple to start the distribution tensor
        probabilityStarter = tuple(self.discretizationBins * np.ones((self.capabilityRepresentationSize), dtype = int))

        # Distribution tensors
        probabilityDistribution = torch.ones(probabilityStarter, dtype = torch.int8)
        # zeroProbability = torch.ones(probabilityStarter, dtype = torch.int8)


        # hardcoded solution: for 1 dim
        if self.capabilityRepresentationSize == 1:
            for j in range(self.discretizationBins):
                step = (j + 0.5) / self.discretizationBins
                if step < self.capabilityEdges[0, 0]:
                    probabilityDistribution[j] = 0
                if step > self.capabilityEdges[0, 1]:
                    probabilityDistribution[j] = 0
            
            probabilityDistribution = probabilityDistribution.float()
            if usecuda:
                probabilityDistribution = probabilityDistribution.cuda()
            probabilityDistribution = dtype(probabilityDistribution)
            probabilityDistribution = probabilityDistribution / torch.sum(probabilityDistribution)

        # hardcoded solution: for 2 dim
        if self.capabilityRepresentationSize == 2:

            for j in range(self.discretizationBins):
                step = (j + 0.5) / self.discretizationBins

                if step < self.capabilityEdges[0, 0]:
                    probabilityDistribution[j,:] = 0
                if step > self.capabilityEdges[0, 1]:
                    probabilityDistribution[j,:] = 0
                if step < self.capabilityEdges[1, 0]:
                    probabilityDistribution[:,j] = 0
                if step > self.capabilityEdges[1, 1]:
                    probabilityDistribution[:,j] = 0
            
            probabilityDistribution = probabilityDistribution.float()
            if usecuda:
                probabilityDistribution = probabilityDistribution.cuda()
            probabilityDistribution = dtype(probabilityDistribution)
            probabilityDistribution = probabilityDistribution / torch.sum(probabilityDistribution)

        self.probabilityDistribution = probabilityDistribution
        return
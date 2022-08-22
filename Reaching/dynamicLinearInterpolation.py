import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import pandas as pd
import time

dof = 7
num_ctrl = 7        
trajecLength = 3000
numTrajectoriesTest = 10

sizeofMatrix = dof * 2 * dof 
inputSize = (2 * sizeofMatrix) + 1 + (2 * dof)

def createListReupdatePoints(trajectoryStates, trajectoryControls, minN, maxN, velGradSensitivty):
    reEvaluatePoints = []
    counterSinceLastEval = 0
    reEvaluatePoints.append(0)

    currentGrads = np.zeros(2*dof)
    lastGrads = np.zeros(2*dof)
    first = True

    for i in range(trajecLength):
        
        # if(i == 0):
        #     lastGrads = currentGrads.copy()

        if(counterSinceLastEval >= minN):

            currState = trajectoryStates[i,:].copy()
            lastState = trajectoryStates[i-minN,:].copy()

            currentGrads = currState - lastState

            if(first):
                first = False
                reEvaluatePoints.append(i)
                counterSinceLastEval = 0
            else:
                if(newEvaluationNeeded(currentGrads, lastGrads, velGradSensitivty)):
                    reEvaluatePoints.append(i)
                    counterSinceLastEval = 0

            lastGrads = currentGrads.copy()
            

        if(counterSinceLastEval >= maxN):
            reEvaluatePoints.append(i)
            counterSinceLastEval = 0

        counterSinceLastEval = counterSinceLastEval + 1

    reEvaluatePoints.append(trajecLength - 1)

    return reEvaluatePoints

def newEvaluationNeeded(currentGrads, lastGrads, sensitivity):
    newEvalNeeded = False

    for i in range(dof):
        # print("current grad: " + str(currentGrads[dof + i]))
        # print("last grad: " + str(lastGrads[dof + i]))
        velGradDiff = currentGrads[dof + i] - lastGrads[dof + i]
        # print(velGradDiff)

        if(velGradDiff > sensitivity):
            newEvalNeeded = True
            #print("new eval needed, diff: " + str(velGradDiff))

        if(velGradDiff < -sensitivity):
            newEvalNeeded = True
            #print("new eval needed, diff: " + str(velGradDiff))


    return newEvalNeeded

def generateLinearInterpolationData(A_matrices, reEvaluationIndicies):
    sizeofAMatrix = len(A_matrices[0])
    linInterpolationData = np.zeros((trajecLength - 1, sizeofAMatrix))

    numBins = len(reEvaluationIndicies) - 1
    print("num bins: " + str(numBins))
    stepsBetween = 0

    for i in range(numBins):

        for j in range(sizeofAMatrix):

            startIndex = reEvaluationIndicies[i]
            startVals = A_matrices[startIndex,:]
            #print("start val: " + str(startVals))

            endIndex = reEvaluationIndicies[i + 1]
            endVals = A_matrices[endIndex,:]
            #print("end val: " + str(endVals))

            diff = endVals - startVals
            #print("diff: " + str(diff))

            stepsBetween = endIndex - startIndex

            linInterpolationData[startIndex,:] = startVals

            for k in range(1, stepsBetween):
                linInterpolationData[startIndex + k,:] = startVals + (diff * (k/stepsBetween))

    return linInterpolationData

def evaluateInterpolation(testTrajectories, states, controls):
    meanSSE_lin = 0 

    SSD_lin = np.zeros((len(testTrajectories)))
    numBins = np.zeros((len(testTrajectories)))
    avgBins = 0

    linInterpolationData = []

    # loop over the test trajectories
    for j in range(len(testTrajectories)):

        reEvaluationIndicies = createListReupdatePoints(states[j], controls[j], 5, 200, 0.01)
        numBins[j] = len(reEvaluationIndicies)
        print(reEvaluationIndicies)
        linInterpolationData = generateLinearInterpolationData(testTrajectories[j], reEvaluationIndicies)

        SSD_lin[j] = calcMeanSumSquaredDiffForTrajec(linInterpolationData, testTrajectories[j])
        print("SSD_lin was: " + str(SSD_lin[j]))

        plt.plot(testTrajectories[j][:, 5])
        plt.plot(linInterpolationData[:, 5])
        plt.show()

    return SSD_lin, numBins

def calcMeanSumSquaredDiffForTrajec(groundTruth, prediction):
    size = len(groundTruth[0])

    array1Size = len(groundTruth)
    array2Size = len(prediction)
    lenofTrajec = array2Size
    
    if(array1Size < array2Size):
        lenofTrajec = array1Size
    
    sumSqDiff = np.zeros((lenofTrajec))

    for i in range(lenofTrajec):
        diffVals = np.zeros((size))
        
        for j in range(size):

            diffVals[j] = groundTruth[i, j] - prediction[i, j]
            sumSqDiff[i] = sumSqDiff[i] + (diffVals[j] * diffVals[j])

        # print(sumSqDiff[i])

    #print(sumSqDiff[0:10])

    meanSumSquared = np.mean(sumSqDiff)

    return meanSumSquared 


def plottingScatter():
    pandas = pd.read_csv('error_lin.csv', header=None)
    error_lin = pandas.to_numpy()

    pandas = pd.read_csv('error_quad.csv', header=None)
    error_quad = pandas.to_numpy()

    pandas = pd.read_csv('numEvalsDynamicsLin.csv', header=None)
    numEvalsLin = pandas.to_numpy()

    pandas = pd.read_csv('sumSquaredDiffsDynamicLin.csv', header=None)
    error_dynlin = pandas.to_numpy()

    normalEvals = np.array([1500, 600, 300, 150, 60, 30, 15, 6])
    #normalEvals = np.array([2, 5, 10, 20, 50, 100, 200, 500])
    data_error_lin = np.zeros((len(error_lin), 2))
    data_error_quad = np.zeros((len(error_quad), 2))
    data_error_dynLin = np.zeros((len(error_quad), 2))

    for i in range(len(error_lin)):
        data_error_lin[i, 0] = error_lin[i]
        data_error_lin[i, 1] = normalEvals[i]

        data_error_quad[i, 0] = error_quad[i]
        data_error_quad[i, 1] = normalEvals[i]

        data_error_dynLin[i, 0] = error_dynlin[i]
        data_error_dynLin[i, 1] = numEvalsLin[i]


def main():


    pandas = pd.read_csv('testing_data/testing_trajectories.csv', header=None)

    testTrajectories = []
    testStates = []
    testControls = []

    for i in range(numTrajectoriesTest):
        testTrajectories.append([])
        tempPandas = pandas.iloc[i*3000:(i + 1)*3000]
        testTrajectories[i] = tempPandas.to_numpy()

    pandas = pd.read_csv('testing_data/testing_states.csv', header=None)

    for i in range(numTrajectoriesTest):
        testStates.append([])
        tempPandas = pandas.iloc[i*3000:(i + 1)*3000]
        testStates[i] = tempPandas.to_numpy()

    pandas = pd.read_csv('testing_data/testing_controls.csv', header=None)

    for i in range(numTrajectoriesTest):
        testControls.append([])
        tempPandas = pandas.iloc[i*3000:((i + 1)*3000)]
        testControls[i] = tempPandas.to_numpy()

    print("DATA LOADED")

    sumSquaredDiffs, numEvals = evaluateInterpolation(testTrajectories, testStates, testControls)

    meanSSD = np.mean(sumSquaredDiffs) 
    avgEvals = np.mean(numEvals)
    print("mean sum sqaured diff: " + str(meanSSD))
    print("average nuumber of evaluaions: " + str(avgEvals))







main()

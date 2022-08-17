import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import time

dof = 7
num_ctrl = 7        
trajecLength = 3000
numTrajectoriesTest = 10

sizeofMatrix = dof * 2 * dof 
inputSize = (2 * sizeofMatrix) + 1 + (2 * dof)

def createListReupdatePoints(trajectoryStates, trajectoryControls, minN, maxN):
    reEvaluatePoints = []
    counterSinceLastEval = 0
    reEvaluatePoints.append(0)

    for i in range(trajecLength):

        if(counterSinceLastEval >= minN):
            if(newEvaluationNeeded(trajectoryStates[i], trajectoryStates[i-1], 0.4)):
                reEvaluatePoints.append(i)
                counterSinceLastEval = 0

        if(counterSinceLastEval >= maxN):
            reEvaluatePoints.append(i)
            counterSinceLastEval = 0

        counterSinceLastEval = counterSinceLastEval + 1

    reEvaluatePoints.append(trajecLength - 1)

    return reEvaluatePoints

def newEvaluationNeeded(currentState, lastState, sensitivity):
    newEvalNeeded = False

    for i in range(dof):
        velDiff = currentState[dof + i] = lastState[dof + i]

        if(velDiff > sensitivity):
            newEvalNeeded = True

        if(velDiff < -sensitivity):
            newEvalNeeded = True


    return newEvalNeeded

def generateLinearInterpolationData(A_matrices, reEvaluationIndicies):
    sizeofAMatrix = len(A_matrices[0])
    linInterpolationData = np.zeros((trajecLength - 1, sizeOfAMatrix))

    numBins = len(reEvaluationIndicies) - 1
    print("num bins: " + str(numBins))
    stepsBetween = 0

    for i in range(numBins):

        for j in range(sizeOfAMatrix):

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




def main():


    pandas = pd.read_csv('Data/testing_trajectories.csv', header=None)

    testTrajectories = []
    states = []

    for i in range(numTrajectoriesTest):
        testTrajectories.append([])
        tempPandas = pandas.iloc[i*3000:(i + 1)*3000]
        testTrajectories[i] = tempPandas.to_numpy()
    
    pandas = pd.read_csv('Data/testing_states.csv', header=None)

    for i in range(numTrajectoriesTest):
        states.append([])
        tempPandas = pandas.iloc[i*3000:(i + 1)*3000]
        states[i] = tempPandas.to_numpy()

    print("DATA LOADED")





    pass




main()

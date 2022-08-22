from operator import delitem
import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import pandas as pd
import time

dof = 9
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
    
    SqDiff = np.zeros((lenofTrajec, size))
    meanSqDiff = np.zeros((size))


    for i in range(lenofTrajec):
        
        for j in range(size):

            diffVals = (groundTruth[i, j] - prediction[i, j]) * (groundTruth[i, j] - prediction[i, j])
            if(diffVals < 10):
                
                SqDiff[i, j] = diffVals
            else:
                SqDiff[i, j] = 0
                print("oopsie big diff: " + str(diffVals))

    for j in range(size):
        meanSqDiff[j] = np.mean(SqDiff[:,j])

    #print("sum squared diff matrices: " + str(sumSqDiff))
    meanSumSquared = np.mean(meanSqDiff)

    return meanSumSquared 


def plottingScatter():
    pandas = pd.read_csv('error_lin.csv', header=None)
    error_lin = pandas.to_numpy()

    pandas = pd.read_csv('error_quad.csv', header=None)
    error_quad = pandas.to_numpy()

    pandas = pd.read_csv('error_NN.csv', header=None)
    error_NN = pandas.to_numpy()

    pandas = pd.read_csv('numEvalsDynamicsLin.csv', header=None)
    numEvalsLin = pandas.to_numpy()

    pandas = pd.read_csv('sumSquaredDiffsDynamicLin.csv', header=None)
    error_dynlin = pandas.to_numpy()

    normalEvals = np.array([1500, 600, 300, 150, 60, 30, 15, 6])
    #normalEvals = np.array([2, 5, 10, 20, 50, 100, 200, 500])
    data_error_lin = np.zeros((len(error_lin), 2))
    data_error_quad = np.zeros((len(error_quad), 2))
    data_error_NN = np.zeros((len(error_NN), 2))
    data_error_dynLin = np.zeros((len(error_dynlin), 2))
    

    for i in range(len(normalEvals)):
        data_error_lin[i, 0] = error_lin[i]
        data_error_lin[i, 1] = normalEvals[i]

        data_error_quad[i, 0] = error_quad[i]
        data_error_quad[i, 1] = normalEvals[i]

        data_error_NN[i, 0] = error_NN[i]
        data_error_NN[i, 1] = normalEvals[i]

    for i in range(len(error_dynlin)):

        data_error_dynLin[i, 0] = error_dynlin[i]
        data_error_dynLin[i, 1] = numEvalsLin[i]

    colors = ["r", "b", "g"]

    print(data_error_lin[:,1])
    print(data_error_lin[:,0])

    plt.scatter(data_error_lin[:,1], data_error_lin[:,0], color = "r", label='Linear Interpolation')
    plt.scatter(data_error_quad[:,1], data_error_quad[:,0], color = "b", label='Quadratic Interpolation')
    plt.scatter(data_error_NN[:,1], data_error_NN[:,0], color = "m", label='NN Interpolation')
    plt.scatter(data_error_dynLin[:,1], data_error_dynLin[:,0], color = "g", label='Dynamic Linear Interpolation')

    plt.xlabel("Number of dynamic Evaluations")
    plt.ylabel("Mean Sum Squared Error")
    plt.title('Comparing different interpolation methods for different step sizes')
    plt.legend()

    plt.show()
    


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
    print("mean sum squared diff: " + str(meanSSD))
    print("average nuumber of evaluaions: " + str(avgEvals))


    np.savetxt("sumSquaredDiffsDynamicLin.csv", sumSquaredDiffs, delimiter=',')
    np.savetxt("numEvalsDynamicsLin.csv", numEvals, delimiter=',')







#main()
plottingScatter()

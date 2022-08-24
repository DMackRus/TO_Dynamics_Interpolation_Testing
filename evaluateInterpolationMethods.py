import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys
import tensorflow as tf

print("PROGRAM START")

if(len(sys.argv) < 2):
    print("not enough arguments")
    exit()
else:
    mode = int(sys.argv[1])
    print(mode)

dof = 0
num_ctrl = 0
startPath = ""

if(mode == 0):
    dof = 2
    num_ctrl = 2
    startPath = "Pendulum/"
elif(mode == 1):
    dof = 7
    num_ctrl = 7
    startPath = "Reaching/"

elif(mode == 2):
    dof = 9
    num_ctrl = 7
    startPath = "Pushing/"

else:
    print("invalid mode specified")
    exit()

trajecLength = 3000
numTrajectoriesTest = 2

sizeofMatrix = dof * 2 * dof 
inputSize = (2 * sizeofMatrix) + 1 + (2 * dof)
model_A = tf.keras.models.load_model(startPath + 'models/A_model.model')


def generateQuadInterpolationData(testTrajec, n):
    quadInterpolationData = np.zeros((trajecLength - (2 * n), sizeofMatrix))

    numBinsQuad = int(trajecLength / n) - 2
    for i in range(numBinsQuad):

        for j in range(sizeofMatrix):

            points = np.zeros((3, 2))
            for k in range(3):
                point = np.array([testTrajec[((i+k) * n),j], k * n])

                points[k] = point

            #solve for coefficients a, b, c

            x_matrix = np.zeros((3, 3))
            y_matrix = np.zeros((1, 3))

            y_matrix[0, 0] = points[0, 0]
            y_matrix[0, 1] = points[1, 0]
            y_matrix[0, 2] = points[2, 0]

            for k in range(3):

                x_matrix[0, k] = points[k, 1] * points[k, 1]
                x_matrix[1, k] = points[k, 1]
                x_matrix[2, k] = 1

            x_inv = np.linalg.inv(x_matrix)


            abc = y_matrix @ x_inv


            quadInterpolationData[(i * n),j] = testTrajec[(i * n),j]


            for k in range(n):
                a = abc[0, 0]
                b = abc[0, 1]
                c = abc[0, 2]

                nextVal = (a * k * k) + (b * k) + c
                quadInterpolationData[(i * n) + k,j] = nextVal

    return quadInterpolationData

def generateLinearInterpolationData(testTrajec, n):
    linInterpolationData = np.zeros((trajecLength - (n), sizeofMatrix))

    numBins = int(trajecLength / n) - 1
    for i in range(numBins):

        for j in range(sizeofMatrix):

            startVals = testTrajec[(i * n),:]
            #print("start val: " + str(startVals))

            endVals = testTrajec[((i + 1) * n),:]
            #print("end val: " + str(endVals))

            diff = endVals - startVals
            #print("diff: " + str(diff))

            linInterpolationData[(i * n),:] = startVals

            for k in range(1, n):
                linInterpolationData[(i * n) + k,:] = startVals + (diff * (k/n))

    return linInterpolationData

def formatNNTestData(n , trajecLength, trajecData_pd, states, sizeOfMatrix, minValsA, maxValsA, minValsStates, maxValsStates):

    # plt.plot(trajecData_pd[:, 1])
    # plt.show()
    for j in range(sizeofMatrix):
        trajecData_pd[:, j] = (trajecData_pd[:, j] - minValsA[j]) / (maxValsA[j] - minValsA[j])

    for j in range(2 * dof):
        states[:, j] = (states[:, j] - minValsStates[j]) / (maxValsStates[j] - minValsStates[j])

    # plt.plot(trajecData_pd[:, 1])
    # plt.show()


    bins = int((trajecLength) / n) - 1

    listsInputs = np.zeros((bins * n, (2 * sizeOfMatrix) + 1 + (2 * dof))) 
    
    print("number of bins " + str(bins))

    for i in range(bins):

        startIndex = (i * (n))
        endIndex = ((i + 1) * (n))
        startRow = trajecData_pd[startIndex,:]
        endRow = trajecData_pd[endIndex,:]
        
        row = np.zeros((1, (2 * sizeOfMatrix) + 1 + (2 * dof)))

        row[0, 0:sizeOfMatrix] = startRow
        row[0, sizeOfMatrix:(2*sizeOfMatrix)] = endRow

        for j in range(0, n):

            row[0, (2*sizeOfMatrix):(2*sizeOfMatrix) + (2 * dof)] = states[startIndex + j,:]
            row[0, -1] = (j / n)

            listsInputs[(i * n) + j,:] = row

            # print("index: " + str((i * n) + j))
            # print(listsInputs[(i * n) + j,:])

    return listsInputs



def generateNNInterpolationData(trajectory, formattedTestData, n, minValsA, maxValsA, minValsStates, maxValsStates):
    
    NNPredictionInterpolationData = np.zeros((trajecLength - (n), sizeofMatrix))

    bins = int(trajecLength / n)
    for i in range(bins - 1):

        for j in range(sizeofMatrix):
            NNPredictionInterpolationData[(i * n),j] = (trajectory[(i * n),j] - minValsA[j]) / (maxValsA[j] - minValsA[j])
            
        for k in range(1, n):
            nextInput = formattedTestData[(i * (n)) + (k),:]
            #print("-------------------------------------------------")
            #print("next input: " + str(nextInput))
            #print(nextInput[1 + 5])
            #print(nextInput[99 + 5])
            #print(nextInput[210])
            nextInput = nextInput.reshape(-1, (inputSize))
            
            
            nextOutput = model_A(nextInput)
            #print(nextOutput[0][5])
            #print("-------------------------------------------------")
            #time.sleep(2)

            NNPredictionInterpolationData[(i * n) + k,:] = nextOutput


    for j in range(sizeofMatrix):
        NNPredictionInterpolationData[:, j] = (NNPredictionInterpolationData[:, j] * (maxValsA[j] - minValsA[j])) + minValsA[j]

            
    return NNPredictionInterpolationData

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

def generateDynLinInterpolation(A_matrices, reEvaluationIndicies):
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


def evaluateInterpolationMethods(nValues, testTrajectories, states, controls,  minValsA, maxValsA, minValsStates, maxValsStates):
    meanSSE_lin = np.zeros((len(nValues)))
    meanSSE_quad = np.zeros((len(nValues)))
    meanSSE_NN = np.zeros((len(nValues)))

    mean_lin_interpolationTime = np.zeros((len(nValues)))
    mean_quad_interpolationTime = np.zeros((len(nValues)))
    mean_NN_interpolationTime = np.zeros((len(nValues)))
    
    SSD_lin = np.zeros((numTrajectoriesTest))
    SSD_quad = np.zeros((numTrajectoriesTest))
    SSD_NN = np.zeros((numTrajectoriesTest))

    lin_interpolationTimes = np.zeros((numTrajectoriesTest))
    quad_interpolationTimes = np.zeros((numTrajectoriesTest))
    NN_interpolationTimes = np.zeros((numTrajectoriesTest))
    dynLin_interpolationTimes = np.zeros((numTrajectoriesTest))

    linInterpolationData = []
    quadInterpolationData = []
    NNInterpolationData = []

    meanSSE_linDyn = 0 
    MAE_dynLin = np.zeros((len(testTrajectories)))
    numBins = np.zeros((len(testTrajectories)))
    avgBins = 0

    linInterpolationData = []


    # loop over the test trajectories
    for j in range(len(testTrajectories)):
        start = time.process_time()
        reEvaluationIndicies = createListReupdatePoints(states[j], controls[j], 5, 200, 0.005)
        numBins[j] = len(reEvaluationIndicies)
        print(reEvaluationIndicies)
        linInterpolationData = generateDynLinInterpolation(testTrajectories[j], reEvaluationIndicies)
        dynLin_interpolationTimes[j] = time.process_time() - start


        MAE_dynLin[j] = calcMeanSumSquaredDiffForTrajec(linInterpolationData, testTrajectories[j])
        print("SSD_lin was: " + str(SSD_lin[j]))

        plt.plot(testTrajectories[j][:, 5])
        plt.plot(linInterpolationData[:, 5])
        plt.show()

    # loop over the different values for n (steps between recalculating derivatives)
    for i in range(len(nValues)):
        n = nValues[i]
        print("current N Value is: " + str(n))

        # loop over the test trajectories
        for j in range(len(testTrajectories)):
            start = time.process_time()
            linInterpolationData = generateLinearInterpolationData(testTrajectories[j], n)
            lin_interpolationTimes[j] = time.process_time() - start
            print("time taken for linear: " + str(lin_interpolationTimes[j]))

            start = time.process_time()
            quadInterpolationData = generateQuadInterpolationData(testTrajectories[j], n)
            quad_interpolationTimes[j] = time.process_time() - start
            print("time taken for quad: " + str(quad_interpolationTimes[j]))

            start = time.process_time()
            formattedData = formatNNTestData(n, trajecLength, testTrajectories[j].copy(), states[j].copy(), sizeofMatrix,minValsA, maxValsA, minValsStates, maxValsStates)
            NNInterpolationData = generateNNInterpolationData(testTrajectories[j].copy(), formattedData.copy(), n, minValsA, maxValsA, minValsStates, maxValsStates)
            NN_interpolationTimes[j] = time.process_time() - start
            print("time taken for NN: " + str(NN_interpolationTimes[j]))

            print("---------------------- Linear interpolation ---------------------")
            SSD_lin[j] = calcMeanSumSquaredDiffForTrajec(linInterpolationData, testTrajectories[j])
            print("SSD_lin was: " + str(SSD_lin[j]))

            print("---------------------- Quadratic interpolation ---------------------")
            SSD_quad[j] = calcMeanSumSquaredDiffForTrajec(quadInterpolationData, testTrajectories[j])
            print("SSD_quad was: " + str(SSD_quad[j]))

            print("---------------------- Neural network interpolation ---------------------")
            SSD_NN[j] = calcMeanSumSquaredDiffForTrajec(NNInterpolationData, testTrajectories[j])
            print("SSD_NN was: " + str(SSD_NN[j]))

            # plt.plot(testTrajectories[j][:, 1])
            # plt.plot(linInterpolationData[:, 1])
            # plt.plot(NNInterpolationData[:, 1])
            # plt.show()

            # if(SSD_lin[j] > 5e-2):
            #     plt.plot(testTrajectories[j][:, 23])
            #     plt.plot(linInterpolationData[:, 23])
            #     plt.plot(NNInterpolationData[:, 23])
            #     plt.show()

        meanSSE_lin[i] = np.mean(SSD_lin) 
        print("------------------------------------------------")
        print("mean SSE Linear: " + str(meanSSE_lin[i]))

        meanSSE_quad[i] = np.mean(SSD_quad) 
        print("mean SSE Quad: " + str(meanSSE_quad[i]))

        meanSSE_NN[i] = np.mean(SSD_NN)
        print("mean SSE NN: " + str(meanSSE_NN[i]))
        print("-------------------------------------------------") 

        mean_lin_interpolationTime[i] = np.mean(lin_interpolationTimes)
        mean_quad_interpolationTime[i] = np.mean(quad_interpolationTimes)
        mean_NN_interpolationTime[i] = np.mean(NN_interpolationTimes)
    
    

    meanSSE_linDyn = np.mean(MAE_dynLin)
    avgBins = n

    return meanSSE_lin, meanSSE_quad, meanSSE_NN, MAE_dynLin, numBins, mean_lin_interpolationTime, mean_quad_interpolationTime, mean_NN_interpolationTime, dynLin_interpolationTimes


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

            diffVals = abs((groundTruth[i, j] - prediction[i, j]))
            SqDiff[i, j] = diffVals
            # if(diffVals < 10):
                
                
            # else:
            #     SqDiff[i, j] = 0
            #     print("oopsie big diff: " + str(diffVals))

    for j in range(size):
        # stddev = np.std(SqDiff[:,j])
        # mean = np.mean(SqDiff[:,j])

        # ok = SqDiff[:,j] > (mean - (3 * stddev))
        # SqDiff[~ok,j] = mean

        # #step 2, values higher than 1 std from mean
        # # ok = SqDiff[:,j] < (mean + ( 3 * stddev))
        # # SqDiff[~ok,j] = mean

        meanSqDiff[j] = np.mean(SqDiff[:,j])

    #print("sum squared diff matrices: " + str(sumSqDiff))
    meanSumSquared = np.mean(meanSqDiff)

    return meanSumSquared 


def main():
    pandas = pd.read_csv(startPath + 'testing_data/testing_trajectories.csv', header=None)
    minValsA = pd.read_csv(startPath + 'testing_data/minValuesA.csv', header=None).to_numpy()
    maxValsA = pd.read_csv(startPath + 'testing_data/maxValuesA.csv', header=None).to_numpy()
    minValsStates = pd.read_csv(startPath + 'testing_data/minValuesStates.csv', header=None).to_numpy()
    maxValsStates = pd.read_csv(startPath + 'testing_data/maxValuesStates.csv', header=None).to_numpy()

    # get the 980th trajectory

    testTrajectories = []
    states = []
    controls = []

    for i in range(numTrajectoriesTest):
        testTrajectories.append([])
        tempPandas = pandas.iloc[i*trajecLength:(i + 1)*trajecLength]
        testTrajectories[i] = tempPandas.to_numpy()
    
    pandas = pd.read_csv(startPath + 'testing_data/testing_states.csv', header=None)

    for i in range(numTrajectoriesTest):
        states.append([])
        tempPandas = pandas.iloc[i*trajecLength:(i + 1)*trajecLength]
        states[i] = tempPandas.to_numpy()

    pandas = pd.read_csv(startPath + 'testing_data/testing_controls.csv', header=None)

    for i in range(numTrajectoriesTest):
        controls.append([])
        tempPandas = pandas.iloc[i*trajecLength:(i + 1)*trajecLength]
        controls[i] = tempPandas.to_numpy()

    print("DATA LOADED")

    # n = 20
    # formattedData = formatNNTestData(n, trajecLength, testTrajectories[0].copy(), states[0].copy(), sizeofMatrix, minValsA, maxValsA, minValsStates, maxValsStates)
    # NNInterpolationData = generateNNInterpolationData(testTrajectories[0].copy(), formattedData.copy(), n, minValsA, maxValsA, minValsStates, maxValsStates)
    # pandas = pd.DataFrame(NNInterpolationData)
    # pandas.to_csv("NNInterpolation_Data.csv", header=None, index=None)

    # index = 5

    # for i in range(20):

    #     plt.plot(testTrajectories[0][:,i])

    #     plt.plot(NNInterpolationData[:,i])
    #     plt.show()


    nValues = np.array([2, 5, 10, 20, 50, 100, 200])
    error_lin, error_quad, error_NN, error_dynLin, evals_dynLin, time_lin, time_quad, time_NN, time_dynLin = evaluateInterpolationMethods(nValues, testTrajectories, states, controls, minValsA, maxValsA, minValsStates, maxValsStates)


    print(error_lin)
    print(error_quad)
    print(error_NN)
    print(error_dynLin)
    print("number of evals dynLin: " + str(evals_dynLin))
    print(time_lin)
    print(time_quad)
    print(time_NN)
    print(time_dynLin)

    np.savetxt(startPath + "Results/error_lin.csv", error_lin, delimiter=",")
    np.savetxt(startPath + "Results/error_quad.csv", error_quad, delimiter=",")
    np.savetxt(startPath + "Results/error_NN.csv", error_NN, delimiter=",")
    np.savetxt(startPath + "Results/error_dynLin.csv", error_dynLin, delimiter=",")
    np.savetxt(startPath + "Results/evals_dynLin.csv", evals_dynLin, delimiter=",")

    np.savetxt(startPath + "Results/time_lin.csv", time_lin, delimiter=",")
    np.savetxt(startPath + "Results/time_quad.csv", time_quad, delimiter=",")
    np.savetxt(startPath + "Results/time_NN.csv", time_NN, delimiter=",")
    np.savetxt(startPath + "Results/time_dynLin.csv", time_dynLin, delimiter=",")
    
    x_vals = np.array([])
    plt.xlabel("steps between recalculating derivatives")
    plt.ylabel("Mean Absolute Error")
    plt.title('Comparing different interpolation methods for different step sizes')
    default_x_ticks = range(len(error_lin))
    plt.xticks(default_x_ticks, nValues)
    plt.plot(error_lin, label = 'linear')
    plt.plot(error_quad, label = 'quadratic')
    plt.plot(error_NN, label = 'NN')
    plt.legend()
    plt.show()

    print("PROGRAM FINISHED")


main()



import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import tensorflow as tf
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
model_A = tf.keras.models.load_model('models/A_model.model')


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

def formatNNTestData(n , trajecLength, trajecData_pd, states, sizeOfMatrix):

    # bins = int((trajecLength) / n) - 1

    # listsInputs = np.zeros((bins * n, (2 * sizeOfMatrix) + 1 + (2 * dof))) 
    
    # # print("number of bins " + str(bins))
    # # print("length of data " + str(len(trajecData_pd)))

    # for i in range(bins):

    #     startIndex = (i * (n))
    #     endIndex = ((i + 1) * (n))
    #     startRow = trajecData_pd[startIndex,:]
    #     endRow = trajecData_pd[endIndex,:]

    #     # if(i % 500 == 0):
    #     #     print("data " + str(i))

    #     for j in range(0, n):

    #         currentState = states[startIndex + j,:]

    #         row = np.array(())
    #         row = np.hstack([row, startRow])
    #         row = np.hstack([row, endRow])
    #         row = np.hstack([row, currentState])
    #         row = np.hstack([row, (j/n)])


    #         listsInputs[(i * n) + j:] = row

    # return listsInputs

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



def generateNNInterpolationData(trajectory, formattedTestData, n):
    
    NNPredictionInterpolationData = np.zeros((trajecLength - (n), sizeofMatrix))

    bins = int(trajecLength / n)
    for i in range(bins - 1):

        NNPredictionInterpolationData[(i * n),:] = trajectory[(i * n),:]
            
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
            
    return NNPredictionInterpolationData


def evaluateInterpolationMethods(nValues, testTrajectories, states):
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

    linInterpolationData = []
    quadInterpolationData = []
    NNInterpolationData = []

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
            formattedData = formatNNTestData(n, trajecLength, testTrajectories[j], states[j], sizeofMatrix)
            NNInterpolationData = generateNNInterpolationData(testTrajectories[j], formattedData, n)
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

            # plt.plot(testTrajectories[j][:, 23])
            # plt.plot(linInterpolationData[:, 23])
            # plt.plot(NNInterpolationData[:, 23])
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


    return meanSSE_lin, meanSSE_quad, meanSSE_NN, mean_lin_interpolationTime, mean_quad_interpolationTime, mean_NN_interpolationTime


def calcMeanSumSquaredDiffForTrajec(groundTruth, prediction):
    size = len(groundTruth[0])

    array1Size = len(groundTruth)
    array2Size = len(prediction)
    lenofTrajec = array2Size
    
    if(array1Size < array2Size):
        lenofTrajec = array1Size
    
    sumSqDiff = np.zeros((size))

    for i in range(lenofTrajec):
        diffVals = np.zeros((size))
        
        for j in range(size):

            diffVals[j] = groundTruth[i, j] - prediction[i, j]
            sumSqDiff[j] = sumSqDiff[j] + (diffVals[j] * diffVals[j])

    #print("sum squared diff matrices: " + str(sumSqDiff))
    meanSumSquared = np.mean(sumSqDiff)

    return meanSumSquared 


def main():
    #nrows=40000
    pandas = pd.read_csv('Data/testing_trajectories.csv', header=None)

    # get the 980th trajectory

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

    # n = 20
    # NNInterpolationData = generateNNInterpolationData(formattedData[0], n)
    # pandas = pd.DataFrame(NNInterpolationData)
    # pandas.to_csv("NNInterpolation_Data.csv", header=None, index=None)

    # index = 5
    # plt.plot(testTrajectories[0][:,index])

    # plt.plot(NNInterpolationData[:,index])
    # plt.show()


    nValues = np.array([2, 5, 10, 20, 50, 100, 200, 500])
    error_lin, error_quad, error_NN, time_lin, time_quad, time_NN = evaluateInterpolationMethods(nValues, testTrajectories, states)


    print(error_lin)
    print(error_quad)
    print(error_NN)
    print(time_lin)
    print(time_quad)
    print(time_NN)

    np.savetxt("error_lin.csv", error_lin, delimiter=",")
    np.savetxt("error_quad.csv", error_quad, delimiter=",")
    np.savetxt("error_NN.csv", error_NN, delimiter=",")
    np.savetxt("time_lin.csv", time_lin, delimiter=",")
    np.savetxt("time_quad.csv", time_quad, delimiter=",")
    np.savetxt("time_NN.csv", time_NN, delimiter=",")
    
    x_vals = np.array([])
    plt.xlabel("steps between recalculating derivatives")
    plt.ylabel("Mean Sum Squared Error")
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



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys

print("PROGRAM START - EVALUATE INTERPOLATION METHODS")

mode = 2

if(len(sys.argv) < 2):
    print("not enough arguments")
    exit()
else:
    mode = int(sys.argv[1])

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
numTrajectoriesTest = 1

def calculateJerkOverTrajectory(trajectoryStates):
    jerk = np.zeros((trajecLength - 2, 2 * dof))


    for i in range(trajecLength - 2):

        state1 = trajectoryStates[i,:].copy()
        state2 = trajectoryStates[i+1,:].copy()
        state3 = trajectoryStates[i+2,:].copy()

        accel1 = state3 - state2
        accel2 = state1 - state1

        currentJerk = accel2 - accel1

        print(currentJerk.shape)
        print(jerk[i,:].shape)

        jerk[i,:] = currentJerk



    return jerk


def createListReupdatePoints(trajectoryStates, trajectoryControls, dynParameters):
    reEvaluatePoints = []
    counterSinceLastEval = 0
    reEvaluatePoints.append(0)

    minN = int(dynParameters[0])
    maxN = int(dynParameters[1])
    velGradSensitivty = dynParameters[2]
    velGradCubeSensitivity = dynParameters[3]

    currentGrads = np.zeros(2*dof)
    lastGrads = np.zeros(2*dof)
    first = True

    for i in range(trajecLength - 1):
        
        # if(i == 0):
        #     lastGrads = currentGrads.copy()

        if(counterSinceLastEval >= minN):

            currState = trajectoryStates[i,:].copy()
            lastState = trajectoryStates[i-1,:].copy()

            currentGrads = currState - lastState

            if(first):
                first = False
                reEvaluatePoints.append(i)
                counterSinceLastEval = 0
            else:
                if(newEvaluationNeeded(currentGrads, lastGrads, velGradSensitivty, velGradCubeSensitivity)):
                    reEvaluatePoints.append(i)
                    counterSinceLastEval = 0

            lastGrads = currentGrads.copy()
            

        if(counterSinceLastEval >= maxN):
            reEvaluatePoints.append(i)
            counterSinceLastEval = 0

        counterSinceLastEval = counterSinceLastEval + 1

    reEvaluatePoints.append(trajecLength - 1)

    return reEvaluatePoints

def newEvaluationNeeded(currentGrads, lastGrads, sensitivity, cubeSensitivity):
    newEvalNeeded = False

    for i in range(dof):
        # print("current grad: " + str(currentGrads[dof + i]))
        # print("last grad: " + str(lastGrads[dof + i]))
        velGradDiff = currentGrads[dof + i] - lastGrads[dof + i]
        # print(velGradDiff)

        if(i < 7):
            if(velGradDiff > sensitivity):
                newEvalNeeded = True
                #print("new eval needed, diff: " + str(velGradDiff))

            if(velGradDiff < -sensitivity):
                newEvalNeeded = True
                #print("new eval needed, diff: " + str(velGradDiff))

        else:
            if(velGradDiff > cubeSensitivity):
                newEvalNeeded = True
                #print("new eval needed, diff: " + str(velGradDiff))

            if(velGradDiff < -cubeSensitivity):
                newEvalNeeded = True
                #print("new eval needed, diff: " + str(velGradDiff))
        


    return newEvalNeeded

def generateDynLinInterpolation(A_matrices, reEvaluationIndicies):
    sizeofAMatrix = len(A_matrices[0])
    linInterpolationData = np.zeros((trajecLength - 1, sizeofAMatrix))
    print("size of trajec: " + str(len(A_matrices)))

    numBins = len(reEvaluationIndicies) - 1
    #print("num bins: " + str(numBins))
    stepsBetween = 0

    for i in range(numBins):

        for j in range(sizeofAMatrix):

            
            startIndex = reEvaluationIndicies[i]
            #print("start index: " + str(startIndex))
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


def evaluateInterpolationMethods(dynParameters, testTrajectories, states, controls):


    dynLin_interpolationTimes = np.zeros((len(dynParameters)))
    MAE_dynLin = np.zeros((len(dynParameters)))
    numBins = np.zeros((len(dynParameters)))

    
    MAE_perTrajec = np.zeros(len(testTrajectories))
    numBins_perTrajec = np.zeros(len(testTrajectories))
    dynInterpTimes_perTrajec = np.zeros(len(testTrajectories))
    

    linInterpolationData = []


    # loop over the test trajectories
    for i in range(len(dynParameters)):
        print("parameters: " + str(i))
        for j in range(len(testTrajectories)):
            print("trajec: " + str(j))


            start = time.process_time()
            reEvaluationIndicies = createListReupdatePoints(states[j].copy(), controls[j].copy(), dynParameters[i].copy())
            numBins_perTrajec[j] = len(reEvaluationIndicies)
            print("num evals: " + str(numBins_perTrajec[j]))
            #print(reEvaluationIndicies)
            linInterpolationData = generateDynLinInterpolation(testTrajectories[j].copy(), reEvaluationIndicies.copy())
            dynInterpTimes_perTrajec[j] = time.process_time() - start


            MAE_perTrajec[j] = calcMeanSumSquaredDiffForTrajec(linInterpolationData, testTrajectories[j].copy())

            # for k in range(20):
            #     print(k)
            #     plt.plot(testTrajectories[j][:, k])
            #     plt.plot(linInterpolationData[:, k])
            #     plt.show()

            index = np.linspace(0, 3000, 3000)
            orange = "#ffa600"

            dynamicColor = "#0057c9"
            baselineColor = "#ff8400"
            
            graphBackground = "#d6d6d6"

            if(j == 0):
                print("dynParams: " + str(i))  

                every_2nd = testTrajectories[j][::4]

                highlightedIndices = np.copy(testTrajectories[j][reEvaluationIndicies, ])
                

                print(len(index))
                print(len(every_2nd))
                plt.scatter(index, testTrajectories[j][:, 7], s=10, color = baselineColor, label='Ground truth')
                #plt.scatter(index, every_2nd[:, 7], s=20, color = "k", label='Ground truth')
                #plt.plot(testTrajectories[j][:, 7],'o', ls='-', ms=2)

                plt.scatter(reEvaluationIndicies, highlightedIndices[:, 7], s=25, color = dynamicColor, label='Ground truth')


                plt.plot(linInterpolationData[:, 7], color = dynamicColor)

                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                #ax.set_facecolor(graphBackground)

                plt.show()


        MAE_dynLin[i] = np.mean(MAE_perTrajec)
        numBins[i] = np.mean(numBins_perTrajec)
        dynLin_interpolationTimes[i] = np.mean(dynInterpTimes_perTrajec)


    return MAE_dynLin, numBins, dynLin_interpolationTimes



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

    #dynParameters = np.array([[2, 5, 0.01, 0.005], [5, 50, 0.0001, 0.005],[20, 50, 0.0001, 0.005], [50, 100, 0.01, 0.0005], [2, 50, 0.0001, 0.0005], [2, 100, 0.00001, 0.0005], [10, 50, 0.001, 0.0005]])

    # [[5, 100, 0.0001, 0.005] - decent
    # [5, 100, 0.0002, 0.0008] - first image
    dynParameters = np.array([[5, 200, 0.0002, 0.0008], [5, 50, 0.0001, 0.005],[20, 50, 0.0001, 0.005], [50, 100, 0.01, 0.0005], [2, 50, 0.0001, 0.0005], [2, 100, 0.00001, 0.0005], [10, 50, 0.001, 0.0005]])

    plt.rcParams['svg.fonttype'] = 'none'
    jerk = calculateJerkOverTrajectory(states[0])

    for i in range(dof):
        plt.plot(jerk[:,i + dof])
        path = "jerk" + str(i) + ".svg"
        plt.savefig(path, format="svg")
        plt.show()

    error_dynLin, evals_dynLin, time_dynLin = evaluateInterpolationMethods(dynParameters, testTrajectories, states, controls)


    print(error_dynLin)
    print("number of evals dynLin: " + str(evals_dynLin))
    print(time_dynLin)

    np.savetxt(startPath + "Results/error_dynLin.csv", error_dynLin, delimiter=",")
    np.savetxt(startPath + "Results/evals_dynLin.csv", evals_dynLin, delimiter=",")
    np.savetxt(startPath + "Results/time_dynLin.csv", time_dynLin, delimiter=",")
    
    # x_vals = np.array([])
    # plt.xlabel("steps between recalculating derivatives")
    # plt.ylabel("Mean Absolute Error")
    # plt.title('Comparing different interpolation methods for different step sizes')
    # default_x_ticks = range(len(error_lin))
    # plt.xticks(default_x_ticks, nValues)
    # plt.plot(error_lin, label = 'linear')
    # plt.plot(error_quad, label = 'quadratic')
    # plt.plot(error_NN, label = 'NN')
    # plt.legend()
    # plt.show()

    print("PROGRAM FINISHED")


main()



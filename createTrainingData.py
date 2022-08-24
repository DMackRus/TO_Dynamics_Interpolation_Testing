import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import sys 

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
n = 20
numTrajectories = 1000

sizeOfAMatrix = dof * 2 * dof 
sizeOfBMatrix = dof * num_ctrl


def main():
    print("MAIN FUNCTION START")

    pandas = pd.read_csv(startPath + 'NN_Data/matrices_A.csv', header=None)
    A_matrices = pandas.to_numpy()

    print("SIZE OF A MATRICES"  + str(A_matrices.shape))

    pandas = pd.read_csv(startPath + 'NN_Data/savedStates.csv', header=None)
    states = pandas.to_numpy()

    pandas = pd.read_csv(startPath + 'NN_Data/savedControls.csv', header=None)
    controls = pandas.to_numpy()

    #only needed if csv is still saving a column of nans at end
    print("size of states: " + str(states.shape))

    normalised_unpackedAMatrices = unpackMatrices(A_matrices)
    print("FINISHED UNPACKING")

    # plt.plot(normalised_unpackedAMatrices[:,1])
    # plt.show()

    # plt.plot(normalised_unpackedAMatrices[:,2])
    # plt.show()

    # plt.plot(normalised_unpackedAMatrices[:,3])
    # plt.show()

    # plt.plot(normalised_unpackedAMatrices[:,4])
    # plt.show()

    # plt.plot(normalised_unpackedAMatrices[:,5])
    # plt.show()

    normalised_unpackedAMatrices = removeOutliers(normalised_unpackedAMatrices)
    print("FINISHED REMOVING OUTLIERS")

    # plt.plot(normalised_unpackedAMatrices[:,1])
    # plt.show()

    # plt.plot(normalised_unpackedAMatrices[:,2])
    # plt.show()

    # plt.plot(normalised_unpackedAMatrices[:,3])
    # plt.show()

    # plt.plot(normalised_unpackedAMatrices[:,4])
    # plt.show()

    # plt.plot(normalised_unpackedAMatrices[:,5])
    # plt.show()

    # normally 985
    testingStartIndex = 985

    testingTrajectories = normalised_unpackedAMatrices[(testingStartIndex * trajecLength):,:]
    testingStates = states[(testingStartIndex * trajecLength):,:]
    testingControls = controls[(testingStartIndex * trajecLength):,:]

    pandas = pd.DataFrame(testingTrajectories)
    pandas.to_csv(startPath + "testing_data/testing_trajectories.csv", header=None, index=None)

    pandas = pd.DataFrame(testingStates)
    pandas.to_csv(startPath + "testing_data/testing_states.csv", header=None, index=None)

    pandas = pd.DataFrame(testingControls)
    pandas.to_csv(startPath + "testing_data/testing_controls.csv", header=None, index=None)

    print("FINISHED SAVING TESTING TRAJECTORIES, STATES AND CONTROLS")

    normalised_unpackedAMatrices, maxValA, minValA = normaliseMatrices(normalised_unpackedAMatrices)
    normalised_states, maxValStates, minValStates = normaliseStates(states)

    np.savetxt(startPath + "testing_data/maxValuesA.csv", maxValA, delimiter=",")
    np.savetxt(startPath + "testing_data/minValuesA.csv", minValA, delimiter=",")
    np.savetxt(startPath + "testing_data/maxValuesStates.csv", maxValStates, delimiter=",")
    np.savetxt(startPath + "testing_data/minValuesStates.csv", minValStates, delimiter=",")

    print("FINISHED NORMALISING MATRICES")

    pandas = pd.DataFrame(normalised_unpackedAMatrices)
    pandas.to_csv(startPath + "NN_Data/normalised_unpackedAMatrices.csv", header=None, index=None)
    print("SAVED NORMALISED UNPACKED A MATRICES")

    trainDataInputs_A, trainDataOutputs_A = createTrainingData(n, trajecLength, normalised_unpackedAMatrices, normalised_states, sizeOfAMatrix)
    print("TRAINING DATA CREATED")

    pandas = pd.DataFrame(trainDataInputs_A)
    pandas.to_csv(startPath + "NN_Data/trainDataInputs_A.csv", header=None, index=None)

    pandas = pd.DataFrame(trainDataOutputs_A)
    pandas.to_csv(startPath + "NN_Data/trainDataOutputs_A.csv", header=None, index=None)

    print("TRAINING DATA SAVED")
    print("PROGRAM EXIT")


def unpackMatrices(matricesA):
    rowAMatrix = np.zeros((1, sizeOfAMatrix))
    unpackedAMatrices = np.zeros((numTrajectories * trajecLength, sizeOfAMatrix))
    for i in range((trajecLength * numTrajectories)):
        if(i % 3000 == 0):
            print("trajec done: " + str(i/3000))

        for j in range(dof):
            for k in range(dof * 2):
                rowAMatrix[0, (j * dof * 2) + k] = matricesA[(i * dof) + j, k]

        unpackedAMatrices[i,:] = rowAMatrix
    return unpackedAMatrices

def removeOutliers(matrices):
    for i in range(numTrajectories * trajecLength):

        for j in range(sizeOfAMatrix):
            if(matrices[i, j] > 50):
                if(i % trajecLength == 0):
                    matrices[i, j] = 50
                else:
                    matrices[i, j] = matrices[i - 1, j]

            if(matrices[i, j] < -50):
                if(i % trajecLength == 0):
                    matrices[i, j] = -50
                else:
                    matrices[i, j] = matrices[i - 1, j]
    
    return matrices


def normaliseMatrices(matrices):
    normalisedMatrices = np.zeros((numTrajectories * trajecLength, sizeOfAMatrix))

    maxValA = np.zeros(sizeOfAMatrix)
    minValA = np.zeros(sizeOfAMatrix)

    for j in range(sizeOfAMatrix):
        maxValA[j] = max(matrices[:, j])
        minValA[j] = min(matrices[:, j])

    # number of trajectories
    for i in range(numTrajectories):
        startIndex = (i * trajecLength)
        endIndex = ((i + 1) * trajecLength)

        for j in range(sizeOfAMatrix):

            for k in range(trajecLength):
                
                normalisedMatrices[startIndex + k, j] = (matrices[(i * trajecLength) + k, j] - minValA[j]) / (maxValA[j] - minValA[j])

    return normalisedMatrices, maxValA, minValA

def normaliseStates(states):

    normalisedStates = np.zeros((numTrajectories * trajecLength, 2 * dof))

    maxValStates = np.zeros(2 * dof)
    minValStates = np.zeros(2 * dof)

    for j in range(2 * dof):
        maxValStates[j] = max(states[:, j])
        minValStates[j] = min(states[:, j])

    for i in range(numTrajectories):
        startIndex = (i * trajecLength)
        endIndex = ((i + 1) * trajecLength)

        for j in range(2 * dof):
            for k in range(trajecLength):
                
                normalisedStates[startIndex + k, j] = (states[(i * trajecLength) + k, j] - minValStates[j]) / (maxValStates[j] - minValStates[j])

    return normalisedStates, maxValStates, minValStates



def createTrainingData(n , trajecLength, trajecData, states, sizeOfMatrix):

    bins = int((trajecLength * numTrajectories) / n) - 1

    listsInputs = np.zeros((bins * n, (2 * sizeOfMatrix) + 1 + (2 * dof))) 
    listsOutputs = np.zeros((bins * n, sizeOfMatrix))
    
    print("number of bins " + str(bins))
    print("length of data " + str(len(trajecData)))

    for i in range(bins):

        startIndex = (i * (n))
        endIndex = ((i + 1) * (n))
        startRow = trajecData[startIndex,:]
        endRow = trajecData[endIndex,:]
        
        row = np.zeros((1, (2 * sizeOfMatrix) + 1 + (2 * dof)))
        rowOutput = np.zeros((1, sizeOfMatrix))

        row[0, 0:sizeOfMatrix] = startRow
        row[0, sizeOfMatrix:(2*sizeOfMatrix)] = endRow

        if(i % 500 == 0):
            print("data " + str(i))
        for j in range(0, n):

            
            row[0, (2*sizeOfMatrix):(2*sizeOfMatrix) + (2 * dof)] = states[startIndex + j,:]
            row[0, -1] = (j / n)

            listsInputs[(i * n) + j,:] = row

            rowOutput = trajecData[startIndex + j,:]

            listsOutputs[(i * n) + j,:] = rowOutput

    return listsInputs, listsOutputs




main()

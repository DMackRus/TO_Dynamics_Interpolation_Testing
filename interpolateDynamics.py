import numpy as np
import pandas as pd


numTrajectoriesTest = 1
class interpolator():
    def __init__(self, trajecNumber, task, trajecLength):

        startPath = ""

        if(task == 0):
            dof = 2
            num_ctrl = 2
            startPath = "Pendulum/"
        elif(task == 1):
            dof = 7
            num_ctrl = 7
            startPath = "Reaching/"

        elif(task == 2):
            dof = 9
            num_ctrl = 7
            startPath = "Pushing/"

        else:
            print("invalid mode specified")
            exit()

        self.dof = dof
        self.num_ctrl = num_ctrl
        self.numStates = self.dof * 2
        self.sizeOfAMatrix = self.numStates * self.numStates
        self.trajecLength = trajecLength

        pandas = pd.read_csv(startPath + 'testing_data/testing_trajectories.csv', header=None)

        self.testTrajectories = []
        self.states = []
        self.controls = []

        for i in range(numTrajectoriesTest):
            self.testTrajectories.append([])
            tempPandas = pandas.iloc[i*trajecLength:(i + 1)*trajecLength]
            self.testTrajectories[i] = tempPandas.to_numpy()
    
        pandas = pd.read_csv(startPath + 'testing_data/testing_states.csv', header=None)

        for i in range(numTrajectoriesTest):
            self.states.append([])
            tempPandas = pandas.iloc[i*trajecLength:(i + 1)*trajecLength]
            self.states[i] = tempPandas.to_numpy()

        pandas = pd.read_csv(startPath + 'testing_data/testing_controls.csv', header=None)

        for i in range(numTrajectoriesTest):
            self.controls.append([])
            tempPandas = pandas.iloc[i*trajecLength:(i + 1)*trajecLength]
            self.controls[i] = tempPandas.to_numpy()

        print("DATA LOADED")

        self.dynParams = []
        self.error_dynLin = []
        self.evals_dynLin = []
        self.time_dynLin = []
        self.displayData_raw = []
        self.displayData_inteprolated = []

    def interpolateTrajectory(self, trajecNumber, dynParams, interpMethod):
        rawTrajec = self.testTrajectories[trajecNumber]

        self.dynParams = dynParams
        #jerkProfile = self.calculateJerkOverTrajectory()

        reEvaluationIndices = self.createEvaluationPoints(self.states[0].copy(), self.controls[0].copy(), self.dynParams.copy())
        numEvals = len(reEvaluationIndices)
        print("num evals: " + str(numEvals))
        #print(reEvaluationIndicies)
        if(interpMethod == 0):
            interpolatedTrajectory = self.generateLinInterpolation(rawTrajec, reEvaluationIndices.copy())
        else:
            interpolatedTrajectory = self.generateQuadInterpolation(rawTrajec, reEvaluationIndices.copy())
        print(interpolatedTrajectory.shape)

        return rawTrajec, interpolatedTrajectory, reEvaluationIndices
    
    def returnTrajecInformation(self):

        self.jerkProfile = self.calcJerkOverTrajectory(self.states[0])


        return self.jerkProfile, self.states[0].copy(), self.controls[0].copy()



    def calcJerkOverTrajectory(self, trajectoryStates):
        print("trajectory states shape: " + str(trajectoryStates.shape))
        jerk = np.zeros((self.trajecLength - 2, self.numStates))

        for i in range(self.trajecLength - 2):

            state1 = trajectoryStates[i,:].copy()
            state2 = trajectoryStates[i+1,:].copy()
            state3 = trajectoryStates[i+2,:].copy()

            accel1 = state3 - state2
            accel2 = state1 - state1

            currentJerk = accel2 - accel1

            jerk[i,:] = currentJerk

        return jerk
    
    def createEvaluationPoints(self, trajectoryStates, trajectoryControls, dynParameters):
        reEvaluatePoints = []
        counterSinceLastEval = 0
        reEvaluatePoints.append(0)

        minN = int(dynParameters[0])
        maxN = int(dynParameters[1])
        velGradSensitivty = dynParameters[2]
        # temp
        velGradCubeSensitivity = 0.0002

        currentGrads = np.zeros(self.numStates)
        lastGrads = np.zeros(self.numStates)
        first = True

        for i in range(self.trajecLength - 1):

            if(counterSinceLastEval >= minN):

                currState = trajectoryStates[i,:].copy()
                lastState = trajectoryStates[i-1,:].copy()

                currentGrads = currState - lastState

                if(first):
                    first = False
                    reEvaluatePoints.append(i)
                    counterSinceLastEval = 0
                else:
                    if(self.newEvaluationNeeded(currentGrads, lastGrads, velGradSensitivty, velGradCubeSensitivity)):
                        reEvaluatePoints.append(i)
                        counterSinceLastEval = 0

                lastGrads = currentGrads.copy()
            
            if(counterSinceLastEval >= maxN):
                reEvaluatePoints.append(i)
                counterSinceLastEval = 0

            counterSinceLastEval = counterSinceLastEval + 1

        reEvaluatePoints.append(self.trajecLength - 1)

        return reEvaluatePoints
    

    def newEvaluationNeeded(self, currentGrads, lastGrads, sensitivity, cubeSensitivity):
        newEvalNeeded = False

        for i in range(self.dof):
            # print("current grad: " + str(currentGrads[dof + i]))
            # print("last grad: " + str(lastGrads[dof + i]))
            velGradDiff = currentGrads[self.dof + i] - lastGrads[self.dof + i]
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
    
    def generateLinInterpolation(self, A_matrices, reEvaluationIndicies):
        sizeofAMatrix = len(A_matrices[0])
        linInterpolationData = np.zeros((self.trajecLength - 1, sizeofAMatrix))
        print("size of trajec: " + str(len(A_matrices)))

        numBins = len(reEvaluationIndicies) - 1
        #print("num bins: " + str(numBins))
        stepsBetween = 0

        for i in range(numBins):
                
            startIndex = reEvaluationIndicies[i]
            startVals = A_matrices[startIndex,:]

            endIndex = reEvaluationIndicies[i + 1]
            endVals = A_matrices[endIndex,:]

            diff = endVals - startVals

            stepsBetween = endIndex - startIndex

            linInterpolationData[startIndex,:] = startVals

            for k in range(1, stepsBetween):
                linInterpolationData[startIndex + k,:] = startVals + (diff * (k/stepsBetween))

        return linInterpolationData
    
    def generateQuadInterpolation(self, A_matrices, reEvaluationIndicies):
        sizeofMatrix = len(A_matrices[0])
        quadInterpolationData = np.zeros((self.trajecLength - 1, sizeofMatrix))

        for i in range(len(reEvaluationIndicies) - 2):
            startIndex = reEvaluationIndicies[i]
            midIndex = reEvaluationIndicies[i + 1]
            endIndex = reEvaluationIndicies[i + 2]

            for j in range(sizeofMatrix):

                if(j == 7):
                    gdsgdfg = 1

                points = np.zeros((3, 2))
                point1 = np.array([A_matrices[startIndex, j], startIndex])
                point2 = np.array([A_matrices[midIndex, j], midIndex])
                point3 = np.array([A_matrices[endIndex, j], endIndex])

                points[0] = point1
                points[1] = point2
                points[2] = point3

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

                    # x_matrix[0, k] = (endIndex - startIndex) * (endIndex - startIndex)
                    # x_matrix[1, k] = (endIndex - startIndex)
                    # x_matrix[2, k] = 1

                x_inv = np.linalg.inv(x_matrix)

                abc = y_matrix @ x_inv

                quadInterpolationData[startIndex,j] = A_matrices[startIndex,j]
                print("start val: " + str(A_matrices[startIndex,j]))
                print("c: " + str(abc[0,2]))

                counter = 0
                for k in range(startIndex, endIndex):
                    a = abc[0, 0]
                    b = abc[0, 1]
                    c = abc[0, 2]

                    nextVal = (a * k * k) + (b * k) + c
                    quadInterpolationData[startIndex + counter, j] = nextVal
                    counter = counter + 1
                    # quadInterpolationData[startIndex + k, j] = A_matrices[startIndex + k,j]

        return quadInterpolationData


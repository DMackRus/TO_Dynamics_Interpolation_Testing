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

    def interpolateTrajectory(self, trajecNumber, dynParams):
        rawTrajec = self.testTrajectories[trajecNumber]

        self.dynParams = dynParams
        #jerkProfile = self.calculateJerkOverTrajectory()

        reEvaluationIndices = self.createEvaluationPoints(self.states[0].copy(), self.controls[0].copy(), self.dynParams.copy())
        numEvals = len(reEvaluationIndices)
        print("num evals: " + str(numEvals))
        #print(reEvaluationIndicies)
        interpolatedTrajectory = self.generateLinInterpolation(rawTrajec, reEvaluationIndices.copy())
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


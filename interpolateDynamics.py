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

        iterativeKeyPoints = self.generateIterativeLinInterpolation(rawTrajec)
        print("evals: ")
        print(reEvaluationIndices)
        print("iterative key points: ")
        print(iterativeKeyPoints)
        iterativeLinInterpTrajectory = self.generateLinInterpolation(rawTrajec, iterativeKeyPoints.copy())
        linInterpTrajectory = self.generateLinInterpolation(rawTrajec, reEvaluationIndices.copy())
        quadInterpTrajectory = self.generateQuadInterpolation(rawTrajec, reEvaluationIndices.copy())
        cubicInterpTrajectory = self.generateCubicInterpolation(rawTrajec, reEvaluationIndices.copy())
        #print(reEvaluationIndicies)

        #store lininterp and quadratic interp into interpolateTrajectory
        interpolatedTrajectory = np.zeros((4, self.trajecLength, len(rawTrajec[0])))
        errors = np.zeros((4))
        interpolatedTrajectory[0,:,:] = linInterpTrajectory.copy()
        interpolatedTrajectory[1,:,:] = iterativeLinInterpTrajectory.copy()
        interpolatedTrajectory[2,:,:] = quadInterpTrajectory.copy()
        interpolatedTrajectory[3,:,:] = cubicInterpTrajectory.copy()
        

        errors[0] = self.calcMeanSumSquaredDiffForTrajec(rawTrajec, linInterpTrajectory)
        errors[1] = self.calcMeanSumSquaredDiffForTrajec(rawTrajec, quadInterpTrajectory)
        errors[2] = self.calcMeanSumSquaredDiffForTrajec(rawTrajec, cubicInterpTrajectory)
        errors[3] = self.calcMeanSumSquaredDiffForTrajec(rawTrajec, iterativeLinInterpTrajectory)

        return rawTrajec, interpolatedTrajectory, errors, reEvaluationIndices, iterativeKeyPoints
    
    def calcMeanSumSquaredDiffForTrajec(self, groundTruth, prediction):
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

            meanSqDiff[j] = np.sum(SqDiff[:,j])

        #print("sum squared diff matrices: " + str(sumSqDiff))
        meanSumSquared = np.sum(meanSqDiff)

        return meanSumSquared 
    
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
    
    def generateIterativeLinInterpolation(self, A_matrices):
        evalPoints = []

        startInterval = int(self.trajecLength / 2)
        numMaxBins = int((self.trajecLength / startInterval))

        for i in range(numMaxBins):
            binComplete = False
            startIndex = i * startInterval
            endIndex = (i + 1) * startInterval
            if(endIndex >= self.trajecLength):
                endIndex = self.trajecLength - 1
            listofIndicesCheck = []
            indexTuple = (startIndex, endIndex)
            listofIndicesCheck.append(indexTuple)
            subListIndices = []
            subListWithMidpoints = []

            while(not binComplete):

                allChecksComplete = True
                for j in range(len(listofIndicesCheck)):

                    approximationGood, midIndex = self.oneCheck(A_matrices, listofIndicesCheck[j])

                    if not approximationGood:
                        allChecksComplete = False
                        
                        indexTuple1 = (listofIndicesCheck[j][0], midIndex)
                        indexTuple2 = (midIndex, listofIndicesCheck[j][1])
                        subListIndices.append(indexTuple1)
                        subListIndices.append(indexTuple2)
                    else:
                        subListWithMidpoints.append(listofIndicesCheck[j][0])
                        subListWithMidpoints.append(midIndex)
                        subListWithMidpoints.append(listofIndicesCheck[j][1])

                if(allChecksComplete):
                    binComplete = True
                    for k in range(len(subListWithMidpoints)):
                        evalPoints.append(subListWithMidpoints[k])

                    subListWithMidpoints = []

                listofIndicesCheck = subListIndices.copy()
                subListIndices = []

        evalPoints.sort()
        evalPoints = list(dict.fromkeys(evalPoints))

        return evalPoints
    
    def oneCheck(self, A_matrices, indexTuple):
        approximationGood = False

        startIndex = indexTuple[0]
        endIndex = indexTuple[1]

        midIndex = int((startIndex + endIndex) / 2)
        startVals = A_matrices[startIndex,:]
        endVals = A_matrices[endIndex,:]

        if((endIndex - startIndex) < 5):
            return True, midIndex

        trueMidVals = A_matrices[midIndex,:]
        diff = endVals - startVals
        linInterpMidVals = startVals + (diff/2)

        sumsqDiff = self.sumsqDiffBetweenAMatrices(trueMidVals, linInterpMidVals)

        if(sumsqDiff < 0.1):
            approximationGood = True

        return approximationGood, midIndex
    
    def sumsqDiffBetweenAMatrices(self, matrix1, matrix2):
        sumsqDiff = 0

        for i in range(len(matrix1)):
            sumsqDiff = sumsqDiff + (matrix1[i] - matrix2[i])**2

        return sumsqDiff

    
    def generateLinInterpolation(self, A_matrices, reEvaluationIndicies):
        sizeofAMatrix = len(A_matrices[0])
        linInterpolationData = np.zeros((self.trajecLength, sizeofAMatrix))

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

        
        linInterpolationData[len(linInterpolationData) - 1,:] = linInterpolationData[len(linInterpolationData) - 2,:]
        return linInterpolationData
    
    def generateQuadInterpolation(self, A_matrices, reEvaluationIndicies):
        sizeofMatrix = len(A_matrices[0])
        quadInterpolationData = np.zeros((self.trajecLength, sizeofMatrix))

        for i in range(len(reEvaluationIndicies) - 2):
            startIndex = reEvaluationIndicies[i]
            midIndex = reEvaluationIndicies[i + 1]
            endIndex = reEvaluationIndicies[i + 2]

            for j in range(sizeofMatrix):

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

                x_inv = np.linalg.inv(x_matrix)

                abc = y_matrix @ x_inv

                quadInterpolationData[startIndex,j] = A_matrices[startIndex,j]

                counter = 0
                for k in range(startIndex, endIndex):
                    a = abc[0, 0]
                    b = abc[0, 1]
                    c = abc[0, 2]

                    nextVal = (a * k * k) + (b * k) + c
                    quadInterpolationData[startIndex + counter, j] = nextVal
                    counter = counter + 1

        quadInterpolationData[len(quadInterpolationData) - 1,:] = quadInterpolationData[len(quadInterpolationData) - 2,:]

        return quadInterpolationData
    
    def generateCubicInterpolation(self, A_matrices, reEvaluationIndicies):
        sizeofMatrix = len(A_matrices[0])
        quadInterpolationData = np.zeros((self.trajecLength, sizeofMatrix))

        for i in range(len(reEvaluationIndicies) - 3):
            startIndex = reEvaluationIndicies[i]
            midIndex1 = reEvaluationIndicies[i + 1]
            midIndex2 = reEvaluationIndicies[i + 2]
            endIndex = reEvaluationIndicies[i + 3]

            for j in range(sizeofMatrix):

                points = np.zeros((4, 2))
                point1 = np.array([A_matrices[startIndex, j], startIndex])
                point2 = np.array([A_matrices[midIndex1, j], midIndex1])
                point3 = np.array([A_matrices[midIndex2, j], midIndex2])
                point4 = np.array([A_matrices[endIndex, j], endIndex])

                points[0] = point1
                points[1] = point2
                points[2] = point3
                points[3] = point4

                #solve for coefficients a, b, c

                x_matrix = np.zeros((4, 4))
                y_matrix = np.zeros((1, 4))

                y_matrix[0, 0] = points[0, 0]
                y_matrix[0, 1] = points[1, 0]
                y_matrix[0, 2] = points[2, 0]
                y_matrix[0, 3] = points[3, 0]

                for k in range(4):
                    x_matrix[0, k] = points[k, 1] * points[k, 1] * points[k, 1]
                    x_matrix[1, k] = points[k, 1] * points[k, 1]
                    x_matrix[2, k] = points[k, 1]
                    x_matrix[3, k] = 1

                x_inv = np.linalg.inv(x_matrix)

                abcd = y_matrix @ x_inv

                quadInterpolationData[startIndex,j] = A_matrices[startIndex,j]

                counter = 0
                for k in range(startIndex, endIndex):
                    a = abcd[0, 0]
                    b = abcd[0, 1]
                    c = abcd[0, 2]
                    d = abcd[0, 3]

                    nextVal = (a * k * k * k) + (b * k * k) + (c * k) + d
                    quadInterpolationData[startIndex + counter, j] = nextVal
                    counter = counter + 1

        quadInterpolationData[len(quadInterpolationData) - 1,:] = quadInterpolationData[len(quadInterpolationData) - 2,:]

        return quadInterpolationData



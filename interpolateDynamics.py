import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
import math

numTrajectoriesTest = 1

class interpolator():
    def __init__(self, task):

        startPath = "savedTrajecInfo/" + task + "/"
        self.trajecNumber = 5
        
        pandas = pd.read_csv(startPath + str(self.trajecNumber) + '/A_matrices.csv', header=None)
        pandas = pandas[pandas.columns[:-1]]
        rows, cols = pandas.shape

        self.task = task
        self.trajecLength = rows 

        self.testTrajectories = []
        self.states = []
        self.controls = []

        for i in range(numTrajectoriesTest):
            self.testTrajectories.append([])
            tempPandas = pandas.iloc[i*self.trajecLength:(i + 1)*self.trajecLength]
            self.testTrajectories[i] = tempPandas.to_numpy()
        pandas = pd.read_csv(startPath + str(self.trajecNumber) + '/states.csv', header=None)
        pandas = pandas[pandas.columns[:-1]]
        rows, cols = pandas.shape
        self.dof = int(cols/2)
        self.numStates = self.dof * 2
        self.sizeOfAMatrix = self.numStates * self.numStates

        for i in range(numTrajectoriesTest):
            self.states.append([])
            tempPandas = pandas.iloc[i*self.trajecLength:(i + 1)*self.trajecLength]
            self.states[i] = tempPandas.to_numpy()
        pandas = pd.read_csv(startPath + str(self.trajecNumber) +  '/controls.csv', header=None)
        pandas = pandas[pandas.columns[:-1]]
        rows, cols = pandas.shape
        self.num_ctrl = cols

        for i in range(numTrajectoriesTest):
            self.controls.append([])
            tempPandas = pandas.iloc[i*self.trajecLength:(i + 1)*self.trajecLength]
            self.controls[i] = tempPandas.to_numpy()

        print("dof is: " + str(self.dof))
        print("num_ctrl is: " + str(self.num_ctrl))
        print("trajec length " + str(self.trajecLength))    

        if(task == 2):
            T = 5.0         # Sample Period
            fs = 100.0      # sample rate, Hz
            cutoff = 1      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
            nyq = 0.5 * fs  # Nyquist Frequency
            order = 2       # sin wave can be approx represented as quadratic
            n = int(T * fs) # total number of samples

            self.filteredTrajectory = self.testTrajectories[0].copy()

            for i in range(len(self.testTrajectories[0][0])):
                
                self.filteredTrajectory[:,i] = self.butter_lowpass_filter(self.testTrajectories[0][:,i].copy(), cutoff, nyq, order)

        else:
            self.filteredTrajectory = self.testTrajectories[0].copy()

        self.dynParams = []
        self.error_dynLin = []
        self.evals_dynLin = []
        self.time_dynLin = []
        self.displayData_raw = []
        self.displayData_inteprolated = []

    def interpolateTrajectory(self, trajecNumber, dynParams):
        rawTrajec = self.testTrajectories[trajecNumber]

        self.dynParams = dynParams
        keyPoints = self.generateKeypoints(rawTrajec, self.controls[0].copy(), self.dynParams.copy())

        # print("keypoints generated")
        # print(keyPoints)
        #Im here

        setIntervalTrajectory = self.generateLinInterpolation(rawTrajec, keyPoints[0].copy())
        adaptiveJerkTrajectory = self.generateLinInterpolation(rawTrajec, keyPoints[1].copy())
        adaptiveAccellTrajectory = self.generateLinInterpolation(rawTrajec, keyPoints[2].copy())
        iterativeErrorTrajectory = self.generateLinInterpolation(rawTrajec, keyPoints[3].copy())

        #store lininterp and quadratic interp into interpolateTrajectory
        interpolatedTrajectory = np.zeros((4, self.trajecLength, len(rawTrajec[0])))
        errors = np.zeros((4))
        interpolatedTrajectory[0,:,:] = setIntervalTrajectory.copy()
        interpolatedTrajectory[1,:,:] = adaptiveJerkTrajectory.copy()
        interpolatedTrajectory[2,:,:] = adaptiveAccellTrajectory.copy()
        interpolatedTrajectory[3,:,:] = iterativeErrorTrajectory.copy()
        
        if(self.task == 2):
            errors[0] = self.calcMeanSumSquaredDiffForTrajec(self.filteredTrajectory, setIntervalTrajectory)
            errors[1] = self.calcMeanSumSquaredDiffForTrajec(self.filteredTrajectory, adaptiveJerkTrajectory)
            errors[2] = self.calcMeanSumSquaredDiffForTrajec(self.filteredTrajectory, adaptiveAccellTrajectory)
            errors[3] = self.calcMeanSumSquaredDiffForTrajec(self.filteredTrajectory, iterativeErrorTrajectory)
        else:
            errors[0] = self.calcMeanSumSquaredDiffForTrajec(rawTrajec, setIntervalTrajectory)
            errors[1] = self.calcMeanSumSquaredDiffForTrajec(rawTrajec, adaptiveJerkTrajectory)
            errors[2] = self.calcMeanSumSquaredDiffForTrajec(rawTrajec, adaptiveAccellTrajectory)
            errors[3] = self.calcMeanSumSquaredDiffForTrajec(rawTrajec, iterativeErrorTrajectory)

        return self.filteredTrajectory, interpolatedTrajectory, rawTrajec, errors, keyPoints
    
    def calcMeanSumSquaredDiffForTrajec(self, groundTruth, prediction):

        array1Size = len(groundTruth)
        array2Size = len(prediction)
        lenofTrajec = array2Size
        
        if(array1Size < array2Size):
            lenofTrajec = array1Size
        
        meanSqDiff = np.zeros((lenofTrajec))

        for i in range(lenofTrajec):
            meanSqDiff[i] = self.sumsqDiffBetweenAMatrices(groundTruth[i], prediction[i])
            
            # for j in range(size):

                # diffVals = abs((groundTruth[i, j] - prediction[i, j]))
                # SqDiff[i, j] = diffVals

        # for j in range(size):
        #     # stddev = np.std(SqDiff[:,j])
        #     # mean = np.mean(SqDiff[:,j])

        #     # ok = SqDiff[:,j] > (mean - (3 * stddev))
        #     # SqDiff[~ok,j] = mean

        #     # #step 2, values higher than 1 std from mean
        #     # # ok = SqDiff[:,j] < (mean + ( 3 * stddev))
        #     # # SqDiff[~ok,j] = mean

        #     meanSqDiff[j] = np.sum(SqDiff[:,j])

        #print("sum squared diff matrices: " + str(sumSqDiff))
        meanSumSquared = np.sum(meanSqDiff)

        return meanSumSquared 
    
    def returnTrajecInformation(self):

        self.jerkProfile = self.calcJerkOverTrajectory(self.states[0])
        self.accelProfile = self.calculateAccellerationOverTrajectory(self.states[0])


        return self.jerkProfile, self.accelProfile, self.states[0].copy(), self.controls[0].copy()

    def calculateAccellerationOverTrajectory(self, trajectoryStates):
        first_order_state_change = np.zeros((self.trajecLength - 1, self.numStates))

        for i in range(self.trajecLength - 1):

            state1 = trajectoryStates[i,:].copy()
            state2 = trajectoryStates[i+1,:].copy()

            currentAccel = state2 - state1

            first_order_state_change[i,:] = currentAccel

        accel = np.zeros((self.trajecLength - 2, self.dof))
        for i in range(self.trajecLength - 2):
            for j in range(self.dof):
                accel[i,j] = first_order_state_change[i, j + self.dof]

        return accel

    def calcJerkOverTrajectory(self, trajectoryStates):
        second_order_state_change = np.zeros((self.trajecLength - 2, self.numStates))

        for i in range(self.trajecLength - 2):

            state1 = trajectoryStates[i,:].copy()
            state2 = trajectoryStates[i+1,:].copy()
            state3 = trajectoryStates[i+2,:].copy()

            accel1 = state3 - state2
            accel2 = state1 - state1

            currentJerk = accel2 - accel1

            second_order_state_change[i,:] = currentJerk

        jerk = np.zeros((self.trajecLength - 2, self.dof))
        for i in range(self.trajecLength - 2):
            for j in range(self.dof):
                jerk[i,j] = second_order_state_change[i, j + self.dof]

        return jerk
    
    def generateKeypoints(self, trajectoryStates, trajectoryControls, dynParameters):
        keyPoints = [[],[],[],[]]

        keyPoints_setInterval = self.keyPoints_setInterval(dynParameters)
        keyPoints_adaptiveJerk = self.keyPoints_adaptiveJerk(trajectoryStates, dynParameters)
        keyPoints_adaptiveAccel = self.keyPoints_adaptiveAccel(trajectoryStates, dynParameters)
        keyPoints_iteratively = self.keyPoints_iteratively(trajectoryStates, dynParameters)

        max_length = max(len(keyPoints_setInterval), len(keyPoints_adaptiveJerk), len(keyPoints_adaptiveAccel), len(keyPoints_iteratively))

        # Stack the keypoints into one array
        # keyPoints = np.vstack((keyPoints_setInterval, keyPoints_adaptiveJerk, keyPoints_adaptiveAccel, keyPoints_iteratively))
        keyPoints[0] = keyPoints_setInterval
        keyPoints[1] = keyPoints_adaptiveAccel
        keyPoints[2] = keyPoints_adaptiveJerk
        keyPoints[3] = keyPoints_iteratively


        return keyPoints
    
    def keyPoints_setInterval(self, dynParameters):
        keyPoints = [[] for x in range(self.dof)]

        for i in range(self.dof):
            keyPoints[i].append(0)

        minN = int(dynParameters[0])
    
        for i in range(self.dof):
            counter = 0
            for j in range(self.trajecLength - 1):
                counter += 1
                if counter >= minN:
                    counter = 0
                    keyPoints[i].append(j)

        for i in range(self.dof):
            keyPoints[i].append(self.trajecLength - 1)

        return keyPoints 
    
    def keyPoints_adaptive_velocity(self, trajectoryStates, dynParameters):
        mainKeyPoints = [[] for x in range(self.dof)]
        keyPoints = [[] for x in range(self.dof)]

        last_direction = [0] * self.dof

        for i in range(self.dof):
            mainKeyPoints[i].append(0)

        velProfile = self.states[0][:, self.dof:2*self.dof]

        direction_temp = []

        for j in range(1, len(velProfile)):
            direction_temp.append(velProfile[j, 2] - velProfile[j-1, 2])

        # plt.plot(velProfile[:, 2])
        # plt.plot(direction_temp)
        # plt.show()
        # velProfile = self.calculateAccellerationOverTrajectory(self.states[0])

        for i in range(self.dof):
            for j in range(1, len(velProfile)):

                current_direction = velProfile[j, i] - velProfile[j-1, i]

                change_in_direction = current_direction * last_direction[i]

                if(change_in_direction <= 0):
                    mainKeyPoints[i].append(j - 1)
                else:
                    pass
                    # if(abs(current_direction - last_direction[i]) > 0.0002):
                    #     mainKeyPoints[i].append(j)
                    
                last_direction[i] = current_direction

        for i in range(self.dof):
            for j in range(len(mainKeyPoints[i])):
                if(j == 0):
                    keyPoints[i].append(mainKeyPoints[i][j])
                else:
                    keyPoints[i].append(mainKeyPoints[i][j])

        for i in range(self.dof):
            keyPoints[i].append(self.trajecLength - 1)
        
        return keyPoints 

    
    # def keyPoints_adaptiveJerk(self, trajectoryStates, dynParameters):
    #     mainKeyPoints = [[] for x in range(self.dof)]
    #     keyPoints = [[] for x in range(self.dof)]

    #     for i in range(self.dof):
    #         mainKeyPoints[i].append(0)

    #     counterSinceLastEval = np.zeros((self.dof))
    #     # outsideHysterisis = [False] * self.dof
    #     # resetToZeroAddKeypoint = [False] * self.dof
    #     # -1 is down, 0 is across and 1 is upwards
    #     last_direction = [0] * self.dof

    #     minN = int(dynParameters[0])
    #     maxN = int(dynParameters[1])
    #     jerkSensitivity = dynParameters[2]
    #     # temp
    #     # velGradCubeSensitivity = 0.0002

    #     jerkProfile = self.calcJerkOverTrajectory(self.states[0])

    #     for i in range(self.dof):
    #         for j in range(1, len(jerkProfile)):

    #             current_direction = jerkProfile[j, i] - jerkProfile[j-1, i]
    #             # print(current_direction)
    #             # if(current_direction > 0.0001):
    #             #     current_direction = 1
    #             # elif(current_direction < -0.0001):
    #             #     current_direction = -1
    #             # else:
    #             #     current_direction = 0

    #             change_in_direction = current_direction * last_direction[i]

    #             if(change_in_direction <= -0.000000000002):
    #                 mainKeyPoints[i].append(j)
                    
    #             last_direction[i] = current_direction


    #             # if(outsideHysterisis[i] == True):
    #             #     if(jerkProfile[j, i] < jerkSensitivity and jerkProfile[j, i] > -jerkSensitivity):
    #             #         mainKeyPoints[i].append(j)
    #             #         outsideHysterisis[i] = False
    #             #         resetToZeroAddKeypoint[i] = True
    #             #         counterSinceLastEval[i] = 0
    #             #     else:
    #             #         pass
    #             #         # counterSinceLastEval[i] += 1
    #             #         # if(counterSinceLastEval[i] >= minN):
    #             #         #     mainKeyPoints[i].append(j)
    #             #         #     counterSinceLastEval[i] = 0

    #             # else:
    #             #     if(jerkProfile[j, i] > jerkSensitivity or jerkProfile[j, i] < -jerkSensitivity):
    #             #         # keyPoints[i].append(j-5)
    #             #         if(mainKeyPoints[i][-1] != j-1):
    #             #             mainKeyPoints[i].append(j-1)
    #             #         mainKeyPoints[i].append(j)
    #             #         outsideHysterisis[i] = True
    #             #         resetToZeroAddKeypoint[i] = False
    #             #         counterSinceLastEval[i] = 0

    #             #     if(resetToZeroAddKeypoint[i] == True):
    #             #         # print("reset to zero add keypoint")
    #             #         if(jerkProfile[j, i] < 0.000001 and jerkProfile[j, i] > -0.000001):
    #             #             mainKeyPoints[i].append(j)
    #             #             resetToZeroAddKeypoint[i] = False
    #             #             counterSinceLastEval[i] = 0

    #             #     if(counterSinceLastEval[i] >= maxN):
    #             #         mainKeyPoints[i].append(j)
    #             #         counterSinceLastEval[i] = 0

    #             #     counterSinceLastEval[i] = counterSinceLastEval[i] + 1


    #     for i in range(self.dof):
    #         for j in range(len(mainKeyPoints[i])):
    #             if(j == 0):
    #                 keyPoints[i].append(mainKeyPoints[i][j])
    #             else:
    #                 # keyPoints[i].append(mainKeyPoints[i][j] - 15)
    #                 # keyPoints[i].append(mainKeyPoints[i][j] - 10)
    #                 # keyPoints[i].append(mainKeyPoints[i][j] - 5)
    #                 keyPoints[i].append(mainKeyPoints[i][j])
    #                 # keyPoints[i].append(mainKeyPoints[i][j] + 5)
    #                 # keyPoints[i].append(mainKeyPoints[i][j] + 10)
    #                 # keyPoints[i].append(mainKeyPoints[i][j] + 15)

    #     for i in range(self.dof):
    #         keyPoints[i].append(self.trajecLength - 1)
        
    #     return keyPoints 

    def keyPoints_adaptiveJerk(self, trajectoryStates, dynParameters):
        mainKeyPoints = [[] for x in range(self.dof)]
        keyPoints = [[] for x in range(self.dof)]

        for i in range(self.dof):
            mainKeyPoints[i].append(0)

        counterSinceLastEval = np.zeros((self.dof))
        outsideHysterisis = [False] * self.dof
        resetToZeroAddKeypoint = [False] * self.dof

        minN = int(dynParameters[0])
        maxN = int(dynParameters[1])
        jerkSensitivity = dynParameters[2]
        # temp
        # velGradCubeSensitivity = 0.0002

        jerkProfile = self.calcJerkOverTrajectory(self.states[0])

        for i in range(self.dof):
            for j in range(1, len(jerkProfile)):

                # if(outsideHysterisis[i] == True):
                #     if(jerkProfile[j, i] < jerkSensitivity and jerkProfile[j, i] > -jerkSensitivity):
                #         mainKeyPoints[i].append(j)
                #         outsideHysterisis[i] = False
                #         resetToZeroAddKeypoint[i] = True
                #         counterSinceLastEval[i] = 0
                #     else:
                #         pass
                #         # counterSinceLastEval[i] += 1
                #         # if(counterSinceLastEval[i] >= minN):
                #         #     mainKeyPoints[i].append(j)
                #         #     counterSinceLastEval[i] = 0

                # else:
                #     if(jerkProfile[j, i] > jerkSensitivity or jerkProfile[j, i] < -jerkSensitivity):
                #         # keyPoints[i].append(j-5)
                #         if(mainKeyPoints[i][-1] != j-1):
                #             mainKeyPoints[i].append(j-1)
                #         mainKeyPoints[i].append(j)
                #         outsideHysterisis[i] = True
                #         resetToZeroAddKeypoint[i] = False
                #         counterSinceLastEval[i] = 0

                #     if(resetToZeroAddKeypoint[i] == True):
                #         # print("reset to zero add keypoint")
                #         if(jerkProfile[j, i] < 0.000001 and jerkProfile[j, i] > -0.000001):
                #             mainKeyPoints[i].append(j)
                #             resetToZeroAddKeypoint[i] = False
                #             counterSinceLastEval[i] = 0

                #     if(counterSinceLastEval[i] >= maxN):
                #         mainKeyPoints[i].append(j)
                #         counterSinceLastEval[i] = 0

                #     counterSinceLastEval[i] = counterSinceLastEval[i] + 1

                if(counterSinceLastEval[i] >= minN):
                    # print("jerk profile: " + str(jerkProfile[j, i]))
                    if(jerkProfile[j, i] > jerkSensitivity or jerkProfile[j, i] < -jerkSensitivity):
                        mainKeyPoints[i].append(j)
                        counterSinceLastEval[i] = 0
                
                if(counterSinceLastEval[i] >= maxN):
                    mainKeyPoints[i].append(j)
                    counterSinceLastEval[i] = 0

                counterSinceLastEval[i] = counterSinceLastEval[i] + 1

        for i in range(self.dof):
            for j in range(len(mainKeyPoints[i])):
                if(j == 0):
                    keyPoints[i].append(mainKeyPoints[i][j])
                else:
                    # keyPoints[i].append(mainKeyPoints[i][j] - 15)
                    # keyPoints[i].append(mainKeyPoints[i][j] - 10)
                    # keyPoints[i].append(mainKeyPoints[i][j] - 5)
                    keyPoints[i].append(mainKeyPoints[i][j])
                    # keyPoints[i].append(mainKeyPoints[i][j] + 5)
                    # keyPoints[i].append(mainKeyPoints[i][j] + 10)
                    # keyPoints[i].append(mainKeyPoints[i][j] + 15)

        for i in range(self.dof):
            keyPoints[i].append(self.trajecLength - 1)
        
        return keyPoints 

    def keyPoints_adaptiveAccel(self, trajectoryStates, dynParameters):
        keyPoints = [[] for x in range(self.dof)]

        for i in range(self.dof):
            keyPoints[i].append(0)

        counterSinceLastEval = np.zeros((self.dof))

        minN = int(dynParameters[0])
        maxN = int(dynParameters[1])
        accelSensitivty = dynParameters[3]
        # temp
        # velGradCubeSensitivity = 0.0002

        accelProfile = self.calculateAccellerationOverTrajectory(self.states[0])

        for i in range(self.dof):
            for j in range(len(accelProfile)):

                if(counterSinceLastEval[i] >= minN):
                    if(accelProfile[j, i] > accelSensitivty or accelProfile[j, i] < -accelSensitivty):
                        keyPoints[i].append(j)
                        counterSinceLastEval[i] = 0
                
                if(counterSinceLastEval[i] >= maxN):
                    keyPoints[i].append(j)
                    counterSinceLastEval[i] = 0

                counterSinceLastEval[i] = counterSinceLastEval[i] + 1

        for i in range(self.dof):
            keyPoints[i].append(self.trajecLength - 1)
        
        return keyPoints 

    def keyPoints_iteratively(self, trajectoryStates, dynParameters):
        keyPoints = [[] for x in range(self.dof)]
        for i in range(self.dof):
            keyPoints[i].append(0)
            # keyPoints[i].append(self.trajecLength - 1)

        # startInterval = int(self.trajecLength / 2)
        # numMaxBins = int((self.trajecLength / startInterval))

        startIndex = 0
        endIndex = self.trajecLength - 1

        for i in range(self.dof):
            binComplete = False
            listofIndicesCheck = []
            indexTuple = (startIndex, endIndex)
            listofIndicesCheck.append(indexTuple)
            subListIndices = []
            subListWithMidpoints = []

            while(not binComplete):

                allChecksComplete = True
                for j in range(len(listofIndicesCheck)):

                    approximationGood, midIndex = self.oneCheck(trajectoryStates, listofIndicesCheck[j], i)

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
                        keyPoints[i].append(subListWithMidpoints[k])

                    subListWithMidpoints = []

                listofIndicesCheck = subListIndices.copy()
                subListIndices = []

        for i in range(self.dof):
            keyPoints[i].sort()
            keyPoints[i] = list(dict.fromkeys(keyPoints[i]))

        return keyPoints
        
    def oneCheck(self, A_matrices, indexTuple, dofNum):
        approximationGood = False

        startIndex = indexTuple[0]
        endIndex = indexTuple[1]

        midIndex = int((startIndex + endIndex) / 2)
        startVals = A_matrices[startIndex,:]
        endVals = A_matrices[endIndex,:]

        if((endIndex - startIndex) < self.dynParams[0]):
            return True, midIndex

        trueMidVals = A_matrices[midIndex,:]
        diff = endVals - startVals
        linInterpMidVals = startVals + (diff/2)

        meanSqDiff = self.meansqDiffBetweenAMatrices(trueMidVals, linInterpMidVals, dofNum)
        # sumsqDiff = self.sumsqDiffBetweenAMatrices(trueMidVals, linInterpMidVals)
        # print("meanSqDiff: " + str(meanSqDiff))

        # 0.05 for reaching and pushing
        #~0.001 for pendulum

        if(meanSqDiff < self.dynParams[4]):
            approximationGood = True

        return approximationGood, midIndex
    
    def sumsqDiffBetweenAMatrices(self, matrix1, matrix2):
        sumsqDiff = 0

        for i in range(len(matrix1)):
            sqDiff = (matrix1[i] - matrix2[i])**2
            if(sqDiff > 0.01):
                #ignore large values
                sqDiff = 0

            sumsqDiff = sumsqDiff + sqDiff

        return sumsqDiff

    def meansqDiffBetweenAMatrices(self, matrix1, matrix2, dofNum):
        sumsqDiff = 0
        counter = 0
        counterSmallVals = 0
        offsets = [0, self.dof]

        for i in range(2):
            for j in range(self.dof):
                index = offsets[i] + dofNum + (j * self.numStates)
                sqDiff = abs(matrix1[index] - matrix2[index])
                counter = counter + 1
                # if(sqDiff < 0.000001):
                #     sqDiff = 0
                #     counterSmallVals += 1
                # #ignore large values
                # # elif(sqDiff > 0.5):
                # #     sqDiff = 0
                # else:
                #     counter = counter + 1

                sumsqDiff = sumsqDiff + sqDiff

        # for i in range(len(matrix1)):
        #     sqDiff = (matrix1[i] - matrix2[i])**2
        #     if(sqDiff < 0.000001):
        #         sqDiff = 0
        #         counterSmallVals += 1
        #     #ignore large values
        #     elif(sqDiff > 0.5):
        #         sqDiff = 0
        #     else:
        #         counter = counter + 1

        #     sumsqDiff = sumsqDiff + sqDiff

        if(counter == 0):
            sumsqDiff = 0
        else:
            sumsqDiff = sumsqDiff / counter
            
        # print("counter: " + str(counter))
        # print("counter small vals: " + str(counterSmallVals))
        
        return sumsqDiff
    
    def generateLinInterpolation(self, A_matrices, reEvaluationIndicies):
        sizeofAMatrix = len(A_matrices[0])
        linInterpolationData = np.zeros((self.trajecLength, sizeofAMatrix))

        # print(reEvaluationIndicies[0])

        for i in range(self.dof):
            for j in range(len(reEvaluationIndicies[i]) - 1):
                    
                startIndex_pos = reEvaluationIndicies[i][j]
                startIndex_vel = reEvaluationIndicies[i][j]
                endIndex_pos = reEvaluationIndicies[i][j + 1]
                endIndex_vel = reEvaluationIndicies[i][j + 1]
                # print("startIndex_pos: " + str(startIndex_pos))
                # print("endIndex_pos: " + str(endIndex_pos))
                for k in range(self.dof):

                    startVals_pos = A_matrices[startIndex_pos, (k * self.numStates) + i]
                    startVals_vel = A_matrices[startIndex_vel, (k * self.numStates) + i + self.dof]

                    endVals_pos = A_matrices[endIndex_pos, (k * self.numStates) + i]
                    endVals_vel = A_matrices[endIndex_vel, (k * self.numStates) + i + self.dof]
                    # print("startVals: " + str(startVals_pos)

                    diff_pos = endVals_pos - startVals_pos
                    stepsBetween_pos = endIndex_pos - startIndex_pos
                    diff_vel = endVals_vel - startVals_vel
                    stepsBetween_vel = endIndex_vel - startIndex_vel

                    linInterpolationData[startIndex_pos, (k * self.numStates) + i] = startVals_pos
                    linInterpolationData[startIndex_pos, (k * self.numStates) + i + self.dof] = startVals_vel

                    for u in range(1, stepsBetween_pos):
                        linInterpolationData[startIndex_pos + u, (k * self.numStates) + i] = startVals_pos + (diff_pos * (u/stepsBetween_pos))
                        linInterpolationData[startIndex_vel + u, (k * self.numStates) + i + self.dof] = startVals_vel + (diff_vel * (u/stepsBetween_vel))

        
        linInterpolationData[len(linInterpolationData) - 1,:] = linInterpolationData[len(linInterpolationData) - 2,:]
        return linInterpolationData
    
    # def generateQuadInterpolation(self, A_matrices, reEvaluationIndicies):
    #     sizeofMatrix = len(A_matrices[0])
    #     quadInterpolationData = np.zeros((self.trajecLength, sizeofMatrix))

    #     for i in range(len(reEvaluationIndicies) - 2):
    #         startIndex = reEvaluationIndicies[i]
    #         midIndex = reEvaluationIndicies[i + 1]
    #         endIndex = reEvaluationIndicies[i + 2]

    #         for j in range(sizeofMatrix):

    #             points = np.zeros((3, 2))
    #             point1 = np.array([A_matrices[startIndex, j], startIndex])
    #             point2 = np.array([A_matrices[midIndex, j], midIndex])
    #             point3 = np.array([A_matrices[endIndex, j], endIndex])

    #             points[0] = point1
    #             points[1] = point2
    #             points[2] = point3

    #             #solve for coefficients a, b, c

    #             x_matrix = np.zeros((3, 3))
    #             y_matrix = np.zeros((1, 3))

    #             y_matrix[0, 0] = points[0, 0]
    #             y_matrix[0, 1] = points[1, 0]
    #             y_matrix[0, 2] = points[2, 0]

    #             for k in range(3):
    #                 x_matrix[0, k] = points[k, 1] * points[k, 1]
    #                 x_matrix[1, k] = points[k, 1]
    #                 x_matrix[2, k] = 1

    #             x_inv = np.linalg.inv(x_matrix)

    #             abc = y_matrix @ x_inv

    #             quadInterpolationData[startIndex,j] = A_matrices[startIndex,j]

    #             counter = 0
    #             for k in range(startIndex, endIndex):
    #                 a = abc[0, 0]
    #                 b = abc[0, 1]
    #                 c = abc[0, 2]

    #                 nextVal = (a * k * k) + (b * k) + c
    #                 quadInterpolationData[startIndex + counter, j] = nextVal
    #                 counter = counter + 1

    #     quadInterpolationData[len(quadInterpolationData) - 1,:] = quadInterpolationData[len(quadInterpolationData) - 2,:]

    #     return quadInterpolationData
    
    # def generateCubicInterpolation(self, A_matrices, reEvaluationIndicies):
    #     sizeofMatrix = len(A_matrices[0])
    #     quadInterpolationData = np.zeros((self.trajecLength, sizeofMatrix))

    #     for i in range(len(reEvaluationIndicies) - 3):
    #         startIndex = reEvaluationIndicies[i]
    #         midIndex1 = reEvaluationIndicies[i + 1]
    #         midIndex2 = reEvaluationIndicies[i + 2]
    #         endIndex = reEvaluationIndicies[i + 3]

    #         for j in range(sizeofMatrix):

    #             points = np.zeros((4, 2))
    #             point1 = np.array([A_matrices[startIndex, j], startIndex])
    #             point2 = np.array([A_matrices[midIndex1, j], midIndex1])
    #             point3 = np.array([A_matrices[midIndex2, j], midIndex2])
    #             point4 = np.array([A_matrices[endIndex, j], endIndex])

    #             points[0] = point1
    #             points[1] = point2
    #             points[2] = point3
    #             points[3] = point4

    #             #solve for coefficients a, b, c

    #             x_matrix = np.zeros((4, 4))
    #             y_matrix = np.zeros((1, 4))

    #             y_matrix[0, 0] = points[0, 0]
    #             y_matrix[0, 1] = points[1, 0]
    #             y_matrix[0, 2] = points[2, 0]
    #             y_matrix[0, 3] = points[3, 0]

    #             for k in range(4):
    #                 x_matrix[0, k] = points[k, 1] * points[k, 1] * points[k, 1]
    #                 x_matrix[1, k] = points[k, 1] * points[k, 1]
    #                 x_matrix[2, k] = points[k, 1]
    #                 x_matrix[3, k] = 1

    #             x_inv = np.linalg.inv(x_matrix)

    #             abcd = y_matrix @ x_inv

    #             quadInterpolationData[startIndex,j] = A_matrices[startIndex,j]

    #             counter = 0
    #             for k in range(startIndex, endIndex):
    #                 a = abcd[0, 0]
    #                 b = abcd[0, 1]
    #                 c = abcd[0, 2]
    #                 d = abcd[0, 3]

    #                 nextVal = (a * k * k * k) + (b * k * k) + (c * k) + d
    #                 quadInterpolationData[startIndex + counter, j] = nextVal
    #                 counter = counter + 1

    #     quadInterpolationData[len(quadInterpolationData) - 1,:] = quadInterpolationData[len(quadInterpolationData) - 2,:]

    #     return quadInterpolationData
    
    def butter_lowpass_filter(self, data, cutoff, nyq, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

def testFilter():
    pass

def filterArray(unfiltered):
    PI = 3.1415
    yn1 = unfiltered[0]
    xn1 = unfiltered[0]

    filtered = []

    for i in range(len(unfiltered)):

        xn = unfiltered[i]
        yn = 0.2283*yn1 + 0.3859*xn + 0.3859*xn1

        xn1 = xn
        yn1 = yn

        filtered.append(yn)

    plt.plot(filtered)
    plt.plot(unfiltered)
    plt.show()

def ICRATemp():
    myInterp = interpolator(1, 1)

    dynParams = [5, 200, 0.0005]

    trueTrajec, interpolatedTrajec, unfilteredTrajec, errors, reEvaluationIndices, iterativeKeyPoints = myInterp.interpolateTrajectory(0, dynParams)
            
    index = 13

    highlightedIndices = np.copy(unfilteredTrajec[reEvaluationIndices, ])
    highlightedIndicesIterative = np.copy(unfilteredTrajec[iterativeKeyPoints, ])
    numEvals = len(reEvaluationIndices)

    yellow = '#EEF30D'
    black = '#000000'
    darkBlue = '#103755'

    plt.figure(figsize=(5,3))
    plt.plot(trueTrajec[:,index], color = darkBlue, label='Ground truth', linewidth=3)
    # plt.plot(interpolatedTrajec[0,:,index], color = yellow, label='interpolated')
    # plt.scatter(reEvaluationIndices, highlightedIndices[:, index], s=10, color = yellow, zorder=10)
    plt.plot(interpolatedTrajec[1,:,index], color = yellow, label='interpolated', linewidth=1, alpha = 1)
    plt.scatter(iterativeKeyPoints, highlightedIndicesIterative[:, index], s=10, color = yellow, alpha = 1, zorder=10)
    # plt.legend()
    # turn off y axis
    plt.gca().axes.get_yaxis().set_visible(False)
    # save as svg
    plt.savefig('test.svg', format='svg', dpi=1200)
    plt.show()

    index += 1

if __name__ == "__main__":
    ICRATemp()

    # myInterp = interpolator(0, 2)

    # # Filter requirements.
    # T = 5.0         # Sample Period
    # fs = 100      # sample rate, Hz
    # cutoff = 1      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    # nyq = 0.5 * fs  # Nyquist Frequency
    # order = 2       # sin wave can be approx represented as quadratic
    # n = int(T * fs) # total number of samples

    # index = 1
    # for i in range(10):
    #     filterArray(myInterp.testTrajectories[0][:,index + i])

    # filterArray(myInterp.testTrajectories[0][:,index])

    # filteredData = myInterp.butter_lowpass_filter(myInterp.testTrajectories[0][:,index], cutoff, fs, order)

    # plt.plot(myInterp.testTrajectories[0][:,index])
    # plt.plot(filteredData)
    # plt.show()




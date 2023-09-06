import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import math
import csv

from plotUtils import *


OPTIMISERS_USED = 0
DATA_FIELDS = 6 # final cost, avg Hz, avg percent derivs, avg time get derivs, avg time bp, avg time fp

def plotMPCData(taskName, taskNumber):
    global OPTIMISERS_USED
    # load the data

    dataNumber = str(taskNumber)
    file_name = 'data/resultsdatampc/' + dataNumber + "/" + taskName + '_testingData.csv'
    data, labels = getDataandLabels(file_name)

    numTrajecs = len(data) - 2

    finalCosts = np.zeros((numTrajecs, OPTIMISERS_USED))
    controlFrequencies = np.zeros((numTrajecs, OPTIMISERS_USED))
    avgPercentDerivs = np.zeros((numTrajecs, OPTIMISERS_USED))
    avgTimeDerivs = np.zeros((numTrajecs, OPTIMISERS_USED))
    avgTimeBP = np.zeros((numTrajecs, OPTIMISERS_USED))
    avgTimeFP = np.zeros((numTrajecs, OPTIMISERS_USED))

    for i in range(numTrajecs):
        for j in range(OPTIMISERS_USED):

            finalCosts[i, j] = data[i + 2, (j * DATA_FIELDS)]
            controlFrequencies[i, j] = data[i + 2, (j * DATA_FIELDS) + 1]
            avgTimeDerivs[i, j] = data[i + 2, (j * DATA_FIELDS) + 2]
            avgTimeBP[i, j] = data[i + 2, (j * DATA_FIELDS) + 3]
            avgTimeFP[i, j] = data[i + 2, (j * DATA_FIELDS) + 4]
            avgPercentDerivs[i, j] = data[i + 2, (j * DATA_FIELDS) + 5]

    # calculate staistical measures
    meanFinalCosts = np.zeros(OPTIMISERS_USED)
    stdFinalCosts = np.zeros(OPTIMISERS_USED)
    meanControlFrequencies = np.zeros(OPTIMISERS_USED)
    stdControlFrequencies = np.zeros(OPTIMISERS_USED)

    for i in range(OPTIMISERS_USED):
        meanFinalCosts[i] = np.mean(finalCosts[:, i])
        stdFinalCosts[i] = np.std(finalCosts[:, i])
        meanControlFrequencies[i] = np.mean(controlFrequencies[:, i])
        stdControlFrequencies[i] = np.std(controlFrequencies[:, i])

    for i in range(OPTIMISERS_USED):
        print("Optimiser: " + labels[i])
        print("Mean final cost: " + str(meanFinalCosts[i]))
        print("Std final cost: " + str(stdFinalCosts[i]))
        print("Mean control frequency: " + str(meanControlFrequencies[i]))
        print("Std control frequency: " + str(stdControlFrequencies[i]))

    fig, axes = plt.subplots(2, 1, figsize = (18,8))
    boxPlotTitle = "Final cost against interpolation methods " + taskName
    yAxisLabel = "Final cost"
    orange = "#edb83b"
    bp1 = box_plot(finalCosts, orange, yAxisLabel, axes[0], labels)

    boxPlotTitle = "Control frequency against interpolation methods " + taskName
    yAxisLabel = "Control frequency (Hz)"
    orange = "#edb83b"
    bp2 = box_plot(controlFrequencies, orange, yAxisLabel, axes[1], labels, baseline_yLine = False)

    fig.suptitle(taskName + " - optimisation information", fontsize=16)
    plt.show()

def plotMPCHorizonData(taskName, methods, taskNumber):
    global OPTIMISERS_USED
    # load the data

    sizeData = True
    finalCostsMethods = np.zeros((len(methods), OPTIMISERS_USED))
    avgHzMethods = np.zeros((len(methods), OPTIMISERS_USED))

    avgTimeDerivsBaseline = np.zeros((OPTIMISERS_USED))
    avgTimeBPBaseline = np.zeros((OPTIMISERS_USED))
    avgTimeFPBaseline = np.zeros((OPTIMISERS_USED))

    for i in range(len(methods)):

        dataNumber = str(taskNumber)
        file_name = 'data/resultsdatampc/' + dataNumber + "/horizonData/" + taskName +  "_" + methods[i] + '.csv'
        data, labels = getDataandLabels(file_name)

        if(sizeData):
            sizeData = False
            finalCostsMethods = np.zeros((len(methods), OPTIMISERS_USED))
            avgHzMethods = np.zeros((len(methods), OPTIMISERS_USED))
            avgTimeDerivsBaseline = np.zeros((OPTIMISERS_USED))
            avgTimeBPBaseline = np.zeros((OPTIMISERS_USED))
            avgTimeFPBaseline = np.zeros((OPTIMISERS_USED))

        numTrajecs = len(data) - 2

        finalCosts = np.zeros((numTrajecs, OPTIMISERS_USED))
        controlFrequencies = np.zeros((numTrajecs, OPTIMISERS_USED))
        avgPercentDerivs = np.zeros((numTrajecs, OPTIMISERS_USED))
        avgTimeDerivs = np.zeros((numTrajecs, OPTIMISERS_USED))
        avgTimeBP = np.zeros((numTrajecs, OPTIMISERS_USED))
        avgTimeFP = np.zeros((numTrajecs, OPTIMISERS_USED))

        for k in range(numTrajecs):
            for j in range(OPTIMISERS_USED):
                finalCosts[k, j] = data[k + 2, (j * DATA_FIELDS)]
                controlFrequencies[k, j] = data[k + 2, (j * DATA_FIELDS) + 1]
                avgTimeDerivs[k, j] = data[k + 2, (j * DATA_FIELDS) + 2]
                avgTimeBP[k, j] = data[k + 2, (j * DATA_FIELDS) + 3]
                avgTimeFP[k, j] = data[k + 2, (j * DATA_FIELDS) + 4]
                avgPercentDerivs[k, j] = data[k + 2, (j * DATA_FIELDS) + 5]

        # calculate staistical measures
        meanFinalCosts = np.zeros(OPTIMISERS_USED)
        stdFinalCosts = np.zeros(OPTIMISERS_USED)
        meanControlFrequencies = np.zeros(OPTIMISERS_USED)
        stdControlFrequencies = np.zeros(OPTIMISERS_USED)

        for j in range(OPTIMISERS_USED):
            meanFinalCosts[j] = np.mean(finalCosts[:, j])
            stdFinalCosts[j] = np.std(finalCosts[:, j])
            meanControlFrequencies[j] = np.mean(controlFrequencies[:, j])
            stdControlFrequencies[j] = np.std(controlFrequencies[:, j])

        finalCostsMethods[i, :] = meanFinalCosts
        avgHzMethods[i, :] = meanControlFrequencies

        if methods[i] == "baseline":
            for j in range(OPTIMISERS_USED):
                avgTimeDerivsBaseline[j] = np.mean(avgTimeDerivs[:, j])
                avgTimeBPBaseline[j] = np.mean(avgTimeBP[:, j])
                avgTimeFPBaseline[j] = np.mean(avgTimeFP[:, j])

        print("-------------------" + methods[i] + "-------------------")
        # for j in range(OPTIMISERS_USED):
        #     print("Optimiser: " + labels[j])
        #     print("avg time getting derivs: " + str(np.mean(avgTimeDerivs[:, j])))
        #     print("avg time backprop: " + str(np.mean(avgTimeBP[:, j])))
        #     print("avg time forward prop: " + str(np.mean(avgTimeFP[:, j])))


            # print("Optimiser: " + labels[i])
            # print("Mean final cost: " + str(meanFinalCosts[j]))
            # print("Std final cost: " + str(stdFinalCosts[j]))
            # print("Mean control frequency: " + str(meanControlFrequencies[j]))
            # print("Std control frequency: " + str(stdControlFrequencies[j]))

    colors = DIVERGENT_COLORS

    fig, axes = plt.subplots(2, 1, figsize = (6, 9))
    plotTitle = "Final cost against horizon length " + taskName
    yAxisLabel = "Final cost"
    linegraph1 = linegraph(finalCostsMethods, axes[0], colors, yAxisLabel, "horizon", labels, methods, logyAxis = False)

    boxPlotTitle = "Control frequency against interpolation methods " + taskName
    yAxisLabel = "Control frequency (Hz)"
    linegraph2 = linegraph(avgHzMethods, axes[1], colors, yAxisLabel, "horizon", labels, methods)

    fig.suptitle(taskName + " - MPC iteration times (baseline)", fontsize=16)
    plt.show()


    # Create stacked bar plot time getting derivs, bp and fp
    fig, axes = plt.subplots(1, 1, figsize = (8, 6))
    plotTitle = "Time taken to get derivatives, backprop and forward prop against horizon length " + taskName
    yAxisLabel = "Time (ms)"
    stackedData = np.zeros((3, OPTIMISERS_USED))

    print("avg time getting derivs: " + str(avgTimeDerivsBaseline))
    print("avg time backprop: " + str(avgTimeBPBaseline))
    print("avg time forward prop: " + str(avgTimeFPBaseline))

    stackedData[0, :] = avgTimeDerivsBaseline
    stackedData[1, :] = avgTimeBPBaseline
    stackedData[2, :] = avgTimeFPBaseline

    stackLabels = ["Derivatives", "Backprop", "Forward prop"]

    stackedBarGraph(stackedData, axes, colors, yAxisLabel, "horizon", labels, stackLabels)

    fig.suptitle(taskName + " - MPC horizon data", fontsize=16)
    plt.show()

def getDataandLabels(filename):
    global OPTIMISERS_USED
    data = np.array([genfromtxt(filename, delimiter = ',')])

    file = open(filename, "r")
    headers = list(csv.reader(file, delimiter=","))
    file.close()

    # Get the number of optimiser automatically depending on number of columns in data
    OPTIMISERS_USED = int(len(headers[0])/DATA_FIELDS)

    lenHeaders = len(headers[0])
    labels = []
    for i in range(int(lenHeaders/DATA_FIELDS)):
        labels.append(headers[0][i*DATA_FIELDS])

    data = data[0]

    return data, labels
    
if __name__ == "__main__":
    methods = ["baseline", "adaptive_jerk", "SI10", "SI5", "magvel_change"]
    plotMPCHorizonData("walker",  methods, 4)
    # plotMPCData("walker", 3)

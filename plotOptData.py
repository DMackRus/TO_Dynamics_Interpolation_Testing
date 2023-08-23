import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import itertools
import pandas as pd
import glob
from scipy import signal
import math
import csv

# def plotResults():
#     # Load data into numpy array
#     taskName = "panda_pushing_clutter"
#     # taskName = "panda_pushing"
#     # taskName = "doublePendulum"

#     data = np.array([genfromtxt('data/resultsData/no_filtering/' + taskName + '_testingData.csv', delimiter = ',')])

#     file = open('data/resultsData/no_filtering/' + taskName + '_testingData.csv', "r")
#     headers = list(csv.reader(file, delimiter=","))
#     file.close()

#     print(headers[0])
#     lenHeaders = len(headers[0])
#     labels = []
#     for i in range(int(lenHeaders/4) - 1):
#         labels.append(headers[0][i*4])


#     print(labels)

#     data = data[0]

#     numTrajecs = len(data) - 2
#     print("num trajecs: " + str(numTrajecs))

#     optTimes = np.zeros((numTrajecs, 3))
#     costReductions = np.zeros((numTrajecs, 3))
#     avgNumDerivs = np.zeros((numTrajecs, 3))
#     avgTimeGettingDerivs = np.zeros((numTrajecs, 3))

#     for i in range(numTrajecs):
#         for j in range(3):

#             optTimes[i, j] = data[i + 2, (j * 4)]
#             costReductions[i, j] = data[i + 2, (j * 4) + 1]
#             avgNumDerivs[i, j] = data[i + 2, (j * 4) + 2]
#             avgTimeGettingDerivs[i, j] = data[i + 2, (j * 4) + 3]

    
#     # fig, axes = plt.subplots(1, 1, figsize = (7,5))
#     # boxPlotTitle = "Optimisation time against interpolation methods " + "panda_pushing_clutter"
#     # yAxisLabel = "Total optimisation time (s)"
#     # orange = "#edb83b"
#     # bp1 = box_plot(optTimes, orange, yAxisLabel, axes, labels)
#     # plt.savefig('plot_times.svg', format='svg', dpi=1200, transparent = True)
#     # plt.show()

#     fig, axes = plt.subplots(1, 1, figsize = (7,5))
#     boxPlotTitle = "Optimisation time against interpolation methods " + "panda_pushing_clutter"
#     yAxisLabel = "Cost Reduction"
#     orange = "#edb83b"
#     bp1 = box_plot(costReductions, orange, yAxisLabel, axes, labels)
#     plt.savefig('cost_reductions.svg', format='svg', dpi=1200, transparent = True)
#     plt.show()

def plotOneTask(taskName, testNumber):

    dataNumber = str(testNumber)

    data = np.array([genfromtxt('data/resultsData/' + dataNumber + "/" + taskName + '_testingData.csv', delimiter = ',')])

    file = open('data/resultsData/'  + dataNumber + "/" + taskName + '_testingData.csv', "r")
    headers = list(csv.reader(file, delimiter=","))
    file.close()

    DATA_FIELDS = 5 # opt time, cost reduction, percentage derivs, time getting derivs, numIterations
    OPTIMISERS_USED = 5

    lenHeaders = len(headers[0])
    labels = []
    for i in range(int(lenHeaders/DATA_FIELDS)):
        labels.append(headers[0][i*DATA_FIELDS])


    print(labels)

    data = data[0]

    numTrajecs = len(data) - 2
    print("num trajecs: " + str(numTrajecs))

    optTimes = np.zeros((numTrajecs, OPTIMISERS_USED))
    costReductions = np.zeros((numTrajecs, OPTIMISERS_USED))
    scaledCosts = np.zeros((numTrajecs, OPTIMISERS_USED))
    avgPercentageDerivs = np.zeros((numTrajecs, OPTIMISERS_USED))
    avgTimeGettingDerivs = np.zeros((numTrajecs, OPTIMISERS_USED))
    numIterations = np.zeros((numTrajecs, OPTIMISERS_USED))


    for i in range(numTrajecs):
        for j in range(OPTIMISERS_USED):

            optTimes[i, j] = data[i + 2, (j * DATA_FIELDS)]
            costReductions[i, j] = data[i + 2, (j * DATA_FIELDS) + 1]
            avgPercentageDerivs[i, j] = data[i + 2, (j * DATA_FIELDS) + 2]
            avgTimeGettingDerivs[i, j] = data[i + 2, (j * DATA_FIELDS) + 3]
            numIterations[i, j] = data[i + 2, (j * DATA_FIELDS) + 4]

    # Format the cost reductions with reference to the baseline cost
    for i in range(numTrajecs):
        for j in range(OPTIMISERS_USED):
            if j == 1:
                print("--------------------------------------------------------")
                print("index: " + str(i))
                print("cost reduction of index: " + str(costReductions[i, j]))
                print("cost reduction of baseline: " + str(costReductions[i, 0]))
            scaledCosts[i, j] = -((costReductions[i, j] - costReductions[i, 0]) / costReductions[i, 0]) * 100

            if j == 1:

                print("cost reduction of index after scaling: " + str(scaledCosts[i, j]))


    fig, axes = plt.subplots(3, 1, figsize = (18,8))
    boxPlotTitle = "Optimisation time against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Total optimisation time (ms)"
    orange = "#edb83b"
    bp1 = box_plot(optTimes, orange, yAxisLabel, axes[0], labels)

    boxPlotTitle = "Cost Reduction against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Cost Reduction"
    orange = "#edb83b"
    bp2 = box_plot(costReductions, orange, yAxisLabel, axes[1], labels, baseline_yLine = False)

    boxPlotTitle = "Num iterations against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Num Iterations"
    orange = "#edb83b"
    bp3 = box_plot(numIterations, orange, yAxisLabel, axes[2], labels)



    fig.suptitle(taskName + " - optimisation information", fontsize=16)
    plt.show()


    fig, axes = plt.subplots(2, 1, figsize = (18,8))
    boxPlotTitle = "Average percentage calculated derivatives against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Average percentage calculate derivatives"
    orange = "#edb83b"
    bp3 = box_plot(avgPercentageDerivs, orange, yAxisLabel, axes[0], labels, False)

    boxPlotTitle = "average time getting derivatives against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Average time getting derivatives (ms)"
    orange = "#edb83b"
    bp4 = box_plot(avgTimeGettingDerivs, orange, yAxisLabel, axes[1], labels)
    fig.suptitle(taskName + " - derivative information", fontsize=16)
    
    # plt.savefig('data/resultsData/'  + dataNumber + "/" + taskName + '_boxplots.svg', format='svg', dpi=1200)
    # plt.savefig('data/resultsData/' + dataNumber + "/" + taskName + '_boxplots.png')
    plt.show()

def plotOneResultMPC(taskName):
    dataNumber = "2"

    data = np.array([genfromtxt('data/resultsdatampc/' + dataNumber + "/" + taskName + '_testingData.csv', delimiter = ',')])

    file = open('data/resultsdatampc/'  + dataNumber + "/" + taskName + '_testingData.csv', "r")
    headers = list(csv.reader(file, delimiter=","))
    file.close()

    DATA_FIELDS = 8 # sucess, finaldistance, execution time, optimisation time,average time getting derivs, avg time bp, avg time fp, avgpercent derivs, 
    OPTIMISERS_USED = 6

    lenHeaders = len(headers[0])
    labels = []
    for i in range(int(lenHeaders/DATA_FIELDS)):
        labels.append(headers[0][i*DATA_FIELDS])

    data = data[0]

    numTrajecs = len(data) - 2
    print("num trajecs: " + str(numTrajecs))

    sucesses = np.zeros((OPTIMISERS_USED))
    finalDistances = np.zeros((numTrajecs, OPTIMISERS_USED))
    executionTimes = np.zeros((numTrajecs, OPTIMISERS_USED))
    optimisationTimes = np.zeros((numTrajecs, OPTIMISERS_USED))
    avgPercentDerivs = np.zeros((numTrajecs, OPTIMISERS_USED))
    avgTimeGettingDerivs = np.zeros((numTrajecs, OPTIMISERS_USED))

    for i in range(numTrajecs):
        for j in range(OPTIMISERS_USED):

            sucesses[j] += data[i + 2, (j * DATA_FIELDS)]
            finalDistances[i, j] = data[i + 2, (j * DATA_FIELDS) + 1]
            executionTimes[i, j] = data[i + 2, (j * DATA_FIELDS) + 2]
            optimisationTimes[i, j] = data[i + 2, (j * DATA_FIELDS) + 3]
            avgTimeGettingDerivs[i, j] = data[i + 2, (j * DATA_FIELDS) + 4]
            avgPercentDerivs[i, j] = data[i + 2, (j * DATA_FIELDS) + 7]
            

    fig, axes = plt.subplots(4, 1, figsize = (18,8))

    #First plot should be a bar graph
    barPlotTitle = "Sucesses - " + taskName
    yAxisLabel = "Number of sucesses"
    orange = "#edb83b"
    # linSpace = np.linspace(0, OPTIMISERS_USED - 1, OPTIMISERS_USED)
    bp0 = bar_plot(sucesses, orange, yAxisLabel, axes[0], labels)

    boxPlotTitle = "Final distance: " + taskName
    yAxisLabel = "Final distance to goal (m)"
    orange = "#edb83b"
    bp1 = box_plot(finalDistances, orange, yAxisLabel, axes[1], labels)

    boxPlotTitle = "Execution times - " + taskName
    yAxisLabel = "Execution time (s)"
    orange = "#edb83b"
    bp2 = box_plot(executionTimes, orange, yAxisLabel, axes[2], labels)

    boxPlotTitle = "Optimisation time - " + taskName
    yAxisLabel = "Optimisation time (s)"
    orange = "#edb83b"
    bp3 = box_plot(optimisationTimes, orange, yAxisLabel, axes[3], labels)


    fig.suptitle(taskName + " - optimisation information (MPC)", fontsize=16)
    # save the figure
    plt.savefig('data/resultsdatampc/' + dataNumber + "/" + taskName + '_boxplots_timings.png')
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize = (18,8))

    boxPlotTitle = "avg percentage derivatives - " + taskName
    yAxisLabel = "avg percentage derivatives"
    orange = "#edb83b"
    bp2 = box_plot(avgPercentDerivs, orange, yAxisLabel, axes[0], labels, True)

    boxPlotTitle = "avg time getting derivatives - " + taskName
    yAxisLabel = "avg time getting derivatives (ms)"
    orange = "#edb83b"
    bp3 = box_plot(avgTimeGettingDerivs, orange, yAxisLabel, axes[1], labels, True)


    fig.suptitle(taskName + " - optimisation information (MPC)", fontsize=16)
    # save the figure
    plt.savefig('data/resultsdatampc/' + dataNumber + "/" + taskName + '_boxplots_derivs.png')
    plt.show()


def plotResults():
    # Load data into numpy array

    # taskNames = ["panda_pushing", "panda_pushing_clutter", "panda_pushing_heavy_clutter"]
    # taskNames = ["panda_box_flick", "panda_box_flick_low_clutter", "panda_box_flick_heavy_clutter"]
    # taskNames = ["panda_pushing"]
    taskNames = ["doublePendulum"]
    testNumber = 9

    for i in range(len(taskNames)):
        plotOneTask(taskNames[i], testNumber)
        # plotOneResultMPC(taskNames[i])
    
    # taskName = "doublePendulum"

def bar_plot(data, fill_color, yAxisTitle, ax, labels):
    normalPosterColour = "#103755"
    highlightPosterColor = "#EEF30D"

    bp = ax.bar(labels, data, color=normalPosterColour)
    ax.set_ylabel(yAxisTitle)
    ax.set_xlabel("Optimiser")
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 100])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(axis='y', which='major', pad=15)
    #horizontal line at 0
    ax.axhline(y=data[0], color= normalPosterColour, linewidth=1.5, alpha=0.5)
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgr

def box_plot(data, fill_color, yAxisTitle, ax, labels, logyAxis = False, baseline_yLine = False):
    normalPosterColour = "#103755"
    highlightPosterColor = "#EEF30D"


    bp = ax.boxplot(data, patch_artist=True, meanprops={"marker":"s","markerfacecolor":highlightPosterColor, "markeredgecolor":highlightPosterColor}, showmeans=True, showfliers=False)
    if logyAxis:
        ax.set_yscale('log')
    black = "#1c1b1a"

    
    
    for element in ['medians']:
        plt.setp(bp[element], color=black)

    for element in ['means']:
        plt.setp(bp[element], color=highlightPosterColor)
        # element.set(color="#a808c4")

    # for bpMean in bp['means']:
    #     bpMean.set(color="#a808c4")

    # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    #     plt.setp(bp[element], color=black)


    dynamicColor = "#ff8400"
    baselineColor = "#0057c9"
    setIntervalColour = "#9d00c9"
    backgroundColour = "#d6d6d6"

    
    index = 0
    for patch in bp['boxes']:
        patch.set(facecolor=normalPosterColour)
        # if(index == 0):
        #     patch.set(facecolor=normalPosterColour)
        # elif(index == numxAxis - 1):

        #     patch.set(facecolor=normalPosterColour)
        # else:
        #      patch.set(facecolor=normalPosterColour)

        # index = index + 1   

    labelSize = 11

    #ax = plt.gca()
    #ax.set_facecolor(backgroundColour)

    # Make this label bigger
    # ax.set(ylabel= yAxisTitle)
    ax.set_ylabel(yAxisTitle, fontsize=labelSize)
    # ax.set_title(yAxisTitle)

    if(baseline_yLine):
        ax.axhline(y=0, color=baselineColor, linewidth=1.5, alpha=0.5)

    xticks = []
    for i in range(len(labels)):
        xticks.append(i + 1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=11)
        
    return bp    


if __name__ == "__main__":
    plotResults()
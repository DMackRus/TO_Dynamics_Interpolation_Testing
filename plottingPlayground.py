import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import itertools
import pandas as pd
import glob
from scipy import signal
import math
import csv


def main():
    # data_list = np.array([[0, 0, 0]])
    # data_2 = np.array([[1, 1, 1]])

    # test = np.concatenate((data_list, data_2), axis=0)
    # print(test)
    unfiltered_data = returnUnfilteredLists()
    # tests = ["1Hz","50Hz","100Hz","200Hz", "500Hz"]
    tests = ["0.05", "0.1", "0.15", "0.2", "0.25"]

    for i in range(len(tests)):
        fileExtension = tests[i]

        filtered_data = returnFilteredLists(fileExtension)
        avgCostDiff, avgFiltered, avgUnfiltered = averageDiffPerIteration(unfiltered_data, filtered_data)

        plt.plot(avgCostDiff, label = 'cost diff')
        plt.plot(avgUnfiltered, label = 'unfiltered')
        plt.plot(avgFiltered, label = 'filtered')
        plt.legend()
        plt.title("Average cost diff - " + tests[i])
        plt.show()

    # average the differnece between the two numbers

    for i in range(50):

        plt.plot(filtered_data[i], label = 'filtered')
        plt.plot(unfiltered_data[i], label = 'unfiltered')
        plt.title("plot " + str(i))
        plt.legend()
        plt.show()

def averageDiffPerIteration(unfiltered, filtered):
    avgCostDiff = []
    avgFiltered = []
    avgUnfiltered = []

    for i in range(len(unfiltered[0])):
        sumDiff = 0
        sumFiltered = 0
        sumUnfiltered = 0
        for j in range(50):
            
            # negative diff means filtered is outperforming filtered
            diff = filtered[j][i] - unfiltered[j][i]
            sumDiff = sumDiff + diff
            sumFiltered += filtered[j][i]
            sumUnfiltered += unfiltered[j][i]
            if(i == 2):
                print("diff "  + str(j) + " is: " + str(diff))
                
        print("sumdiff is: " + str(sumDiff))

        avgCostDiff.append(sumDiff / len(unfiltered))
        avgFiltered.append(sumFiltered / len(unfiltered))
        avgUnfiltered.append(sumUnfiltered / len(unfiltered))


    return avgCostDiff, avgFiltered, avgUnfiltered

def returnFilteredLists(fileExtension):
    data_list = []
    maxLength = 0

    # Load the 50 csvs
    for i in range(50):
        temp = np.array([genfromtxt('data/filtering_' + fileExtension + '/' + str(i) + '.csv', delimiter=',')])
        temp = np.delete(temp, -1)
        
        tempList = list(temp)
        if(len(tempList) > maxLength):
            maxLength = len(tempList)

        data_list.append(tempList)

    maxLength = 9
    print("max length is: " + str(maxLength))

    for i in range(50):
        maxVal = data_list[i][0]
        for j in range(len(data_list[i])):
            data_list[i][j] = data_list[i][j] / maxVal

    for i in range(50):
        lenofCurrent = len(data_list[i]) - 1
        diff = maxLength - lenofCurrent
        for j in range(diff):
            data_list[i].append(data_list[i][lenofCurrent])

    return data_list

def returnUnfilteredLists():
    data_list = []
    maxLength = 0

    # Load the 50 csvs
    for i in range(50):
        temp = np.array([genfromtxt('data/no_filtering/' + str(i) + '.csv', delimiter=',')])
        temp = np.delete(temp, -1)
        
        tempList = list(temp)
        if(len(tempList) > maxLength):
            maxLength = len(tempList)

        data_list.append(tempList)

    maxLength = 9
    print("max length is: " + str(maxLength))

    for i in range(50):
        maxVal = data_list[i][0]
        for j in range(len(data_list[i])):
            data_list[i][j] = data_list[i][j] / maxVal

    for i in range(50):
        lenofCurrent = len(data_list[i]) - 1
        diff = maxLength - lenofCurrent
        for j in range(diff):
            data_list[i].append(data_list[i][lenofCurrent])

    return data_list

def makeFilter():
    w0 = 2*np.pi*500; # pole frequency (rad/s)
    num = w0        # transfer function numerator coefficients
    den = [1,w0]    # transfer function denominator coefficients
    lowPass = signal.TransferFunction(num,den) # Transfer function

    samplingFreq = 1 / 0.004

    dt = 1.0/samplingFreq
    discreteLowPass = lowPass.to_discrete(dt,method='gbt',alpha=0.5)
    print(discreteLowPass)

    b = discreteLowPass.num;
    a = -discreteLowPass.den;
    print("Filter coefficients b_i: " + str(b))
    print("Filter coefficients a_i: " + str(a[1:]))

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

def plotOneTask(taskName):

    dataNumber = "4"

    data = np.array([genfromtxt('data/resultsData/' + dataNumber + "/" + taskName + '_testingData.csv', delimiter = ',')])

    file = open('data/resultsData/'  + dataNumber + "/" + taskName + '_testingData.csv', "r")
    headers = list(csv.reader(file, delimiter=","))
    file.close()

    DATA_FIELDS = 5 # opt time, cost reduction, percentage derivs, time getting derivs, numIterations
    OPTIMISERS_USED = 11

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


    fig, axes = plt.subplots(3, 1, figsize = (18,8))
    boxPlotTitle = "Optimisation time against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Total optimisation time (s)"
    orange = "#edb83b"
    bp1 = box_plot(optTimes, orange, yAxisLabel, axes[0], labels)

    boxPlotTitle = "Cost Reduction against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Cost Reduction"
    orange = "#edb83b"
    bp2 = box_plot(costReductions, orange, yAxisLabel, axes[1], labels)

    boxPlotTitle = "Num iterations against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Cost Reduction"
    orange = "#edb83b"
    bp3 = box_plot(numIterations, orange, yAxisLabel, axes[2], labels)



    fig.suptitle(taskName + " - optimisation information", fontsize=16)
    plt.show()


    fig, axes = plt.subplots(2, 1, figsize = (18,8))
    boxPlotTitle = "Average percentage calculated derivatives against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Average percentage calculate derivatives"
    orange = "#edb83b"
    bp3 = box_plot(avgPercentageDerivs, orange, yAxisLabel, axes[0], labels, True)

    boxPlotTitle = "average time getting derivatives against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Average time getting derivatives (s)"
    orange = "#edb83b"
    bp4 = box_plot(avgTimeGettingDerivs, orange, yAxisLabel, axes[1], labels)
    fig.suptitle(taskName + " - derivative information", fontsize=16)
    
    # plt.savefig('data/resultsData/'  + dataNumber + "/" + taskName + '_boxplots.svg', format='svg', dpi=1200)
    # plt.savefig('data/resultsData/' + dataNumber + "/" + taskName + '_boxplots.png')
    plt.show()


def plotResults():
    # Load data into numpy array

    taskNames = ["panda_pushing", "panda_pushing_clutter", "panda_pushing_heavy_clutter"]

    for i in range(len(taskNames)):
        plotOneTask(taskNames[i])
    
    # taskName = "doublePendulum"

    

def box_plot(data, fill_color, yAxisTitle, ax, labels, logyAxis = False):
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

    labelSize = 13

    #ax = plt.gca()
    #ax.set_facecolor(backgroundColour)

    # Make this label bigger
    # ax.set(ylabel= yAxisTitle)
    ax.set_ylabel(yAxisTitle, fontsize=labelSize)
    # ax.set_title(yAxisTitle)

    xticks = []
    for i in range(len(labels)):
        xticks.append(i + 1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=11)
        
    return bp

def K_matrices():
    #Load some data into numpy array
    K_normal = np.array([genfromtxt('K_data/K_normal.csv', delimiter = ',')])
    K_parallel = np.array([genfromtxt('K_data/K_parallel.csv', delimiter = ',')])

    K_normal = K_normal[0]
    K_parallel = K_parallel[0]

    index = 20
    for i in range(100):

        index += 1
        plt.plot(K_normal[:, index], label = "Normal")
        plt.plot(K_parallel[:, index], label = "Parallel")
        plt.legend()
        plt.show()

def costMatrices():
    path = "cost_data/"

    l_x = np.array([genfromtxt(path + 'l_x.csv', delimiter = ',')])
    l_x = l_x[0]

    l_x_fd = np.array([genfromtxt(path + 'l_x_fd.csv', delimiter = ',')])
    l_x_fd = l_x_fd[0]

    l_xx = np.array([genfromtxt(path + 'l_xx.csv', delimiter = ',')])
    l_xx = l_xx[0]

    # index = 8

    # print(len(l_x[0]))

    # for i in range(len(l_x[0]) - 1):
    #     plt.plot(l_x[:, index], label = "l_x")
    #     plt.plot(l_x_fd[:, index], label = "l_f_fd")
    #     plt.legend()
    #     plt.show()
    #     index += 1

    index = (9 * 2 * 9 * 2) - 1

    print(len(l_xx[index]))

    for i in range(len(l_x[0]) - 1):
        plt.plot(l_xx[:, index], label = "l_xx")
        plt.legend()
        plt.show()
        index += 1


    


if __name__ == "__main__":
    plotResults()
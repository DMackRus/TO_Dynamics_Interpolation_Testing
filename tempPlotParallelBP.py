import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import itertools
import pandas as pd
import glob
from scipy import signal
import math
import csv
import matplotlib.patches as mpatches

def plotOneTask(taskName, titles):

    normalPosterColour = "#103755"
    highlightPosterColor = "#EEF30D"
    alternatePosterColor = "#18b842"

    data_ser = "2"
    data_para = "3"

    serialFileName = 'data/resultsData/' + data_ser + "/" + taskName + '_testingData.csv'
    print(serialFileName)

    ser_data = np.array([genfromtxt(serialFileName, delimiter = ',')])
    para_data = np.array([genfromtxt('data/resultsData/' + data_para + "/" + taskName + '_testingData.csv', delimiter = ',')])


    file_ser = open('data/resultsData/'  + data_ser + "/" + taskName + '_testingData.csv', "r")
    headers = list(csv.reader(file_ser, delimiter=","))
    file_ser.close()

    file_para = open('data/resultsData/'  + data_para + "/" + taskName + '_testingData.csv', "r")
    headers = list(csv.reader(file_para, delimiter=","))
    file_para.close()

    print(headers[0])
    lenHeaders = len(headers[0])
    labelsSer = []
    labelsPara = []
    labelsTotal = []
    for i in range(int(lenHeaders/4)):
        labelsSer.append(headers[0][i*4])
        labelsPara.append(headers[0][i*4] + " (p)")
        labelsTotal.append(headers[0][i*4])
        labelsTotal.append(headers[0][i*4] + " (p)")


    print(labelsSer)
    print(labelsPara)

    labelsTotal = ["Baseline", "Baseline", "SetInterval5", "SetInterval5", "adaptive_jerk", "adaptive_jerk"]

    ser_data = ser_data[0]
    para_data = para_data[0]

    numTrajecs = len(ser_data) - 2
    print("num trajecs: " + str(numTrajecs))

    optTimes = np.zeros((numTrajecs, 6))
    costReductions = np.zeros((numTrajecs, 6))
    avgNumDerivs = np.zeros((numTrajecs, 6))
    avgTimeGettingDerivs = np.zeros((numTrajecs, 6))

    for i in range(numTrajecs):
        counter = 0
        for j in range(5):

            if(j != 3 and j!=4):

                optTimes[i, (counter*2)] = ser_data[i + 2, (j * 4)]
                optTimes[i, (counter*2) + 1] = para_data[i + 2, (j * 4)]

                costReductions[i, (counter*2)] = ser_data[i + 2, (j * 4) + 1]
                costReductions[i, (counter*2) + 1] = para_data[i + 2, (j * 4) + 1]

                avgNumDerivs[i, (counter*2)] = ser_data[i + 2, (j * 4) + 2]
                avgNumDerivs[i, (counter*2) + 1] = para_data[i + 2, (j * 4) + 2]

                avgTimeGettingDerivs[i, (counter*2)] = ser_data[i + 2, (j * 4) + 3]
                avgTimeGettingDerivs[i, (counter*2) + 1] = para_data[i + 2, (j * 4) + 3]

                counter += 1


    fig, axes = plt.subplots(2, 1, figsize = (11,8))
    boxPlotTitle = "Optimisation time against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Total optimisation time (s)"
    orange = "#edb83b"
    bp1 = box_plot(optTimes, orange, yAxisLabel, axes[0], labelsTotal)

    boxPlotTitle = "Cost Reduction against interpolation methods " + "panda_pushing_clutter"
    yAxisLabel = "Cost Reduction"
    orange = "#edb83b"
    bp2 = box_plot(costReductions, orange, yAxisLabel, axes[1], labelsTotal)

    # boxPlotTitle = "Average calculated derivatives against interpolation methods " + "panda_pushing_clutter"
    # yAxisLabel = "Average calculated derivatives"
    # orange = "#edb83b"
    # bp3 = box_plot(avgNumDerivs, orange, yAxisLabel, axes[1, 0], labelsTotal)

    # boxPlotTitle = "average time getting derivatives against interpolation methods " + "panda_pushing_clutter"
    # yAxisLabel = "Average time getting derivatives (s)"
    # orange = "#edb83b"
    # bp4 = box_plot(avgTimeGettingDerivs, orange, yAxisLabel, axes[1,1], labelsTotal)
    fig.suptitle(titles, fontsize=16)
    normalBP = mpatches.Patch(color=normalPosterColour, label='Normal backwards pass')
    parallelBP = mpatches.Patch(color=alternatePosterColor, label='Parallel approximate backwards pass')
    fig.legend(handles=[normalBP, parallelBP]) 
    # plt.savefig('data/resultsData/'  + dataNumber + "/" + taskName + '_boxplots.svg', format='svg', dpi=1200)
    plt.savefig('data/resultsData/' + taskName + '_boxplots.png')
    # plt.show()

def plotResults():
    # Load data into numpy array

    taskNames = ["panda_pushing_heavy_clutter", "panda_pushing_clutter", "panda_pushing"]
    # taskNames = ["panda_pushing_heavy_clutter"]
    # titles = ["Panda pushing into heavy clutter"]
    titles = ["Panda pushing into heavy clutter", "Panda pushing into mild clutter", "Panda pushing no clutter"]

    for i in range(len(taskNames)):
        plotOneTask(taskNames[i], titles[i])

    

def box_plot(data, fill_color, yAxisTitle, ax, labels):
    normalPosterColour = "#103755"
    highlightPosterColor = "#EEF30D"
    alternatePosterColor = "#18b842"


    bp = ax.boxplot(data, patch_artist=True, meanprops={"marker":"s","markerfacecolor":highlightPosterColor, "markeredgecolor":highlightPosterColor}, showmeans=True, showfliers=False)
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
        if(index % 2 == 0):
            patch.set(facecolor=normalPosterColour)
        else:
            patch.set(facecolor=alternatePosterColor)

        index = index + 1   

    labelSize = 13

    #ax = plt.gca()
    #ax.set_facecolor(backgroundColour)

    # Make this label bigger
    # ax.set(ylabel= yAxisTitle)
    ax.set_ylabel(yAxisTitle, fontsize=labelSize)
    # ax.set_title(yAxisTitle)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(labels, fontsize=11)
        
    return bp


if __name__ == '__main__':
    plotResults()
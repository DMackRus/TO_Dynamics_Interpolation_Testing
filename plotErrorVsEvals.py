
import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import pandas as pd
import sys
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


print("PROGRAM START - PLOT ERROR VS EVALUATIONS")

# if(len(sys.argv) < 2):
#     print("not enough arguments")
#     exit()
# else:
#     mode = int(sys.argv[1])
#     print(mode)

dof = 0
num_ctrl = 0
startPath = ""
savePath = ""

mode = 1

if(mode == 0):
    dof = 2
    num_ctrl = 2
    startPath = "Pendulum/"
    savePath = "MAE - Pendulum.svg"
elif(mode == 1):
    dof = 7
    num_ctrl = 7
    startPath = "Reaching/"
    savePath = "MAE - Reaching.svg"

elif(mode == 2):
    dof = 9
    num_ctrl = 7
    startPath = "Pushing/"
    savePath = "MAE - Pushing.svg"

else:
    print("invalid mode specified")
    exit()

trajecLength = 3000
n = 20

sizeOfAMatrix = dof * 2 * dof 


def main():
    
    pandas = pd.read_csv(startPath + 'Results/error_lin.csv', header=None)
    error_lin = pandas.to_numpy()

    pandas = pd.read_csv(startPath + 'Results/error_quad.csv', header=None)
    error_quad = pandas.to_numpy()

    pandas = pd.read_csv(startPath + 'Results/error_NN.csv', header=None)
    error_NN = pandas.to_numpy()

    pandas = pd.read_csv(startPath + 'Results/evals_dynLin.csv', header=None)
    numEvalsLin = pandas.to_numpy()

    pandas = pd.read_csv(startPath + 'Results/error_dynLin.csv', header=None)
    error_dynlin = pandas.to_numpy()

    normalEvals = np.array([600, 300, 150, 60, 30, 15, 6])
    #normalEvals = np.array([2, 5, 10, 20, 50, 100, 200, 500])
    data_error_lin = np.zeros((len(error_lin), 2))
    data_error_quad = np.zeros((len(error_quad), 2))
    data_error_NN = np.zeros((len(error_NN), 2))
    data_error_dynLin = np.zeros((len(error_dynlin), 2))

    for i in range(len(error_lin)):
        data_error_lin[i, 0] = error_lin[i]
        data_error_lin[i, 1] = normalEvals[i]

        data_error_quad[i, 0] = error_quad[i]
        data_error_quad[i, 1] = normalEvals[i]

        data_error_NN[i, 0] = error_NN[i]
        data_error_NN[i, 1] = normalEvals[i]

    
    for i in range(len(error_dynlin)):

        data_error_dynLin[i, 0] = error_dynlin[i]
        data_error_dynLin[i, 1] = numEvalsLin[i]

    # meanDynError = np.mean(error_dynlin)
    # meanDynEvals = np.mean(numEvalsLin)
        
    print(data_error_dynLin)

    markers = ["s", "o", "^", "D", "X", "d", "P"]
    markersDyn = ["s", "D", "X", "P", "^", "o", "d"]

    for i in range(len(error_lin)):
        plt.scatter(data_error_lin[i,1], data_error_lin[i,0], color = "r", label='Linear', marker=markers[i])


        plt.scatter(data_error_quad[i,1], data_error_quad[i,0], color = "b", label='Quadratic', marker=markers[i])
        #plt.scatter(data_error_NN[i,1], data_error_NN[i,0], color = "m", label='NN Interpolation', marker=markers[i])
        plt.scatter(data_error_dynLin[i,1], data_error_dynLin[i,0], color = "g", label='Adaptive-Linear', marker=markersDyn[i])

    #plt.scatter(meanDynEvals, meanDynError, color = "k", label='Mean Dynamic Linear Interpolation')

    dynamicColor = "#0057c9"
    baselineColor = "#ff8400"
            
    #graphBackground = "#d6d6d6"
    #ax = plt.gca()
    #ax.set_facecolor(graphBackground)

    labelSize = 12
    markerSize = 8
    plt.xlabel("Number of dynamic gradient evaluations", fontsize=labelSize)
    plt.ylabel("Mean Abolute Error", fontsize=labelSize)
    #plt.title('Comparing different interpolation methods for different step sizes')

    customLegend = [Line2D([0], [0], color='r', lw=1, label='Linear'),
                     Line2D([0], [0], color='b', lw=1, label='Quadratic'),
                     Line2D([0], [0], color='g', lw=1, label='Adaptive-Linear'),
                     #Line2D([0], [0], color='m', lw=1, label='NN'),
                     Line2D([0], [0], marker=markers[0], color='w', label='Interval=5',
                          markerfacecolor='k', markersize=markerSize),
                          Line2D([0], [0], marker=markers[1], color='w', label='Interval=10',
                          markerfacecolor='k', markersize=markerSize),
                          Line2D([0], [0], marker=markers[2], color='w', label='Interval=20',
                          markerfacecolor='k', markersize=markerSize),
                          Line2D([0], [0], marker=markers[3], color='w', label='Interval=50',
                          markerfacecolor='k', markersize=markerSize),
                          Line2D([0], [0], marker=markers[4], color='w', label='Interval=100',
                          markerfacecolor='k', markersize=markerSize),
                          Line2D([0], [0], marker=markers[5], color='w', label='Interval=200',
                          markerfacecolor='k', markersize=markerSize),
                          Line2D([0], [0], marker=markers[6], color='w', label='Interval=500',
                          markerfacecolor='k', markersize=markerSize)
                                                            ]

    # Line2D([0], [0], marker=markers[6], color='w', label='Interval=200',markerfacecolor='k', markersize=markerSize)
    plt.legend(handles=customLegend)

    plt.savefig(savePath, format="svg")

    plt.show()

    








main()
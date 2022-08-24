
import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import pandas as pd
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

    normalEvals = np.array([1500, 600, 300, 150, 60, 30, 15, 6])
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
        
    print(data_error_dynLin)

    plt.scatter(data_error_lin[:,1], data_error_lin[:,0], color = "r", label='Linear Interpolation')
    plt.scatter(data_error_quad[:,1], data_error_quad[:,0], color = "b", label='Quadratic Interpolation')
    plt.scatter(data_error_NN[:,1], data_error_NN[:,0], color = "m", label='NN Interpolation')
    plt.scatter(data_error_dynLin[:,1], data_error_dynLin[:,0], color = "g", label='Dynamic Linear Interpolation')

    plt.xlabel("Number of dynamic gradient evaluations")
    plt.ylabel("Mean Abolute Error")
    plt.title('Comparing different interpolation methods for different step sizes')
    plt.legend()

    plt.show()








main()
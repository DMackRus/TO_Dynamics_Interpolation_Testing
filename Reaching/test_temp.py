
import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

dof = 7
num_ctrl = 7        
trajecLength = 3000
n = 20
numTrajectories = 1000

sizeOfAMatrix = dof * 2 * dof 


def main():

    pandas = pd.read_csv('time_lin.csv', header=None)
    time_lin = pandas.to_numpy()
    time_lin = np.reshape(time_lin, -1)
    print(time_lin.shape)

    pandas = pd.read_csv('time_quad.csv', header=None)
    time_quad = pandas.to_numpy()
    time_quad = np.reshape(time_quad, -1)
    print(time_quad.shape)

    pandas = pd.read_csv('time_NN.csv', header=None)
    time_NN = pandas.to_numpy()
    time_NN = np.reshape(time_NN, -1)
    print(time_NN.shape)


    labels = ['2', '5', '10', '20', '50', '100', '200', '500']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, time_lin, width, label='linear_interpolation')
    rects2 = ax.bar(x + width/2, time_quad, width, label='quadratic_interpolation')
    rects3 = ax.bar(x + width/2, time_NN, width, label='NN_interpolation')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (s)')
    ax.set_title('Time taken for Interpolation at different number of steps between derivatives')
    #ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()



    # for i in range(sizeOfAMatrix):
    #     plt.plot(A_matrices[:,i])
    #     plt.show()

    # pandas = pd.read_csv('Data/savedStates.csv', header=None)
    # states = pandas.to_numpy()

    # pandas = pd.read_csv('Data/savedControls.csv', header=None)
    # controls = pandas.to_numpy()

    # trajecStartIndex = 990

    # A_matrices = A_matrices[(trajecStartIndex * trajecLength):,:]
    # states = states[(trajecStartIndex * trajecLength):,:]
    # controls = controls[(trajecStartIndex * trajecLength):,:]

    # pandas = pd.DataFrame(A_matrices)
    # pandas.to_csv("Data/testing_trajectories.csv", header=None, index=None)

    # pandas = pd.DataFrame(states)
    # pandas.to_csv("Data/testing_states.csv", header=None, index=None)

    # pandas = pd.DataFrame(controls)
    # pandas.to_csv("Data/testing_controls.csv", header=None, index=None)



    # print("size of A_matrices " + str(A_matrices.shape))

    # # for i in range(sizeOfAMatrix):
    # #     plt.plot(A_matrices[:,i])
    # #     plt.show()





main()
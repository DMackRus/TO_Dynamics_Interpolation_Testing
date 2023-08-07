import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from interpolateDynamics import *


def main():
    # Create an interpolator object for a given task
    task = "doublePendulum"

    methods = []
    errors_methods = []
    percentage_derivs_methods = []

    # ------------ Iterative error -------------------------
    methods.append("iterativeError")
    avg_errors, avg_percentage_derivs = iter_error_method()
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    # ------------ Adaptive acceleration -------------------------



    # Adaptive_jerk
    

    methods.append("adaptiveJerk")
    errors = np.zeros((numTrajectories, numMethods))
    percentage_derivs = np.zeros((numTrajectories, numMethods))
    for i in range(numTrajectories):
        myInterpolator = interpolator(task, i)
        dof = myInterpolator.dof
        horizon = myInterpolator.trajecLength
        total_column_derivs = dof * horizon
        _, _, _, task_errors, task_keyPoints = myInterpolator.interpolateTrajectory(0, dynParams)

        for j in range(numMethods):
            errors[i][j] = task_errors[j]
            percentage_derivs[i][j] = (( len(task_keyPoints[j][0]) + len(task_keyPoints[j][1]) ) / total_column_derivs) * 100

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    

    plot_results(task, methods, errors_methods, percentage_derivs_methods)

def iter_error_method(task, numTrajectories):
    # Params for double pendulum
    numMethods = 3
    iter_error_threshold = [0.0001, 0.00001, 0.000007]
    minN = 5
    maxN = 50
    dynParams = [None] * numMethods
    for i in range(3):
        dynParams[i] = derivative_interpolator("iterativeError", minN, maxN, 0, 0, iter_error_threshold[i])
    # Iterative error
    
    errors = np.zeros((numTrajectories, numMethods))
    percentage_derivs = np.zeros((numTrajectories, numMethods))
    for i in range(numTrajectories):
        myInterpolator = interpolator(task, i)
        dof = myInterpolator.dof
        horizon = myInterpolator.trajecLength
        total_column_derivs = dof * horizon
        _, _, _, task_errors, task_keyPoints = myInterpolator.interpolateTrajectory(0, dynParams)

        for j in range(numMethods):
            errors[i][j] = task_errors[j]
            percentage_derivs[i][j] = (( len(task_keyPoints[j][0]) + len(task_keyPoints[j][1]) ) / total_column_derivs) * 100

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs

def adaptive_jerk_method(task, numTrajectories):
    # Params for double pendulum
    numMethods = 3
    minN = 5
    maxN = 50
    jerkThreshold = [0.1, 0.15, 0.2]
    dynParams = [None] * numMethods
    for i in range(3):
        dynParams[i] = derivative_interpolator("adaptiveJerk", minN, maxN, 0, jerkThreshold[i], 0)
    
    errors = np.zeros((numTrajectories, numMethods))
    percentage_derivs = np.zeros((numTrajectories, numMethods))
    for i in range(numTrajectories):
        myInterpolator = interpolator(task, i)
        dof = myInterpolator.dof
        horizon = myInterpolator.trajecLength
        total_column_derivs = dof * horizon
        _, _, _, task_errors, task_keyPoints = myInterpolator.interpolateTrajectory(0, dynParams)

        for j in range(numMethods):
            errors[i][j] = task_errors[j]
            percentage_derivs[i][j] = (( len(task_keyPoints[j][0]) + len(task_keyPoints[j][1]) ) / total_column_derivs) * 100

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs


def plot_results(task, methodNames, avg_errors, avg_percentage_derivs):

    print("Average errors: ", avg_errors)
    print("Average percetn derivs:: ", avg_percentage_derivs)

    # plot the reuslts - scatter graph
    plt.title("Average Error vs Percentage of Derivatives - " + str(task))
    plt.xlabel("Percentage of column derivatives calculated")
    plt.ylabel("Average MSE")
    plt.scatter(avg_percentage_derivs[0], avg_errors[0], color = 'r', label = methodNames[0])
    plt.scatter(avg_percentage_derivs[1], avg_errors[1], color = 'g', label = methodNames[1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
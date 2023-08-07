import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from interpolateDynamics import *


def main():
    # Create an interpolator object for a given task
    # taskName = "doublePendulum"
    taskName = "doublePendulum"
    numTrajectories = 6

    methods = []
    errors_methods = []
    percentage_derivs_methods = []

    # ------------ Iterative error -------------------------
    methods.append("Iterative Error")
    avg_errors, avg_percentage_derivs = iter_error_method(taskName, numTrajectories)
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    print("iterative_error complete - 25 %")

    # ------------ Adaptive acceleration -------------------
    methods.append("Adaptive Acell")
    avg_errors, avg_percentage_derivs = adaptive_acell_method(taskName, numTrajectories)
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    print("adaptive_acell complete - 50 %")

    # ------------ Adaptive jerk --------------------------
    methods.append("Adaptive Jerk")
    avg_errors, avg_percentage_derivs = adaptive_jerk_method(taskName, numTrajectories)
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    print("adaptive_jerk complete - 75 %")

    # ------------- Mag vel change -----------------------
    methods.append("Mag Vel Change")
    avg_errors, avg_percentage_derivs = mag_vel_change_method(taskName, numTrajectories)
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    

    plot_results(taskName, methods, errors_methods, percentage_derivs_methods)

def iter_error_method(task, numTrajectories):
    # Params for double pendulum
    if(task == "doublePendulum"):
        numMethods = 3
        iter_error_threshold = [0.0001, 0.00001, 0.000007]
        minN = 5
        maxN = 50
        dynParams = [None] * numMethods
    elif(task == "panda_reaching"):
        numMethods = 3
        iter_error_threshold = [0.004, 0.008, 0.001]
        minN = 10
        maxN = 200
        dynParams = [None] * numMethods
    
    
    for i in range(3):
        dynParams[i] = derivative_interpolator("iterativeError", minN, maxN, 0, 0, iter_error_threshold[i], 0)
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
            sum_keyPoints = 0
            for k in range(dof):
                sum_keyPoints += len(task_keyPoints[j][k])

            percentage_derivs[i][j] = (sum_keyPoints / total_column_derivs) * 100

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs

def adaptive_acell_method(task, numTrajectories):
    # Params for double pendulum
    if(task == "doublePendulum"):
        numMethods = 3
        minN = 5
        maxN = 50
        acellThreshold = [0.1, 0.15, 0.2]
        dynParams = [None] * numMethods
    elif(task == "panda_reaching"):
        numMethods = 3
        minN = 10
        maxN = 200
        acellThreshold = [0.005, 0.001, 0.007]
        dynParams = [None] * numMethods

    for i in range(3):
        dynParams[i] = derivative_interpolator("adaptiveAccel", minN, maxN, acellThreshold[i], 0, 0, 0)
    
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
            sum_keyPoints = 0
            for k in range(dof):
                sum_keyPoints += len(task_keyPoints[j][k])

            percentage_derivs[i][j] = (sum_keyPoints / total_column_derivs) * 100

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs

def adaptive_jerk_method(task, numTrajectories):
    # Params for double pendulum
    if(task == "doublePendulum"):
        numMethods = 3
        minN = 5
        maxN = 50
        jerkThreshold = [0.1, 0.15, 0.2]
        dynParams = [None] * numMethods
    elif(task == "panda_reaching"):
        numMethods = 3
        minN = 10
        maxN = 200
        jerkThreshold = [0.005, 0.001, 0.007]
        dynParams = [None] * numMethods

    for i in range(3):
        dynParams[i] = derivative_interpolator("adaptiveJerk", minN, maxN, 0, jerkThreshold[i], 0, 0)
    
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
            sum_keyPoints = 0
            for k in range(dof):
                sum_keyPoints += len(task_keyPoints[j][k])

            percentage_derivs[i][j] = (sum_keyPoints / total_column_derivs) * 100

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs

def mag_vel_change_method(task, numTrajectories):
        # Params for double pendulum
    if(task == "doublePendulum"):
        numMethods = 3
        minN = 5
        maxN = 50
        mag_vel_change = [1.5, 2, 2.5]
        dynParams = [None] * numMethods
    elif(task == "panda_reaching"):
        numMethods = 3
        minN = 10
        maxN = 200
        mag_vel_change = [1.5, 2, 2.5]
        dynParams = [None] * numMethods

    for i in range(3):
        dynParams[i] = derivative_interpolator("magVelChange", minN, maxN, 0, 0, 0, mag_vel_change[i])
    
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
            sum_keyPoints = 0
            for k in range(dof):
                sum_keyPoints += len(task_keyPoints[j][k])

            percentage_derivs[i][j] = (sum_keyPoints / total_column_derivs) * 100

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs

def plot_results(task, methodNames, avg_errors, avg_percentage_derivs):

    print("Average errors: ", avg_errors)
    print("Average percetn derivs:: ", avg_percentage_derivs)

    colors = ['#D96E26', '#38D926', '#2691D9', '#C726D9']

    # plot the reuslts - scatter graph
    plt.title("Average Error vs Percentage of Derivatives - " + str(task))
    plt.xlabel("Percentage of column derivatives calculated")
    plt.ylabel("Average MSE")
    for i in range(len(methodNames)):
        plt.scatter(avg_percentage_derivs[i], avg_errors[i], color = colors[i], label = methodNames[i])
    plt.legend()
    plt.savefig("results_interpolation_accuracy/" + str(task) + ".png")
    plt.show()

    


if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from interpolateDynamics import *


def main():
    # Create an interpolator object for a given task
    all_tasks = ["acrobot", "doublePendulum", "panda_reaching", "panda_pushing", "panda_pushing_low_clutter", "panda_pushing_heavy_clutter", 
                "panda_boxFlick", "panda_boxFlick_low_clutter", "panda_boxFlick_heavy_clutter",
                "kinova_ballPush_side", "kinova_ballPush_forwards", "kinova_ball_lift", "mini_cheetah_running"]
    # taskName = "doublePendulum"
    # taskName = "panda_reaching"
    # taskName = "panda_pushing"
    # taskName = "acrobot"
    taskName = "panda_pushing_heavy_clutter"
    numTrajectories = 100

    methods = []
    errors_methods = []
    percentage_derivs_methods = []
    dyn_parameters = []

    # ------------ Adaptive jerk --------------------------
    methods.append("Adaptive Jerk")
    avg_errors, avg_percentage_derivs, dynParams = adaptive_jerk_method(taskName, numTrajectories)
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    dynParams_list = []
    for i in range(len(dynParams)):
        dynParams_list.append([dynParams[i].keyPoint_method, dynParams[i].minN, dynParams[i].maxN, 
            dynParams[i].acellThreshold, dynParams[i].jerkThreshold, dynParams[i].iterative_error_threshold, dynParams[i].vel_change_required])

    dyn_parameters.append(dynParams_list)

    print("adaptive_jerk complete - 25 %")

    # ------------ Adaptive acceleration -------------------
    methods.append("Adaptive Acell")
    avg_errors, avg_percentage_derivs, dynParams = adaptive_acell_method(taskName, numTrajectories)
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)
    
    dynParams_list = []
    for i in range(len(dynParams)):
        dynParams_list.append([dynParams[i].keyPoint_method, dynParams[i].minN, dynParams[i].maxN, 
            dynParams[i].acellThreshold, dynParams[i].jerkThreshold, dynParams[i].iterative_error_threshold, dynParams[i].vel_change_required])

    dyn_parameters.append(dynParams_list)

    print("adaptive_acell complete - 50 %")

    # ------------ Iterative error -------------------------
    methods.append("Iterative Error")
    avg_errors, avg_percentage_derivs, dynParams = iter_error_method(taskName, numTrajectories)
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    dynParams_list = []
    for i in range(len(dynParams)):
        dynParams_list.append([dynParams[i].keyPoint_method, dynParams[i].minN, dynParams[i].maxN, 
            dynParams[i].acellThreshold, dynParams[i].jerkThreshold, dynParams[i].iterative_error_threshold, dynParams[i].vel_change_required])

    dyn_parameters.append(dynParams_list)

    print("iterative_error complete - 75 %")

    # ------------- Mag vel change -----------------------
    methods.append("Mag Vel Change")
    avg_errors, avg_percentage_derivs, dynParams = mag_vel_change_method(taskName, numTrajectories)
    errors_methods.append(avg_errors)
    percentage_derivs_methods.append(avg_percentage_derivs)

    dynParams_list = []
    for i in range(len(dynParams)):
        dynParams_list.append([dynParams[i].keyPoint_method, dynParams[i].minN, dynParams[i].maxN, 
            dynParams[i].acellThreshold, dynParams[i].jerkThreshold, dynParams[i].iterative_error_threshold, dynParams[i].vel_change_required])

    dyn_parameters.append(dynParams_list)

    # Save all the data
    np.savez("results_interpolation_accuracy/" + taskName + "_results.npz", methods=methods, 
                errors_methods=errors_methods, percentage_derivs_methods=percentage_derivs_methods, dyn_parameters=dyn_parameters)

    # Find the best hyper parameters for each method as per some pareto front optimisation
    # best_errors, best_percentage_derivs = return_best_pareto_front_settings(methods, avg_errors, avg_percentage_derivs)
    # print(methods)
    # print("Best errors: ", best_errors)
    # print("Best percentage derivs: ", best_percentage_derivs)


    plot_results(taskName, methods, errors_methods, percentage_derivs_methods)

def return_best_pareto_front_settings(methods, error_methods, percentage_derivs_methods, dyn_parameters):
    best_errors_methods = []
    best_percentage_derivs_methods = []
    best_dyn_parameters_methods = []

    for i in range(len(methods)):
        best_cost = None
        best_percentage_derivs = None
        best_error = []

        for j in range(len(error_methods[i])):
            new_cost = (100 * error_methods[i][j]) + percentage_derivs_methods[i][j]
            if(best_cost == None or (new_cost < best_cost)):
                best_cost = new_cost
                best_error = error_methods[i][j]
                best_percentage_derivs = percentage_derivs_methods[i][j]
                best_dyn_parameters = dyn_parameters[i][j]

        best_errors_methods.append(best_error)
        best_percentage_derivs_methods.append(best_percentage_derivs)
        best_dyn_parameters_methods.append(best_dyn_parameters)


    return best_errors_methods, best_percentage_derivs_methods, best_dyn_parameters_methods

def iter_error_method(task, numTrajectories):
    # Params for double pendulum
    if(task == "doublePendulum"):
        iter_error_threshold = [0.001, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
        minN = [2, 5, 10, 15, 20]
        maxN = 50
        
    elif(task == "panda_reaching"):
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        minN = [2, 5, 10, 15, 20]
        maxN = 200

    elif(task == "panda_pushing"):
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        minN = [2, 5, 10, 15, 20]
        maxN = 200

    elif(task == "panda_pushing_low_clutter"):
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        minN = [2, 5, 10, 15, 20]
        maxN = 200

    elif(task == "panda_pushing_heavy_clutter"):
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        minN = [2, 5, 10, 15, 20]
        maxN = 200

    elif(task == "acrobot"):
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        minN = [2, 5, 10, 15, 20]
        maxN = 200

    numMethods = len(iter_error_threshold) * len(minN)
    dynParams = [None] * numMethods
    
    index = 0
    for i in range(len(iter_error_threshold)):
        for j in range(len(minN)):
            dynParams[index] = derivative_interpolator("iterativeError", minN[j], maxN, 0, 0, iter_error_threshold[i], 0)
            index += 1

    end_percentage = 75
    
    errors = np.zeros((numTrajectories, numMethods))
    percentage_derivs = np.zeros((numTrajectories, numMethods))
    for i in range(numTrajectories):
        myInterpolator = interpolator(task, i)
        dof = myInterpolator.dof_vel
        horizon = myInterpolator.trajecLength
        total_column_derivs = dof * horizon
        _, _, _, task_errors, task_keyPoints, task_w_keyPoints = myInterpolator.interpolateTrajectory(0, dynParams)

        for j in range(numMethods):
            errors[i][j] = task_errors[j]
            sum_keyPoints = 0
            for k in range(dof):
                sum_keyPoints += len(task_keyPoints[j][k])

            percentage_derivs[i][j] = (sum_keyPoints / total_column_derivs) * 100

        print(f'iterative error {i/numTrajectories * end_percentage}%')

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs, dynParams

def adaptive_acell_method(task, numTrajectories):
    # Params for double pendulum
    if(task == "doublePendulum"):
        minN = [2, 5, 10, 15, 20]
        maxN = 50
        acellThreshold = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

    elif(task == "panda_reaching"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        acellThreshold = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

    elif(task == "panda_pushing"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        acellThreshold = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

    elif(task == "panda_pushing_low_clutter"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        acellThreshold = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

    elif(task == "panda_pushing_heavy_clutter"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        acellThreshold = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

    elif(task == "acrobot"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        acellThreshold = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    
    numMethods = len(acellThreshold) * len(minN)
    dynParams = [None] * numMethods

    end_percentage = 50

    index = 0
    for i in range(len(acellThreshold)):
        for j in range(len(minN)):
            dynParams[index] = derivative_interpolator("adaptiveAccel", minN[j], maxN, acellThreshold[i], 0, 0, 0)
            index += 1
    
    errors = np.zeros((numTrajectories, numMethods))
    percentage_derivs = np.zeros((numTrajectories, numMethods))
    for i in range(numTrajectories):
        myInterpolator = interpolator(task, i)
        dof = myInterpolator.dof_vel
        horizon = myInterpolator.trajecLength
        total_column_derivs = dof * horizon
        _, _, _, task_errors, task_keyPoints, task_w_keyPoints = myInterpolator.interpolateTrajectory(0, dynParams)

        for j in range(numMethods):
            errors[i][j] = task_errors[j]
            sum_keyPoints = 0
            for k in range(dof):
                sum_keyPoints += len(task_keyPoints[j][k])

            percentage_derivs[i][j] = (sum_keyPoints / total_column_derivs) * 100

        print(f'adaptive acell {i/numTrajectories * end_percentage}%')

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs, dynParams

def adaptive_jerk_method(task, numTrajectories):
    # Params for double pendulum
    if(task == "doublePendulum"):
        minN = [2, 5, 10, 15, 20]
        maxN = 50
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        
    elif(task == "panda_reaching"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    elif(task == "panda_pushing"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    elif(task == "panda_pushing_low_clutter"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    elif(task == "panda_pushing_heavy_clutter"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    elif(task == "acrobot"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    numMethods = len(jerkThreshold) * len(minN)
    dynParams = [None] * numMethods

    index = 0
    for i in range(len(jerkThreshold)):
        for j in range(len(minN)):
            dynParams[index] = derivative_interpolator("adaptiveJerk", minN[j], maxN, 0, jerkThreshold[i], 0, 0)
            index += 1

    end_percentage = 25
    
    errors = np.zeros((numTrajectories, numMethods))
    percentage_derivs = np.zeros((numTrajectories, numMethods))
    for i in range(numTrajectories):
        myInterpolator = interpolator(task, i)
        dof = myInterpolator.dof_vel
        horizon = myInterpolator.trajecLength
        total_column_derivs = dof * horizon
        _, _, _, task_errors, task_keyPoints, task_w_keyPoints = myInterpolator.interpolateTrajectory(0, dynParams)

        for j in range(numMethods):
            errors[i][j] = task_errors[j]
            sum_keyPoints = 0
            for k in range(dof):
                sum_keyPoints += len(task_keyPoints[j][k])

            percentage_derivs[i][j] = (sum_keyPoints / total_column_derivs) * 100

        print(f' adaptive jerk - {i/numTrajectories * 25}%')

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs, dynParams

def mag_vel_change_method(task, numTrajectories):
        # Params for double pendulum
    if(task == "doublePendulum"):
        minN = [2, 5, 10, 15, 20]
        maxN = 50
        mag_vel_change = [0.1, 0.5, 1, 1.5, 2, 2.5]

    elif(task == "panda_reaching"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        mag_vel_change = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    elif(task == "panda_pushing"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        mag_vel_change = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    elif(task == "panda_pushing_low_clutter"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        mag_vel_change = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    elif(task == "panda_pushing_heavy_clutter"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        mag_vel_change = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    elif(task == "acrobot"):
        minN = [2, 5, 10, 15, 20]
        maxN = 200
        mag_vel_change = [0.1, 0.5, 1, 1.5, 2, 2.5]

    numMethods = len(mag_vel_change) * len(minN)
    dynParams = [None] * numMethods

    index = 0
    for i in range(len(mag_vel_change)):
        for j in range(len(minN)):
            dynParams[index] = derivative_interpolator("magVelChange", minN[j], maxN, 0, 0, 0, mag_vel_change[i])
            index += 1

    end_percentage = 100
    
    errors = np.zeros((numTrajectories, numMethods))
    percentage_derivs = np.zeros((numTrajectories, numMethods))
    for i in range(numTrajectories):
        myInterpolator = interpolator(task, i)
        dof = myInterpolator.dof_vel
        horizon = myInterpolator.trajecLength
        total_column_derivs = dof * horizon
        _, _, _, task_errors, task_keyPoints, task_w_keyPoints = myInterpolator.interpolateTrajectory(0, dynParams)

        for j in range(numMethods):
            errors[i][j] = task_errors[j]
            sum_keyPoints = 0
            for k in range(dof):
                sum_keyPoints += len(task_keyPoints[j][k])

            percentage_derivs[i][j] = (sum_keyPoints / total_column_derivs) * 100

        print(f' mag vel change - {i/numTrajectories * end_percentage}%')

    # Calculate the average error and percentage of derivatives
    avg_errors = np.mean(errors, axis=0)
    avg_percentage_derivs = np.mean(percentage_derivs, axis=0)

    return avg_errors, avg_percentage_derivs, dynParams

def plot_results(task, methodNames, avg_errors, avg_percentage_derivs):

    # print("Average errors: ", avg_errors)
    # print("Average percetn derivs:: ", avg_percentage_derivs)

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

    # task_name = "acrobot"
    # data = np.load("results_interpolation_accuracy/" + task_name + "_results.npz")
    # method_names = data["methods"]

    # avg_errors = data["errors_methods"]
    # avg_percentage_derivs = data["percentage_derivs_methods"]
    # dyn_parameters = data["dyn_parameters"]

    # best_errors, best_percentage_derivs, best_dyn_parameters = return_best_pareto_front_settings(method_names, avg_errors, avg_percentage_derivs, dyn_parameters)
    # print(method_names)
    # print("Best errors: ", best_errors)
    # print("Best percentage derivs: ", best_percentage_derivs)
    # print("Best dyn parameters: ", best_dyn_parameters)

    # plot_results(task_name, method_names, avg_errors, avg_percentage_derivs)
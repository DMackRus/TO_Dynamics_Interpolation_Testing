import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from interpolateDynamics import *
from interpolation_settings import *


def main():
    # Create an interpolator object for a given task

    # all_tasks = ["acrobot", "panda_reaching", "panda_pushing", "panda_pushing_low_clutter", "panda_pushing_heavy_clutter", "walker", "box_sweep",
    #                 "kinova_forward", "kinova_side", "kinova_lift"]
    all_tasks = ["mini_cheetah", "box_slide"]

    for taskName in all_tasks:
        print("----------------------- " + taskName + " -----------------------")
        numTrajectories = 100

        methods = []
        errors_methods = []
        percentage_derivs_methods = []
        dyn_parameters = []

        error_set_interval = []
        percentage_derivs_set_interval = []


        set_interval_methods, jerk_keypoint_methods, vel_keypoint_methods, iter_error_keypoint_methods = return_interpolation_settings(taskName)

        # Set interval methods
        error_set_interval, percentage_derivs_set_interval = evaluate_approximation(taskName, set_interval_methods, numTrajectories)
        print("set interval complete")
        
        # ----------------------------------- Adaptive jerk methods ------------------------------------------------
        dynParams_list = []

        avg_errors, avg_percentage_derivs = evaluate_approximation(taskName, jerk_keypoint_methods, numTrajectories)
        errors_methods.append(avg_errors)
        percentage_derivs_methods.append(avg_percentage_derivs)
        methods.append("Adaptive Jerk")
        for i in range(len(jerk_keypoint_methods)):
            dynParams_list.append([jerk_keypoint_methods[i].keyPoint_method, jerk_keypoint_methods[i].minN, jerk_keypoint_methods[i].maxN, 
                0, jerk_keypoint_methods[i].jerkThreshold, 0, 0])

        dyn_parameters.append(dynParams_list)
        print("jerk keypoint complete")

        # ------------------------------------ Iterative error methods ---------------------------------------------
        dynParams_list = []

        avg_errors, avg_percentage_derivs = evaluate_approximation(taskName, iter_error_keypoint_methods, numTrajectories)
        errors_methods.append(avg_errors)
        percentage_derivs_methods.append(avg_percentage_derivs)
        methods.append("Iterative Error")
        for i in range(len(iter_error_keypoint_methods)):
            dynParams_list.append([iter_error_keypoint_methods[i].keyPoint_method, iter_error_keypoint_methods[i].minN, iter_error_keypoint_methods[i].maxN, 
                0, 0, iter_error_keypoint_methods[i].iterative_error_threshold, 0])

        dyn_parameters.append(dynParams_list)
        print("iterative keypoint complete")

        #  -----------------------------------magvel change error methods ------------------------------------------
        dynParams_list = []

        avg_errors, avg_percentage_derivs = evaluate_approximation(taskName, vel_keypoint_methods, numTrajectories)
        errors_methods.append(avg_errors)
        percentage_derivs_methods.append(avg_percentage_derivs)
        methods.append("Mag Vel Change")
        for i in range(len(vel_keypoint_methods)):
            dynParams_list.append([vel_keypoint_methods[i].keyPoint_method, vel_keypoint_methods[i].minN, vel_keypoint_methods[i].maxN, 
                0, 0, 0, vel_keypoint_methods[i].vel_change_required])
        dyn_parameters.append(dynParams_list)
        print("magvel change keypoint complete")

        # Save all the data
        np.savez("results_interpolation_accuracy/" + taskName + "_results.npz", methods=methods, 
                    errors_methods=errors_methods, percentage_derivs_methods=percentage_derivs_methods, dyn_parameters=dyn_parameters)

        # Save set inteval methods in separate file due to incompatible lengths
        np.savez("results_interpolation_accuracy/" + taskName + "_set_interval.npz", error_set_interval=error_set_interval, percentage_derivs_set_interval=percentage_derivs_set_interval)

def evaluate_approximation(task_name, keypoint_methods, numTrajectories):

    numMethods = len(keypoint_methods)

    errors = np.zeros((numTrajectories, numMethods))
    percentage_derivs = np.zeros((numTrajectories, numMethods))
    for i in range(numTrajectories):
        myInterpolator = interpolator(task_name, i)
        dof = myInterpolator.dof_vel
        horizon = myInterpolator.trajecLength
        total_column_derivs = dof * horizon
        _, _, _, task_errors, task_keyPoints, task_w_keyPoints = myInterpolator.interpolateTrajectory(0, keypoint_methods)

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
 
def plot_results(task, method_names, avg_errors, avg_percentage_derivs, set_interval_errors, set_interval_percentage_derivs, error_threshold,
                        optimal_solutions, low_error_solutions, low_percentage_solutions, dyn_parameters):


    colors = ['#D96E26', '#38D926', '#2691D9', '#C726D9', '#000000']
    setIntervals = [2, 5, 10, 15, 20]

    # plot the results - scatter graph
    plt.figure(figsize=(8, 7))
    # plt.title(str(task))
    # plt.title("Acrobot", fontsize=20)
    
    plt.xlabel("Percentage of column derivatives calculated", fontsize=14)
    plt.ylabel("MAE averaged over all trajectories", fontsize=14)
    plt.axhline(y=error_threshold, color='r', linestyle='dashed', label = "Error Threshold")
    size_c = 40
    size_m = 80
    for i in range(len(method_names)):
        for j in range(len(avg_errors[i])):
            if(optimal_solutions[i+1] == j):
                plt.scatter(avg_percentage_derivs[i][j], avg_errors[i][j], color = colors[i], s = size_m, marker = '*', label = return_dyn_params_string(dyn_parameters[i][optimal_solutions[i+1]]))
            elif(low_error_solutions[i+1] == j):
                plt.scatter(avg_percentage_derivs[i][j], avg_errors[i][j], color = colors[i], s = size_m, marker = 'd', label = return_dyn_params_string(dyn_parameters[i][low_error_solutions[i+1]]))
            elif(low_percentage_solutions[i+1] == j):
                plt.scatter(avg_percentage_derivs[i][j], avg_errors[i][j], color = colors[i], s = size_m, marker = 'x', label = return_dyn_params_string(dyn_parameters[i][low_percentage_solutions[i+1]]))
            else:
                plt.scatter(avg_percentage_derivs[i][j], avg_errors[i][j], color = colors[i], s = size_c)

    for i in range(len(set_interval_errors)):
        if i == optimal_solutions[0]:
            plt.scatter(set_interval_percentage_derivs[i], set_interval_errors[i], color = colors[-1], s = size_m, marker = '*', label = "Set Interval, minN: " + str(setIntervals[i]))
        elif i == low_error_solutions[0]:
            plt.scatter(set_interval_percentage_derivs[i], set_interval_errors[i], color = colors[-1], s = size_m, marker = 'd', label = "Set Interval, minN: " + str(setIntervals[i]))
        elif i == low_percentage_solutions[0]:
            plt.scatter(set_interval_percentage_derivs[i], set_interval_errors[i], color = colors[-1], s = size_m, marker = 'x', label = "Set Interval, minN: " + str(setIntervals[i]))
        else:
            plt.scatter(set_interval_percentage_derivs[i], set_interval_errors[i], color = colors[-1], s = size_c)
    
    #set y min to zero
    # axes = plt.gca()
    # axes.set_ylim([-0.001,0.01])

    plt.legend()
    plt.savefig("results_interpolation_accuracy/" + str(task) + ".png")
    plt.show()

def return_dyn_params_string(dyn_params):
    dyn_params_string = ""

    dyn_params_string = dyn_params_string + str(dyn_params[0])
    dyn_params_string = dyn_params_string + ", minN: " + str(int(dyn_params[1]))
    dyn_params_string = dyn_params_string + ", maxN: " + str(int(dyn_params[2]))
    if dyn_params[0] == 'adaptiveJerk':
        dyn_params_string = dyn_params_string + " - " + str(dyn_params[4])
    elif dyn_params[0] == 'iterativeError':
        dyn_params_string = dyn_params_string + " - " + str(dyn_params[5])
    elif dyn_params[0] == 'magVelChange':
        dyn_params_string = dyn_params_string + " - " + str(dyn_params[6])

    return dyn_params_string

def get_percentage_of_max_error(methods, error_methods, percentage):
    max_error = 0

    for i in range(len(methods)):
        for j in range(len(error_methods[i])):

            if error_methods[i][j] > max_error:
                max_error = error_methods[i][j]

    return max_error * (percentage / 100)

    
if __name__ == "__main__":
    # main()

    # task_names = ["doublePendulum", "panda_reaching", "panda_pushing", "panda_pushing_low_clutter", "panda_pushing_heavy_clutter", "acrobot", "kinova_side", "kinova_forward", "kinova_lift"]
    # task_names = ["acrobot", "panda_reaching", "panda_pushing", "panda_pushing_low_clutter", "panda_pushing_heavy_clutter", "box_sweep", "walker", "kinova_side", "kinova_forward", "kinova_lift"]
    # task_names = ["panda_reaching", "panda_pushing", "panda_pushing_low_clutter", "panda_pushing_heavy_clutter", "box_sweep", "walker"]
    # task_names = ["kinova_forward", "kinova_lift", "kinova_side"]
    task_names = ["box_slide"]

    for task in task_names:
        data = np.load("results_interpolation_accuracy/" + task + "_results.npz")
        data_set_interval = np.load("results_interpolation_accuracy/" + task + "_set_interval.npz")
        method_names = data["methods"]

        avg_errors = data["errors_methods"]
        avg_percentage_derivs = data["percentage_derivs_methods"]
        dyn_parameters = data["dyn_parameters"]

        # error_threshold = get_percentage_of_max_error(method_names, avg_errors, 20)
        error_threshold = 0.1


        print("------------------------------------ " + task + " ------------------------------------")
        print(method_names)

        set_interval_errors = data_set_interval["error_set_interval"]
        set_interval_percentage_derivs = data_set_interval["percentage_derivs_set_interval"]

        # Acquire three things, the optimal solutions for each method, the best solution in terms of percentage error and the best solution in terms 
        # of error

        # Optimal solution for each method
        optimal_solutions        = [None] * (len(method_names) + 1)
        low_error_solutions      = [None] * (len(method_names) + 1)
        low_percentage_solutions = [0] * (len(method_names) + 1)

        optimal_errors          = [None] * (len(method_names) + 1)
        optimal_percentages     = [None] * (len(method_names) + 1)
        low_error_errors       = [None] * (len(method_names) + 1)
        low_error_percentages  = [None] * (len(method_names) + 1)
        low_percentage_errors  = [None] * (len(method_names) + 1)
        low_percentage_percentages  = [None] * (len(method_names) + 1)

        # Get the optimal solutions for each method
        lowest_percentages = np.array([100, 100, 100, 100])

        # ------------------- temp limits -----------------------
        # MAX_PERCENTAGE = 30
        # MAX_ERROR = 0.01

        # # Loop through the set inteval methods and adaptive methods and remove all solutions with a percentage error above 30%
        # # and error above 0.01

        # # ------------------------------------ Set interval methods ---------------------------------------------
        # for i in range(len(set_interval_errors)):
        #     if set_interval_errors[i] > MAX_ERROR or set_interval_percentage_derivs[i] > MAX_PERCENTAGE:
        #         set_interval_errors[i] = None
        #         set_interval_percentage_derivs[i] = 1000

        # # ------------------------------------ Adaptive methods ---------------------------------------------
        # for i in range(len(method_names)):
        #     for j in range(len(avg_errors[i])):
        #         if avg_errors[i][j] > MAX_ERROR or avg_percentage_derivs[i][j] > MAX_PERCENTAGE:
        #             avg_errors[i][j] = None
        #             avg_percentage_derivs[i][j] = 1000

        #===================================================================================================================

        # ------------------------------------ Optimal solutions ---------------------------------------------
        # Set interval methods
        for i in range(len(set_interval_errors)):
            if set_interval_errors[i] < error_threshold:
                if set_interval_percentage_derivs[i] < lowest_percentages[0]:
                    lowest_percentages[0] = set_interval_percentage_derivs[i]
                    optimal_solutions[0] = i

                    
        optimal_errors[0] = set_interval_errors[optimal_solutions[0]]
        optimal_percentages[0] = set_interval_percentage_derivs[optimal_solutions[0]]
        #remove the optimal solution from the list 
        # set_interval_errors = np.delete(set_interval_errors, optimal_solutions[0])
        # set_interval_percentage_derivs = np.delete(set_interval_percentage_derivs, optimal_solutions[0])

        for i in range(len(method_names)):
            for j in range(len(avg_errors[i])):
                if avg_errors[i][j] < error_threshold:
                    if avg_percentage_derivs[i][j] < lowest_percentages[i + 1]:
                        lowest_percentages[i + 1] = avg_percentage_derivs[i][j]
                        optimal_solutions[i + 1] = j

        for i in range(len(method_names)):
            optimal_errors[i+1] = avg_errors[i][optimal_solutions[i+1]]
            optimal_percentages[i+1] = avg_percentage_derivs[i][optimal_solutions[i+1]]
            #remove the optimal solution from the list
            # print("length: " + str(len(avg_errors[i])))
            # print("optimal solution: " + str(optimal_solutions[i+1]))
            # avg_errors[i] = np.delete(avg_errors[i], optimal_solutions[i+1])
            # avg_percentage_derivs[i] = np.delete(avg_percentage_derivs[i], optimal_solutions[i+1])

        # ----------------------------------------------------------------------------------------------------



        # ------------------------------------ Low error solutions ---------------------------------------------

        # Get the lowest error solution
        lowest_errors = np.array([np.inf, np.inf, np.inf, np.inf])

        # Set interval methods
        for i in range(len(set_interval_errors)):
            if set_interval_errors[i] < lowest_errors[0]:
                lowest_errors[0] = set_interval_errors[i]
                low_error_solutions[0] = i

        low_error_errors[0] = set_interval_errors[optimal_solutions[0]]
        low_error_percentages[0] = set_interval_percentage_derivs[optimal_solutions[0]]
        #remove the optimal solution from the list
        # set_interval_errors[optimal_solutions[0]] = np.inf
        # set_interval_percentage_derivs[optimal_solutions[0]] = 100

        for i in range(len(method_names)):
            for j in range(len(avg_errors[i])):
                if avg_errors[i][j] < lowest_errors[i + 1]:
                    lowest_errors[i + 1] = avg_errors[i][j]
                    low_error_solutions[i + 1] = j

        for i in range(len(method_names)):
            low_error_errors[i+1] = avg_errors[i][low_error_solutions[i+1]]
            low_error_percentages[i+1] = avg_percentage_derivs[i][low_error_solutions[i+1]]
            #remove the optimal solution from the list
            # avg_errors[i][low_error_solutions] = np.inf
            # avg_percentage_derivs[i] = 100

        # -----------------------------------------------------------------------------------------------------


        # ------------------------------------ Low percentage solutions ---------------------------------------------

        # Get the lowest percentage solution
        for i in range(len(set_interval_errors)):

            if set_interval_percentage_derivs[low_percentage_solutions[0]] == None:
                low_percentage_solutions[0] = i
                # print("here?")
            elif set_interval_percentage_derivs[i] < set_interval_percentage_derivs[low_percentage_solutions[0]] and set_interval_percentage_derivs[i] != None:
                low_percentage_solutions[0] = i
                # print("low percent sol: " + str(low_percentage_solutions[0]))

        low_percentage_errors[0] = set_interval_errors[low_percentage_solutions[0]]
        low_percentage_percentages[0] = set_interval_percentage_derivs[low_percentage_solutions[0]]
        #remove the optimal solution from the list
        # set_interval_errors = np.delete(set_interval_errors, low_percentage_solutions[0])
        # set_interval_percentage_derivs = np.delete(set_interval_percentage_derivs, low_percentage_solutions[0])

        for i in range(len(method_names)):
            for j in range(len(avg_errors[i])):                
                if avg_percentage_derivs[i][j] < avg_percentage_derivs[i][low_percentage_solutions[i + 1]]:
                    low_percentage_solutions[i + 1] = j

        for i in range(len(method_names)):
            low_percentage_errors[i+1] = avg_errors[i][low_percentage_solutions[i+1]]
            low_percentage_percentages[i+1] = avg_percentage_derivs[i][low_percentage_solutions[i+1]]
            #remove the optimal solution from the list
            # avg_errors[i] = np.delete(avg_errors[i], low_percentage_solutions[i+1])
            # avg_percentage_derivs[i] = np.delete(avg_percentage_derivs[i], low_percentage_solutions[i+1])

        # -----------------------------------------------------------------------------------------------------

        # print("Optimal solution for each method: ", optimal_solutions)
        # print("Lowest error solution for each method: ", low_error_solutions)
        # print("Lowest percentage solution for each method: ", low_percentage_solutions)
        # print("optimal solution for adative jerk: ", dyn_parameters[0][optimal_solutions[1]])

        print("---------------------------------------- Optimal errors and percentages -----------------------------------------")
        print("set interval: " + str(set_interval_errors[optimal_solutions[0]]) + " " + str(set_interval_percentage_derivs[optimal_solutions[0]]))
        for i in range(len(method_names)):
            print(method_names[i] + ": " + str(avg_errors[i][optimal_solutions[i+1]]) + " " + str(avg_percentage_derivs[i][optimal_solutions[i+1]]))

        plot_results(task, method_names, avg_errors, avg_percentage_derivs, set_interval_errors, set_interval_percentage_derivs, error_threshold,
                        optimal_solutions, low_error_solutions, low_percentage_solutions, dyn_parameters)
from dataclasses import dataclass

@dataclass
class derivative_interpolator():
    keyPoint_method: str
    minN: int
    maxN: int
    acellThreshold: float
    jerkThreshold: float
    iterative_error_threshold: float
    vel_change_required: float

def return_interpolation_settings(task_name):
    interpolation_settings = []

    if task_name == "acrobot":
        minN = [2, 5, 10, 20]
        maxN_multiplier = [2, 3]
        jerkThreshold = [0.001, 0.005, 0.01, 0.1]
        mag_vel_change = [0.1, 0.5, 1, 2]
        iter_error_threshold = [0.0005, 0.0001, 0.00005, 0.00001]

    elif task_name == "panda_reaching":
        minN = [2, 5, 10, 15, 20]
        maxN_multiplier = [2, 3, 4, 5]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    elif(task_name == "panda_pushing"):
        minN = [2, 5, 10, 15, 20]
        maxN_multiplier = [2, 3, 4, 5]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    elif(task_name == "panda_pushing_low_clutter"):
        minN = [2, 5, 10, 15, 20]
        maxN_multiplier = [2, 3, 4, 5]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    elif(task_name == "panda_pushing_heavy_clutter"):
        minN = [2, 5, 10, 15, 20]
        maxN_multiplier = [2, 3, 4, 5]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        
    elif(task_name == "box_sweep"):
        minN = [2, 5, 10, 15, 20]
        maxN_multiplier = [2, 3, 4, 5]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.1, 0.5, 1, 1.5, 2, 2.5]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        
    elif(task_name == "walker"):
        minN = [1, 2, 5, 10, 20]
        maxN_multiplier = [2, 3, 4, 5]
        jerkThreshold = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        mag_vel_change = [0.1, 0.2, 0.5, 1, 1.5, 2]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        
    elif(task_name == "kinova_side"):
        minN = [1, 2, 5]
        maxN_multiplier = [2, 3, 4]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.1, 0.5, 1, 1.5, 2, 2.5]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        
    elif(task_name == "kinova_forward"):
        minN = [1, 2, 5]
        maxN_multiplier = [2, 3, 4]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.1, 0.5, 1, 1.5, 2, 2.5]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        
    elif(task_name == "kinova_lift"):
        minN = [1, 2, 5]
        maxN_multiplier = [2, 3, 4]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.1, 0.5, 1, 1.5, 2, 2.5]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    # TODO - look at these values
    elif(task_name == "mini_cheetah"):
        minN = [1, 2, 5]
        maxN_multiplier = [2, 3, 4]
        jerkThreshold = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        mag_vel_change = [0.1, 0.5, 1, 1.5, 2, 2.5]
        iter_error_threshold = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    # return 3 lists of derivative interpolators, one for jerk, one for vel and one for iterative error
    # each list will have minN * maxN_multiplier interpolators

    jerk_keypoint_methods = []
    vel_keypoint_methods = []
    iter_error_keypoint_methods = []
    set_interval_methods = []

    # Jerk methods
    for i in range(len(minN)):
        for j in range(len(maxN_multiplier)):
            for k in range(len(jerkThreshold)):
                maxN = minN[i] * maxN_multiplier[j]
                jerk_keypoint_methods.append(derivative_interpolator("adaptiveJerk", minN[i], maxN, 0, jerkThreshold[k], 0, 0))

    # vel methods
    for i in range(len(minN)):
        for j in range(len(maxN_multiplier)):
            for k in range(len(mag_vel_change)):
                maxN = minN[i] * maxN_multiplier[j]
                vel_keypoint_methods.append(derivative_interpolator("magVelChange", minN[i], maxN, 0, 0, 0, mag_vel_change[k]))

    # iterative error methods
    for i in range(len(minN)):
        for j in range(len(maxN_multiplier)):
            for k in range(len(iter_error_threshold)):
                maxN = minN[i] * maxN_multiplier[j]
                iter_error_keypoint_methods.append(derivative_interpolator("iterativeError", minN[i], maxN, 0, 0, iter_error_threshold[k], 0))

    set_interval_minN = [2, 5, 10, 15, 20]
    for i in range(len(set_interval_minN)):
        set_interval_methods.append(derivative_interpolator("setInterval", set_interval_minN[i], 0, 0, 0, 0, 0))

    return set_interval_methods, jerk_keypoint_methods, vel_keypoint_methods, iter_error_keypoint_methods
        
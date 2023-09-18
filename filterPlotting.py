import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

def pltFilterData(modelName):

    unfiltered_data = returnUnfilteredLists(modelName)
    # tests_FIR = ["FIR_0", "FIR_1", "FIR_2", "FIR_3", "FIR_4"]
    tests_FIR = ["FIR_0"]
    # tests_lowPass = ["lowPass0.050000", "lowPass0.100000", "lowPass0.150000", "lowPass0.200000", "lowPass0.250000", "lowPass0.300000"]
    tests_lowPass = ["lowPass0.200000"]
    num_tests_FIR = len(tests_FIR)

    tests = tests_FIR + tests_lowPass
    plot_unfiltered = True

    orange = '#FFA500'
    green = '#008000'
    black = '#000000'

    for i in range(len(tests)):
        fileExtension = tests[i]

        filtered_data = returnFilteredLists(modelName, fileExtension)
        avgCostDiff, avgFiltered, avgUnfiltered = averageDiffPerIteration(unfiltered_data, filtered_data)

        if plot_unfiltered:
            plt.plot(avgUnfiltered, label = 'unfiltered', linewidth = 1, color = black)
            plot_unfiltered = False

        if(i < num_tests_FIR):
            plt.plot(avgFiltered, label = fileExtension)
        else:
            plt.plot(avgFiltered, label = fileExtension)


        plt.legend()
        plt.title("Average cost diff filtering" )
        # plt.show()

    plt.show()

    # average the differnece between the two numbers

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

def returnFilteredLists(modelName, fileExtension):
    data_list = []
    maxLength = 0
    num_trajecs = 100

    for i in range(num_trajecs):
        temp = np.array([genfromtxt('data/filtering/' + modelName + "/" + fileExtension + '/' + str(i) + '.csv', delimiter=',')])
        temp = np.delete(temp, -1)
        
        tempList = list(temp)
        if(len(tempList) > maxLength):
            maxLength = len(tempList)

        data_list.append(tempList)

    maxLength = 9
    print("max length is: " + str(maxLength))

    for i in range(num_trajecs):
        maxVal = data_list[i][0]
        for j in range(len(data_list[i])):
            data_list[i][j] = data_list[i][j] / maxVal

    for i in range(num_trajecs):
        lenofCurrent = len(data_list[i]) - 1
        diff = maxLength - lenofCurrent
        for j in range(diff):
            data_list[i].append(data_list[i][lenofCurrent])

    return data_list

def returnUnfilteredLists(modelName):
    data_list = []
    maxLength = 0
    num_trajecs = 100


    for i in range(num_trajecs):
        temp = np.array([genfromtxt('data/filtering/' + modelName + "/none/" + str(i) + '.csv', delimiter=',')])
        temp = np.delete(temp, -1)
        
        tempList = list(temp)
        if(len(tempList) > maxLength):
            maxLength = len(tempList)

        data_list.append(tempList)

    maxLength = 9
    print("max length is: " + str(maxLength))

    for i in range(num_trajecs):
        maxVal = data_list[i][0]
        for j in range(len(data_list[i])):
            data_list[i][j] = data_list[i][j] / maxVal

    for i in range(num_trajecs):
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

if __name__ == "__main__":
    pltFilterData("push_nCl")
    # pltFilterData("push_mCl")
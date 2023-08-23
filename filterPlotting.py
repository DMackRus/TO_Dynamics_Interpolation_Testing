def main():
    # data_list = np.array([[0, 0, 0]])
    # data_2 = np.array([[1, 1, 1]])

    # test = np.concatenate((data_list, data_2), axis=0)
    # print(test)
    unfiltered_data = returnUnfilteredLists()
    # tests = ["1Hz","50Hz","100Hz","200Hz", "500Hz"]
    tests = ["0.05", "0.1", "0.15", "0.2", "0.25"]

    for i in range(len(tests)):
        fileExtension = tests[i]

        filtered_data = returnFilteredLists(fileExtension)
        avgCostDiff, avgFiltered, avgUnfiltered = averageDiffPerIteration(unfiltered_data, filtered_data)

        plt.plot(avgCostDiff, label = 'cost diff')
        plt.plot(avgUnfiltered, label = 'unfiltered')
        plt.plot(avgFiltered, label = 'filtered')
        plt.legend()
        plt.title("Average cost diff - " + tests[i])
        plt.show()

    # average the differnece between the two numbers

    for i in range(50):

        plt.plot(filtered_data[i], label = 'filtered')
        plt.plot(unfiltered_data[i], label = 'unfiltered')
        plt.title("plot " + str(i))
        plt.legend()
        plt.show()

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

def returnFilteredLists(fileExtension):
    data_list = []
    maxLength = 0

    # Load the 50 csvs
    for i in range(50):
        temp = np.array([genfromtxt('data/filtering_' + fileExtension + '/' + str(i) + '.csv', delimiter=',')])
        temp = np.delete(temp, -1)
        
        tempList = list(temp)
        if(len(tempList) > maxLength):
            maxLength = len(tempList)

        data_list.append(tempList)

    maxLength = 9
    print("max length is: " + str(maxLength))

    for i in range(50):
        maxVal = data_list[i][0]
        for j in range(len(data_list[i])):
            data_list[i][j] = data_list[i][j] / maxVal

    for i in range(50):
        lenofCurrent = len(data_list[i]) - 1
        diff = maxLength - lenofCurrent
        for j in range(diff):
            data_list[i].append(data_list[i][lenofCurrent])

    return data_list

def returnUnfilteredLists():
    data_list = []
    maxLength = 0

    # Load the 50 csvs
    for i in range(50):
        temp = np.array([genfromtxt('data/no_filtering/' + str(i) + '.csv', delimiter=',')])
        temp = np.delete(temp, -1)
        
        tempList = list(temp)
        if(len(tempList) > maxLength):
            maxLength = len(tempList)

        data_list.append(tempList)

    maxLength = 9
    print("max length is: " + str(maxLength))

    for i in range(50):
        maxVal = data_list[i][0]
        for j in range(len(data_list[i])):
            data_list[i][j] = data_list[i][j] / maxVal

    for i in range(50):
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
import os 
import sys

if(len(sys.argv) < 2):
    print("not enough arguments")
    exit()
else:
    mode = int(sys.argv[1])

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


def main():

    # Create training Data and save all necessary files
    program = "python3 createTrainingData.py " + str(mode)
    os.system(program)

    # Train a neural network on the training data
    program = "python3 trainModel.py " + str(mode)
    os.system(program)

    # Evalute the four interpolation methods and compute a MAE metric for them
    program = "python3 evaluateInterpolationMethods.py " + str(mode)
    os.system(program)





main()
import matplotlib.pyplot as plt
from platform import python_version
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
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
numTrajectories = 1000

sizeOfAMatrix = dof * 2 * dof 
sizeOfBMatrix = dof * num_ctrl

optimiser = tf.keras.optimizers.Adam(learning_rate = 0.01)

def createModel():

    model_A = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.InputLayer(input_shape=(211), name='Input_Layer'),
        tf.keras.layers.Dense(4000, activation=tf.nn.relu),
        tf.keras.layers.Dense(4000, activation=tf.nn.relu),
        tf.keras.layers.Dense(dof * 2 * dof, activation=tf.nn.relu)
    ])

    model_A.compile(optimizer='adam', loss=tf.losses.mean_squared_error, metrics=['mse'])

    return model_A


def main():
    print("PROGRAM STARTED")

    model_A = createModel()

    pandas = pd.read_csv(startPath + 'NN_Data/trainDataInputs_A.csv', header=None)
    loadInputData_A = pandas.to_numpy()

    pandas = pd.read_csv(startPath + 'NN_Data/trainDataOutputs_A.csv', header=None)
    loadOutputData_A = pandas.to_numpy()
    print("DATA LOADED")

    print(loadOutputData_A.shape)
    print(loadInputData_A.shape)

    splitIndex = 960 * 3000
    loadInputData_A = loadInputData_A[:splitIndex,:]
    loadOutputData_A = loadOutputData_A[:splitIndex,:]

    print(loadInputData_A.shape)

    # x, x_test, y, y_test = train_test_split(loadInputData_A, loadOutputData_A, test_size=0.15, shuffle=True)

    # Split the remaining data to train and validation
    x_train, x_val, y_train, y_val = train_test_split(loadInputData_A, loadOutputData_A, test_size=0.15, shuffle=True)

    print("DATA SEPARATED INTO TRAIN AND VALIDATION")

    # Training the Keras model
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model_A.fit(x=x_train, y=y_train, batch_size=100, epochs=50, validation_data=(x_val, y_val), callbacks = [callback])
    print("MODEL FINISHED TRAINING")

    model_A.save(startPath + "models/A_model.model")

    # plt.plot(model_A.history.history['loss'])
    # plt.plot(model_A.history.history['val_loss'])
    # plt.show()

    # difference = Outputs.iloc[0].to_numpy() - result
    # print(difference)


    print("PRORAM ENDED")












main()
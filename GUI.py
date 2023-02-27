import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from numpy import genfromtxt
# from ttkwidgets.autocomplete import AutocompleteEntry
from interpolateDynamics import *

class dynamicsGUI():
    def __init__(self, master):
        self.master = master
        self.master.title("GUI")
        self.master.geometry("1400x700")
        self.master.resizable(True, True)
        self.master.title('Interactive Dynamics')

        darkBlue = '#0a0a0a'
        lightBlue = '#00008B'
        orange = '#f2a52a'

        self.plotFrame = tk.Frame(self.master)
        self.plotFrame.pack(side=TOP, fill=X)

        self.AB_widgetsFrame = tk.Frame(self.master, bg=lightBlue)
        self.AB_widgetsFrame.pack(side=LEFT, fill=BOTH, expand=True)

        self.state_widgetFrame = tk.Frame(self.master, bg=lightBlue)
        self.state_widgetFrame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.totalGraphData = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # the figure that will contain the plot
        self.fig_AB = plt.Figure(figsize = (7, 5),
                    dpi = 100)
        self.fig_trajecInfo = plt.Figure(figsize = (7, 5),
                    dpi = 100)
    
        # adding the subplot
        self.plot_AB = self.fig_AB.add_subplot(111)
        self.plot_trajecInfo = self.fig_trajecInfo.add_subplot(111)
    
        # plotting the graph
        self.plot_AB.plot(self.totalGraphData)
        self.plot_trajecInfo.plot(self.totalGraphData)
        

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas_AB = FigureCanvasTkAgg(self.fig_AB,
                                master = self.plotFrame) 
        
        self.canvas_trajecInfo = FigureCanvasTkAgg(self.fig_trajecInfo, master = self.plotFrame)
  
        # placing the canvas on the Tkinter window
        self.canvas_AB.get_tk_widget().grid(row=0, column=0)
        self.canvas_trajecInfo.get_tk_widget().grid(row=0, column=1)
    
        # # creating the Matplotlib toolbar
        # toolbar = NavigationToolbar2Tk(self.canvas,
        #                             self.master)
        # toolbar.update()
    
        # # placing the toolbar on the Tkinter window
        # self.canvas.get_tk_widget().pack(side=LEFT)

        # frame = Frame(self.master, bg='#f25252')
        # frame.pack(expand=True)

        settingsWidth = 50

        self.label_minN = tk.Label(self.AB_widgetsFrame, text="minN", width=settingsWidth)
        self.entry_minN = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_minN.insert(0, "5")
        self.label_maxN = tk.Label(self.AB_widgetsFrame, text="maxN", width=settingsWidth)
        self.entry_maxN = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_maxN.insert(0, "200")
        self.label_jerkSensitivity = tk.Label(self.AB_widgetsFrame, text="jerkSensitivity", width=settingsWidth)
        self.entry_jerkSensitivity = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_jerkSensitivity.insert(0, "0.0002")
        self.label_displayIndex = tk.Label(self.AB_widgetsFrame, text="displayIndex", width=settingsWidth)
        self.entry_displayIndex = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_displayIndex.insert(0, "2")

        self.label_minN.pack(side=TOP)
        self.entry_minN.pack(side=TOP)

        self.label_maxN.pack(side=TOP)
        self.entry_maxN.pack(side=TOP)

        self.label_jerkSensitivity.pack(side=TOP)
        self.entry_jerkSensitivity.pack(side=TOP)

        self.label_displayIndex.pack(side=TOP)
        self.entry_displayIndex.pack(side=TOP)

        self.button = tk.Button(self.AB_widgetsFrame, text="Evaluate", command=self.displayMode_callback)
        self.button.pack(side=TOP)

        #label for state type
        self.label_stateType = tk.Label(self.state_widgetFrame, text="state type", width=settingsWidth)
        #entry box for state type
        self.stateTypes = ["Jerk", "Acceleration", "Velocity", "Position"]

        # make a list box with my state types
        #self.listbox_stateType = tk.Listbox(self.state_widgetFrame, width=settingsWidth)

        self.entry_stateIndex = tk.Entry(self.state_widgetFrame, width=settingsWidth)
        self.entry_stateIndex.insert(0, "0")

        self.button_drawTrajecInfo = tk.Button(self.state_widgetFrame, text="draw trajectory information", command = self.drawTrajecInfo, width=settingsWidth)
        
        
        self.label_stateType.pack(side=TOP)
        #self.listbox_stateType.pack(side=TOP)
        self.entry_stateIndex.pack(side=TOP)


        
        self.button_drawTrajecInfo.pack(side=TOP)

        self.interpolator = interpolator(0, 1, 3000)
        self.dynParams = []
        self.trajectoryNumber = 0

        #self.updatePlot()
        #self.drawTrajecInfo()

    def drawTrajecInfo(self):

        jerkProfile, states, controls = self.interpolator.returnTrajecInformation()
        index = int(self.entry_stateIndex.get())

        self.plot_trajecInfo.clear()
        self.plot_trajecInfo.plot(states[:,index], label='Ground truth')
        #self.plot_trajecInfo.scatter(self.reEvaluationIndices, highlightedIndices[:, index], s=25, color = interpolatedColor)
        #self.plot_trajecInfo.plot(self.interpolatedTrajec[:,index], color = interpolatedColor, label = 'Interpolated')
        self.plot_trajecInfo.legend(loc='upper right')
        self.canvas_trajecInfo.draw()

    def displayMode_callback(self):

        self.updatePlot()

    def updatePlot(self):
        dynParams = self.returnDynParams()

        # if dyn params are the same, don't recompute
        if dynParams == self.dynParams:
            print("same dyn params - no recomputation")
        else:
            self.dynParams = dynParams
            self.trueTrajec, self.interpolatedTrajec, self.reEvaluationIndices = self.interpolator.interpolateTrajectory(0, self.dynParams)

        index = int(self.entry_displayIndex.get())

        #plotting colours
        orange = "#ffa600"
        groundTruthColor = "#0057c9"
        interpolatedColor = "#ff8400"
        graphBackground = "#d6d6d6"

        #every_2nd = testTrajectories[j][::4]

        highlightedIndices = np.copy(self.trueTrajec[self.reEvaluationIndices, ])

        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)

        #print the shape of the interpolated trajectory
        try:
            print(self.interpolatedTrajec.shape)
        except:
            print("no interpolated trajectory")

        self.plot_AB.clear()
        self.plot_AB.plot(self.trueTrajec[:,index], color = groundTruthColor, label='Ground truth')
        self.plot_AB.scatter(self.reEvaluationIndices, highlightedIndices[:, index], s=25, color = interpolatedColor)
        self.plot_AB.plot(self.interpolatedTrajec[:,index], color = interpolatedColor, label = 'Interpolated')
        self.plot_AB.legend(loc='upper right')
        self.plot_AB.title.set_text("A matrix val over trajectory ")
        self.canvas_AB.draw()

    def returnDynParams(self):
        minN = int(self.entry_minN.get())
        maxN = int(self.entry_maxN.get())
        jerkSensitivity = float(self.entry_jerkSensitivity.get())

        return [minN, maxN, jerkSensitivity]
    

if __name__ == "__main__":
    root = Tk()
    myGUI = dynamicsGUI(root)
    root.mainloop()
    



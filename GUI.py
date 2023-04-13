import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from numpy import genfromtxt
# from ttkwidgets.autocomplete import AutocompleteEntry
from interpolateDynamics import *
from matplotlib.offsetbox import AnchoredText

class dynamicsGUI():
    def __init__(self, master):
        self.master = master
        self.master.title("GUI")
        self.master.geometry("1400x700")
        self.master.resizable(True, True)
        self.master.title('Interactive Dynamics')

        self.interpolationTypes = ["linear", "iterativeLin", "quadratic", "cubic"]
        self.interpTypeNum = 1

        self.darkBlue = '#103755'
        self.lightBlue = '#90CBCB'
        self.teal = '#59CBCB'
        self.yellow = '#EEF30D'
        self.white = '#FFFFFF'

        self.numEvals = 0

        self.plotFrame = tk.Frame(self.master)
        self.plotFrame.pack(side=TOP, fill=X)

        self.AB_widgetsFrame = tk.Frame(self.master, bg=self.darkBlue)
        self.AB_widgetsFrame.pack(side=LEFT, fill=BOTH, expand=True)

        self.state_widgetFrame = tk.Frame(self.master, bg=self.darkBlue)
        self.state_widgetFrame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.totalGraphData = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # the figure that will contain the plot
        self.fig_AB = plt.Figure(figsize = (7, 5),
                    dpi = 100)
        self.fig_AB.set_facecolor(self.darkBlue)
        self.fig_trajecInfo = plt.Figure(figsize = (7, 5),
                    dpi = 100)
        self.fig_trajecInfo.set_facecolor(self.darkBlue)
    
        # adding the subplot
        self.plot_AB = self.fig_AB.add_subplot(111)
        self.plot_trajecInfo = self.fig_trajecInfo.add_subplot(111)

        self.plot_AB.set_title('A matrix val over trajectory', fontsize=15, color= self.white, fontweight='bold')
        self.plot_trajecInfo.set_title('trajec Info', fontsize=15, color= self.white, fontweight='bold')

        self.plot_AB.tick_params(axis='x', colors=self.white)
        self.plot_AB.tick_params(axis='y', colors=self.white)
        self.plot_trajecInfo.tick_params(axis='x', colors=self.white)
        self.plot_trajecInfo.tick_params(axis='y', colors=self.white)
        self.plot_AB.set_facecolor(color = self.teal)
        self.plot_trajecInfo.set_facecolor(color = self.teal)
        at = AnchoredText(str(self.numEvals),
                  prop=dict(size=15), frameon=True, loc='lower right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.plot_AB.add_artist(at)
    
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

        settingsWidth = 10

        self.label_minN = tk.Label(self.AB_widgetsFrame, text="minN", width=settingsWidth)
        self.entry_minN = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_minN.insert(0, "5")
        self.label_maxN = tk.Label(self.AB_widgetsFrame, text="maxN", width=settingsWidth)
        self.entry_maxN = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_maxN.insert(0, "200")
        self.label_jerkSensitivity = tk.Label(self.AB_widgetsFrame, text="jerkSensitivity", width=settingsWidth)
        self.entry_jerkSensitivity = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_jerkSensitivity.insert(0, "0.002")
        self.label_displayIndex = tk.Label(self.AB_widgetsFrame, text="displayIndex", width=settingsWidth)
        self.entry_displayIndex = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_displayIndex.insert(0, "2")
        self.button_evaluate = tk.Button(self.AB_widgetsFrame, text="Evaluate", command=self.displayMode_callback)

        self.button_MinN_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incMinN_callback)
        self.button_MinN_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decMinN_callback)

        self.button_MaxN_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incMaxN_callback)
        self.button_MaxN_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decMaxN_callback)

        self.button_jerkSens_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incJerkSens_callback)
        self.button_jerkSens_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decJerkSens_callback)

        self.button_displayIndex_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incdisplayIndex_callback)
        self.button_displayIndex_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decdisplayIndex_callback)

        self.label_interpType = tk.Label(self.AB_widgetsFrame, text = "Interpolation type", width=settingsWidth)
        self.entry_interpType = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_interpType.insert(0, self.interpolationTypes[self.interpTypeNum])
        self.button_interpType_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incInterpType_callback)
        self.button_interpType_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decInterpType_callback)

        self.label_minN.grid(row=0, column=0, columnspan = 3, sticky='EW')

        self.button_MinN_dec.grid(row=1, column=0)
        self.entry_minN.grid(row=1, column=1)
        self.button_MinN_inc.grid(row=1, column=2)
        
        self.label_maxN.grid(row=2, column=0, columnspan = 3, sticky='EW')
        
        self.button_MaxN_dec.grid(row=3, column=0)
        self.entry_maxN.grid(row=3, column=1)
        self.button_MaxN_inc.grid(row=3, column=2)

        self.label_jerkSensitivity.grid(row=4, columnspan = 3, sticky='EW')

        self.button_jerkSens_dec.grid(row=5, column=0)
        self.entry_jerkSensitivity.grid(row=5, column=1)
        self.button_jerkSens_inc.grid(row=5, column=2)

        self.label_displayIndex.grid(row=6, column=0, columnspan = 3, sticky='EW')

        self.button_displayIndex_dec.grid(row=7, column=0)
        self.entry_displayIndex.grid(row=7, column=1)
        self.button_displayIndex_inc.grid(row=7, column=2)

        self.button_evaluate.grid(row=0, column=3,columnspan = 3, sticky='EW')

        self.label_interpType.grid(row=1, column=3, columnspan = 3, sticky='EW')

        self.button_interpType_dec.grid(row=2, column=3)
        self.entry_interpType.grid(row=2, column=4)
        self.button_interpType_inc.grid(row=2, column=5)

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

    def drawTrajecInfo(self):

        jerkProfile, states, controls = self.interpolator.returnTrajecInformation()
        index = int(self.entry_stateIndex.get())

        self.plot_trajecInfo.clear()
        self.plot_trajecInfo.plot(states[:,index], label='Ground truth')
        #self.plot_trajecInfo.scatter(self.reEvaluationIndices, highlightedIndices[:, index], s=25, color = interpolatedColor)
        #self.plot_trajecInfo.plot(self.interpolatedTrajec[:,index], color = interpolatedColor, label = 'Interpolated')
        self.plot_trajecInfo.set_title('trajec Info', fontsize=15, color= self.white, fontweight='bold')
        self.canvas_trajecInfo.draw()

    def incMinN_callback(self):
        val = int(self.entry_minN.get())
        self.entry_minN.delete(0, END)
        self.entry_minN.insert(0, val+1)


    def decMinN_callback(self):
        val = int(self.entry_minN.get())
        self.entry_minN.delete(0, END)
        self.entry_minN.insert(0, val-1)

    def incMaxN_callback(self):
        val = int(self.entry_maxN.get())
        self.entry_maxN.delete(0, END)
        self.entry_maxN.insert(0, val+1)


    def decMaxN_callback(self):
        val = int(self.entry_maxN.get())
        self.entry_maxN.delete(0, END)
        self.entry_maxN.insert(0, val-1)

    def incJerkSens_callback(self):
        val = float(self.entry_jerkSensitivity.get())
        self.entry_jerkSensitivity.delete(0, END)
        self.entry_jerkSensitivity.insert(0, str(val+0.0001))
    
    def decJerkSens_callback(self):
        val = float(self.entry_jerkSensitivity.get())
        self.entry_jerkSensitivity.delete(0, END)
        self.entry_jerkSensitivity.insert(0, str(val-0.0001))

    def incdisplayIndex_callback(self):
        val = int(self.entry_displayIndex.get())
        self.entry_displayIndex.delete(0, END)
        self.entry_displayIndex.insert(0, val+1)
        self.updatePlot()
    
    def decdisplayIndex_callback(self):
        val = int(self.entry_displayIndex.get())
        self.entry_displayIndex.delete(0, END)
        self.entry_displayIndex.insert(0, val-1)
        self.updatePlot()

    def incInterpType_callback(self):
        self.interpTypeNum = self.interpTypeNum + 1
        self.entry_interpType.delete(0, END)
        self.entry_interpType.insert(0, self.interpolationTypes[self.interpTypeNum])
        self.updatePlot()

    def decInterpType_callback(self):
        self.interpTypeNum = self.interpTypeNum - 1
        self.entry_interpType.delete(0, END)
        self.entry_interpType.insert(0, self.interpolationTypes[self.interpTypeNum])
        self.updatePlot()

    def displayMode_callback(self):

        self.updatePlot()

    def updatePlot(self):
        dynParams = self.returnDynParams()

        # if dyn params are the same, don't recompute
        if dynParams == self.dynParams:
            pass
        else:
            self.dynParams = dynParams
            self.trueTrajec, self.interpolatedTrajec, self.errors, self.reEvaluationIndices, self.iterativeKeyPoints = self.interpolator.interpolateTrajectory(0, self.dynParams)

        index = int(self.entry_displayIndex.get())

        dynamicColor = "#EEF30D"
        baselineColor = "#000000"

        highlightedIndices = np.copy(self.trueTrajec[self.reEvaluationIndices, ])
        highlightedIndicesIterative = np.copy(self.trueTrajec[self.iterativeKeyPoints, ])
        self.numEvals = len(self.reEvaluationIndices)

        self.plot_AB.clear()

        self.plot_AB.plot(self.trueTrajec[:,index], color = baselineColor, label='Ground truth')

        if(self.interpTypeNum == 1):
            self.plot_AB.scatter(self.iterativeKeyPoints, highlightedIndicesIterative[:, index], s=10, color = dynamicColor, zorder=10)
        else:
            self.plot_AB.scatter(self.reEvaluationIndices, highlightedIndices[:, index], s=10, color = dynamicColor, zorder=10)

        self.plot_AB.plot(self.interpolatedTrajec[self.interpTypeNum,:,index], color = dynamicColor, label = 'Interpolated')
        self.plot_AB.legend(loc='upper right')
        self.plot_AB.set_title('A matrix val over trajectory', fontsize=15, color= self.white, fontweight='bold')

        evalsString = ""
        if(self.interpTypeNum == 1):
            evalsString = "Evals: " + str(len(self.iterativeKeyPoints))

        else:
            evalsString = "Evals: " + str(self.numEvals)

        at = AnchoredText(evalsString,
                    prop=dict(size=15), frameon=True, loc='lower right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.plot_AB.add_artist(at)


        at2 = AnchoredText("Error: " + str(round(self.errors[self.interpTypeNum], 2)) + "",
                       loc='lower left', prop=dict(size=8), frameon=True,
                       bbox_to_anchor=(0., 1.),
                       bbox_transform=self.plot_AB.transAxes
                       )
        at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.plot_AB.add_artist(at2)
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
    



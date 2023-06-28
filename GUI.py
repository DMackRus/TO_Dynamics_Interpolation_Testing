import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import tkinter as tk
from ttkwidgets.autocomplete import AutocompleteEntry
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from numpy import genfromtxt
from interpolateDynamics import *
from matplotlib.offsetbox import AnchoredText

class dynamicsGUI():
    def __init__(self, master):
        self.master = master
        self.master.title("GUI")
        self.master.geometry("1400x700")
        self.master.resizable(True, True)
        self.master.title('Interactive Dynamics')

        self.interpolationTypes = ["setInterval", "adaptive_accel", "adaptive_jerk", "iterative_error"]
        self.interpTypeNum = 2
        self.stateTypes = ["Position", "Velocity", "Acceleration", "Jerk", "Control"]
        self.stateDisplayNumber = 1
        self.stateDisplayDof = 2

        self.darkBlue = '#103755'
        self.lightBlue = '#90CBCB'
        self.teal = '#59CBCB'
        self.yellow = '#EEF30D'
        self.white = '#FFFFFF'
        self.black = '#000000'

        self.numEvals = 0
        self.dynParams = []

        self.setupGUI()
        self.load_callback()

        self.trajectoryNumber = 0

        self.updatePlot_trajecInfo()

    def setupGUI(self):
        self.plotFrame = tk.Frame(self.master)
        self.plotFrame.pack(side=TOP, fill=X)

        self.AB_widgetsFrame = tk.Frame(self.master, bg=self.darkBlue)
        self.AB_widgetsFrame.pack(side=LEFT, fill=BOTH, expand=True)

        self.state_widgetFrame = tk.Frame(self.master, bg=self.darkBlue)
        self.state_widgetFrame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.setupPlots()
        self.setupButtons()

    def setupPlots(self):

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
        
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas_AB = FigureCanvasTkAgg(self.fig_AB,
                                master = self.plotFrame) 
        
        self.canvas_trajecInfo = FigureCanvasTkAgg(self.fig_trajecInfo, master = self.plotFrame)

        # placing the canvas on the Tkinter window
        self.canvas_AB.get_tk_widget().grid(row=0, column=0)
        self.canvas_trajecInfo.get_tk_widget().grid(row=0, column=1)

        toolbar = NavigationToolbar2Tk(self.canvas_trajecInfo,
                                    self.master)
        # toolbar.grid(row=1, column=0)
        toolbar.update()

        # # creating the Matplotlib toolbar
        
    
        # # placing the toolbar on the Tkinter window
        # self.canvas.get_tk_widget().pack(side=LEFT)

        # frame = Frame(self.master, bg='#f25252')
        # frame.pack(expand=True)

    def setupButtons(self):
        settingsWidth = 10

        # ------------------------------------ A/B matrix widgets --------------------------------------------------------
        self.label_minN = tk.Label(self.AB_widgetsFrame, text="minN", width=settingsWidth)
        self.entry_minN = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_minN.insert(0, "5")
        self.label_maxN = tk.Label(self.AB_widgetsFrame, text="maxN", width=settingsWidth)
        self.entry_maxN = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_maxN.insert(0, "200")
        self.label_jerkSensitivity = tk.Label(self.AB_widgetsFrame, text="jerkSensitivity", width=settingsWidth)
        self.entry_jerkSensitivity = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_jerkSensitivity.insert(0, "0.00015") 
        self.label_displayIndexRow = tk.Label(self.AB_widgetsFrame, text="displayIndexRow", width=settingsWidth)
        self.entry_displayIndexRow = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_displayIndexRow.insert(0, "0")
        self.label_displayIndexCol = tk.Label(self.AB_widgetsFrame, text="displayIndexCol", width=settingsWidth)
        self.entry_displayIndexCol = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_displayIndexCol.insert(0, "2")
        self.button_evaluate = tk.Button(self.AB_widgetsFrame, text="Evaluate", command=self.displayMode_callback)

        self.button_MinN_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incMinN_callback)
        self.button_MinN_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decMinN_callback)

        self.button_MaxN_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incMaxN_callback)
        self.button_MaxN_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decMaxN_callback)

        self.button_jerkSens_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incJerkSens_callback)
        self.button_jerkSens_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decJerkSens_callback)

        self.button_displayIndexRow_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incdisplayIndexRow_callback)
        self.button_displayIndexRow_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decdisplayIndexRow_callback)
        self.button_displayIndexCol_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incdisplayIndexCol_callback)
        self.button_displayIndexCol_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decdisplayIndexCol_callback)

        self.label_interpType = tk.Label(self.AB_widgetsFrame, text = "Interpolation type", width=settingsWidth)
        self.entry_interpType = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_interpType.insert(0, self.interpolationTypes[self.interpTypeNum])
        self.button_interpType_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incInterpType_callback)
        self.button_interpType_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decInterpType_callback)

        self.label_tasks = tk.Label(self.AB_widgetsFrame, text = "Tasks", width=int(settingsWidth * 2))
        self.taskNames = ["panda_reaching", "panda_pushing", "panda_pushing_clutter"]
        self.entry_tasks = AutocompleteEntry(self.AB_widgetsFrame, width=int(settingsWidth * 2), completevalues=self.taskNames)
        self.entry_tasks.insert(0, self.taskNames[0])
        self.button_tasks = tk.Button(self.AB_widgetsFrame, text="Load", command=self.load_callback)

        # ------------------------------------------------------------------------------------------------------------------------

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

        self.label_displayIndexRow.grid(row=0, column=4, columnspan = 3, sticky='EW')

        self.button_displayIndexRow_dec.grid(row=1, column=4)
        self.entry_displayIndexRow.grid(row=1, column=5)
        self.button_displayIndexRow_inc.grid(row=1, column=6)

        self.label_displayIndexCol.grid(row=2, column=4, columnspan = 3, sticky='EW')

        self.button_displayIndexCol_dec.grid(row=3, column=4)
        self.entry_displayIndexCol.grid(row=3, column=5)
        self.button_displayIndexCol_inc.grid(row=3, column=6)

        self.button_evaluate.grid(row=0, column=7,columnspan = 3, sticky='EW')

        self.label_interpType.grid(row=1, column=7, columnspan = 3, sticky='EW')

        self.button_interpType_dec.grid(row=2, column=7)
        self.entry_interpType.grid(row=2, column=8)
        self.button_interpType_inc.grid(row=2, column=9)

        self.label_tasks.grid(row=0, column=10, columnspan = 3, sticky='EW')
        self.entry_tasks.grid(row=1, column=10, columnspan = 3, sticky='EW')
        self.button_tasks.grid(row=2, column=10, columnspan = 3, sticky='EW')

        # ------ state widgets ------
        #label for state type
        self.label_stateType = tk.Label(self.state_widgetFrame, text="state type", width=settingsWidth)
        self.entry_stateType = tk.Entry(self.state_widgetFrame, width=settingsWidth)
        self.entry_stateType.insert(0, self.stateTypes[self.stateDisplayNumber])
        self.button_stateType_inc = tk.Button(self.state_widgetFrame, text="+", command=self.incStateType_callback)
        self.button_stateType_dec = tk.Button(self.state_widgetFrame, text="-", command=self.decStateType_callback)
        
        self.label_dofIndex = tk.Label(self.state_widgetFrame, text="dof row", width=settingsWidth)
        self.entry_dofIndex = tk.Entry(self.state_widgetFrame, width=settingsWidth)
        self.entry_dofIndex.insert(0, str(self.stateDisplayDof))
        self.button_dofIndex_inc = tk.Button(self.state_widgetFrame, text="+", command=self.incDofIndex_callback)
        self.button_dofIndex_dec = tk.Button(self.state_widgetFrame, text="-", command=self.decDofIndex_callback)


        # make a list box with my state types
        #self.listbox_stateType = tk.Listbox(self.state_widgetFrame, width=settingsWidth)
        
        self.label_stateType.grid(row = 0, column = 0, columnspan = 3, sticky='EW')
        self.button_stateType_dec.grid(row = 1, column = 0)
        self.entry_stateType.grid(row = 1, column = 1)
        self.button_stateType_inc.grid(row = 1, column = 2)

        self.label_dofIndex.grid(row = 2, column = 0, columnspan = 3, sticky='EW')
        self.button_dofIndex_dec.grid(row = 3, column = 0)
        self.entry_dofIndex.grid(row = 3, column = 1)
        self.button_dofIndex_inc.grid(row = 3, column = 2)

    # ------ callback functions ------

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

    def incdisplayIndexRow_callback(self):
        val = int(self.entry_displayIndexRow.get())
        val = val + 1
        if val > self.numDOFs - 1:
            val = self.numDOFs - 1
        self.entry_displayIndexRow.delete(0, END)
        self.entry_displayIndexRow.insert(0, val)
        self.updatePlot_derivatives()
    
    def decdisplayIndexRow_callback(self):
        val = int(self.entry_displayIndexRow.get())
        val = val - 1
        if val < 0:
            val = 0
        self.entry_displayIndexRow.delete(0, END)
        self.entry_displayIndexRow.insert(0, val)
        self.updatePlot_derivatives()

    def incdisplayIndexCol_callback(self):
        val = int(self.entry_displayIndexCol.get())
        val = val + 1
        if val > (self.numDOFs * 2) - 1:
            val = (self.numDOFs * 2) - 1
        self.entry_displayIndexCol.delete(0, END)
        self.entry_displayIndexCol.insert(0, val)
        self.updatePlot_derivatives()
    
    def decdisplayIndexCol_callback(self):
        val = int(self.entry_displayIndexCol.get())
        val = val - 1
        if val < 0:
            val = 0
        self.entry_displayIndexCol.delete(0, END)
        self.entry_displayIndexCol.insert(0, val)
        self.updatePlot_derivatives()

    def incInterpType_callback(self):
        self.interpTypeNum = self.interpTypeNum + 1
        if(self.interpTypeNum > len(self.interpolationTypes) - 1):
            self.interpTypeNum = len(self.interpolationTypes) - 1
        self.entry_interpType.delete(0, END)
        self.entry_interpType.insert(0, self.interpolationTypes[self.interpTypeNum])
        self.updatePlot_derivatives()

    def decInterpType_callback(self):
        self.interpTypeNum = self.interpTypeNum - 1
        if(self.interpTypeNum < 0):
            self.interpTypeNum = 0
        self.entry_interpType.delete(0, END)
        self.entry_interpType.insert(0, self.interpolationTypes[self.interpTypeNum])
        self.updatePlot_derivatives()

    def incStateType_callback(self):
        self.stateDisplayNumber = self.stateDisplayNumber + 1
        if(self.stateDisplayNumber > len(self.stateTypes) - 1):
            self.stateDisplayNumber = len(self.stateTypes) - 1
        
        self.entry_stateType.delete(0, END)
        self.entry_stateType.insert(0, self.stateTypes[self.stateDisplayNumber])

        self.updatePlot_trajecInfo()

    def decStateType_callback(self):
        self.stateDisplayNumber = self.stateDisplayNumber - 1
        if(self.stateDisplayNumber < 0):
            self.stateDisplayNumber = 0
        self.entry_stateType.delete(0, END)
        self.entry_stateType.insert(0, self.stateTypes[self.stateDisplayNumber])

        self.updatePlot_trajecInfo()

    def incDofIndex_callback(self):
        self.stateDisplayDof = self.stateDisplayDof + 1
        if(self.stateDisplayDof > self.numDOFs - 1):
            self.stateDisplayDof = self.numDOFs - 1

        self.entry_dofIndex.delete(0, END)
        self.entry_dofIndex.insert(0, self.stateDisplayDof)

        self.updatePlot_trajecInfo()

    def decDofIndex_callback(self):
        self.stateDisplayDof = self.stateDisplayDof - 1
        if(self.stateDisplayDof < 0):
            self.stateDisplayDof = 0

        self.entry_dofIndex.delete(0, END)
        self.entry_dofIndex.insert(0, self.stateDisplayDof)

        self.updatePlot_trajecInfo()
    # -------------------------------------------------

    def displayMode_callback(self):

        self.updatePlot_derivatives()

    def load_callback(self):
        self.task = self.entry_tasks.get()
        self.interpolator = interpolator(self.task)
        self.numDOFs = self.interpolator.dof

        self.dynParams = []
        self.updatePlot_derivatives()
        self.updatePlot_trajecInfo()

    # ------------------ Update right plot - trajectory information ---------------------
    def updatePlot_trajecInfo(self):

        jerkProfile, accelProfile, states, controls = self.interpolator.returnTrajecInformation()

        #extend acceleration profile to match length of states
        accelProfile = np.concatenate((accelProfile, np.zeros((states.shape[0] - accelProfile.shape[0], accelProfile.shape[1]))), axis=0)

        #extend jerk profile to match length of states
        jerkProfile = np.concatenate((jerkProfile, np.zeros((states.shape[0] - jerkProfile.shape[0], jerkProfile.shape[1]))), axis=0)

        self.plot_trajecInfo.clear()
        displayDof = int(self.entry_dofIndex.get())

        displayKeypoints = self.keyPoints[self.interpTypeNum]
        displayKeypoints = displayKeypoints[displayDof]
        print("display key points length: ", len(displayKeypoints))
        highlightedIndices = []
        
        #Position
        if(self.stateDisplayNumber == 0):
            self.plot_trajecInfo.plot(states[:,displayDof], label='Position', color = self.black)
            highlightedIndices = np.copy(states[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof], s=10, color = self.yellow, zorder=10)
        #Velocity
        elif(self.stateDisplayNumber == 1):
            self.plot_trajecInfo.plot(states[:,displayDof+self.numDOFs], label='Velocity', color = self.black)
            highlightedIndices = np.copy(states[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof + self.numDOFs], s=10, color = self.yellow, zorder=10)
        #Acceleration
        elif(self.stateDisplayNumber == 2):
            self.plot_trajecInfo.plot(accelProfile[:,displayDof], label='Acceleration', color = self.black)
            highlightedIndices = np.copy(accelProfile[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof], s=10, color = self.yellow, zorder=10)
        #Jerk
        elif(self.stateDisplayNumber == 3):
            self.plot_trajecInfo.plot(jerkProfile[:,displayDof], label='Jerk', color = self.black)
            highlightedIndices = np.copy(jerkProfile[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof], s=10, color = self.yellow, zorder=10)
        #Control
        elif(self.stateDisplayNumber == 4):
            self.plot_trajecInfo.plot(controls[:,displayDof], label='Control', color = self.black)
            highlightedIndices = np.copy(controls[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof], s=10, color = self.yellow, zorder=10)
        
        self.plot_trajecInfo.legend()
        self.plot_trajecInfo.set_title('trajec Info', fontsize=15, color= self.white, fontweight='bold')
        self.canvas_trajecInfo.draw()

    # ------------------ Update left plot - trajectory derivatives ---------------------
    def updatePlot_derivatives(self):
        dynParams = self.returnDynParams()
        print("dynParams: ", dynParams)
        print("self.dynParams: ", self.dynParams)

        # if dyn params are the same, don't recompute
        if dynParams == self.dynParams:
            pass
        else:
            self.dynParams = dynParams
            self.trueTrajec, self.interpolatedTrajec, self.unfilteredTrajec, self.errors, self.keyPoints = self.interpolator.interpolateTrajectory(0, self.dynParams)

        index = (int(self.entry_displayIndexRow.get()) * self.numDOFs * 2) + int(self.entry_displayIndexCol.get())

        col = int(self.entry_displayIndexCol.get())

        #Remove None values from list
        # displayKeypoints = [x for x in self.keyPoints[self.interpTypeNum] if x is not None]
        displayKeypoints = self.keyPoints[self.interpTypeNum]
        displayKeypoints = displayKeypoints[col % self.numDOFs]
        highlightedIndices = np.copy(self.unfilteredTrajec[displayKeypoints, ])

        self.numEvals = len(displayKeypoints)

        self.plot_AB.clear()

        self.plot_AB.plot(self.trueTrajec[:,index], color = self.black, label='Ground truth')
        self.plot_AB.plot(self.unfilteredTrajec[:,index], color = 'orange', label='Unfiltered')

        # Plot keypoints
        self.plot_AB.scatter(displayKeypoints, highlightedIndices[:, index], s=10, color = self.yellow, zorder=10)

        self.plot_AB.plot(self.interpolatedTrajec[self.interpTypeNum,:,index], color = self.yellow, label = 'Interpolated')
        # self.plot_AB.legend(loc='upper right')
        self.plot_AB.set_title('A matrix val over trajectory', fontsize=15, color= self.white, fontweight='bold')

        evalsString = "Evals: " + str(self.numEvals)

        at = AnchoredText(evalsString,
                    prop=dict(size=15), frameon=True, loc='lower right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        self.plot_AB.add_artist(at)


        at2 = AnchoredText("Error: " + str(round(self.errors[self.interpTypeNum], 2)) + "",
                       loc='lower left', prop=dict(size=8), frameon=True,
                       bbox_to_anchor=(0., 1.05),
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
    



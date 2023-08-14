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
        self.master.geometry("1400x900")
        self.master.resizable(True, True)
        self.master.title('Interactive Dynamics')
        
        self.interpolationTypes = ["setInterval", "adaptiveAccel", "adaptiveJerk", "iterativeError", "magVelChange"]
        self.interpTypeNum = 2
        self.stateTypes = ["Position", "Velocity", "Acceleration", "Jerk", "Control"]
        self.stateDisplayNumber = 1
        self.stateDisplayDof = 1

        self.taskNames = ["doublePendulum", "acrobot", "panda_reaching", "panda_pushing", "panda_pushing_low_clutter", "panda_pushing_heavy_clutter", "kinova_forward", "kinova_side", "kinova_lift"]
        self.startingDynParams = [[5, 50, 0.1, 0.1, 0.000007],
                                     [10, 200, 0.005, 0.005, 0.004], 
                                     [10, 200, 0.005, 0.005, 0.005],
                                     [10, 200, 0.005, 0.005, 0.005], 
                                     [10, 200, 0.005, 0.005, 0.005], 
                                     [10, 200, 0.005, 0.005, 0.005],
                                     [10, 200, 0.005, 0.005, 0.005],
                                     [2, 10, 0.0005, 0.0005, 0.0005],
                                     [10, 200, 0.005, 0.005, 0.005]]

        # dictionary of starting dyn params
        self.startingDynParamsDict = {}
        for i in range(len(self.taskNames)):
            self.startingDynParamsDict[self.taskNames[i]] = self.startingDynParams[i]


        self.darkBlue = '#103755'
        self.lightBlue = '#90CBCB'
        self.teal = '#59CBCB'
        self.yellow = '#EEF30D'
        self.white = '#FFFFFF'
        self.black = '#000000'

        self.numEvals = 0
        self.dynParams = []
        self.trajectoryNumber = 0
        self.dof_pos = 0
        self.dof_vel = 0
        self.num_ctrl = 0

        self.setupGUI()
        self.load_callback()

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

        # toolbar = NavigationToolbar2Tk(self.canvas_trajecInfo,
        #                             self.master)
        # toolbar.grid(row=1, column=0)
        # toolbar.update()

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
        self.entry_jerkSensitivity.insert(0, "0.005") 
        self.label_displayIndexRow = tk.Label(self.AB_widgetsFrame, text="displayIndexRow", width=settingsWidth)
        self.entry_displayIndexRow = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_displayIndexRow.insert(0, "0")
        self.label_displayIndexCol = tk.Label(self.AB_widgetsFrame, text="displayIndexCol", width=settingsWidth)
        self.entry_displayIndexCol = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_displayIndexCol.insert(0, "0")
        self.label_acellSensitivity = tk.Label(self.AB_widgetsFrame, text="acellSensitivity", width=settingsWidth)
        self.entry_acellSensitivity = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_acellSensitivity.insert(0, "0.005")
        self.label_iterativeErrorThreshold = tk.Label(self.AB_widgetsFrame, text="iterativeErrorThreshold", width=settingsWidth)
        self.entry_iterativeErrorThreshold = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_iterativeErrorThreshold.insert(0, "0.003")
        self.label_velChangeSensitivity = tk.Label(self.AB_widgetsFrame, text="velChangeSensitivity", width=settingsWidth)
        self.entry_velChangeSensitivity = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_velChangeSensitivity.insert(0, "0.1")
        self.button_evaluate = tk.Button(self.AB_widgetsFrame, text="Evaluate", command=self.displayMode_callback)

        self.button_MinN_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incMinN_callback)
        self.button_MinN_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decMinN_callback)

        self.button_MaxN_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incMaxN_callback)
        self.button_MaxN_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decMaxN_callback)

        self.button_jerkSens_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incJerkSens_callback)
        self.button_jerkSens_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decJerkSens_callback)

        self.button_acellSens_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incAcellSens_callback)
        self.button_acellSens_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decAcellSens_callback)

        self.button_iterativeErrorThresh_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incIterativeError_callback)
        self.button_iterativeErrorThresh_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decIterativeError_callback)

        self.button_velChangeSens_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incVelChangeSens_callback)
        self.button_velChangeSens_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decVelChangeSens_callback)

        self.button_displayIndexRow_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incdisplayIndexRow_callback)
        self.button_displayIndexRow_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decdisplayIndexRow_callback)
        self.button_displayIndexCol_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incdisplayIndexCol_callback)
        self.button_displayIndexCol_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decdisplayIndexCol_callback)

        self.label_interpType = tk.Label(self.AB_widgetsFrame, text = "Interpolation type", width=settingsWidth)
        self.entry_interpType = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_interpType.insert(0, self.interpolationTypes[self.interpTypeNum])
        self.button_interpType_inc = tk.Button(self.AB_widgetsFrame, text="+", command=self.incInterpType_callback)
        self.button_interpType_dec = tk.Button(self.AB_widgetsFrame, text="-", command=self.decInterpType_callback)

        self.label_tasks = tk.Label(self.AB_widgetsFrame, text = "Task Name", width=int(settingsWidth * 2))
        self.entry_tasks = AutocompleteEntry(self.AB_widgetsFrame, width=int(settingsWidth * 2), completevalues=self.taskNames)
        self.entry_tasks.insert(0, self.taskNames[7]) # 7 kinova side
        self.label_trajecNum = tk.Label(self.AB_widgetsFrame, text = "Trajectory Number", width=settingsWidth)
        self.entry_trajecNum = tk.Entry(self.AB_widgetsFrame, width=settingsWidth)
        self.entry_trajecNum.insert(1, "1")
        self.button_tasks = tk.Button(self.AB_widgetsFrame, text="Load", command=self.load_callback)

        self.showFilter = 0
        self.check_filterShow = tk.Checkbutton(self.AB_widgetsFrame, text='Show filtered value', command=self.check_filterShow_callback)

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

        self.label_acellSensitivity.grid(row=6, columnspan = 3, sticky='EW')
        self.button_acellSens_dec.grid(row=7, column=0)
        self.entry_acellSensitivity.grid(row=7, column=1)
        self.button_acellSens_inc.grid(row=7, column=2)

        self.label_iterativeErrorThreshold.grid(row=8, columnspan = 3, sticky='EW')
        self.button_iterativeErrorThresh_dec.grid(row=9, column=0)
        self.entry_iterativeErrorThreshold.grid(row=9, column=1)
        self.button_iterativeErrorThresh_inc.grid(row=9, column=2)

        self.label_velChangeSensitivity.grid(row=10, columnspan = 3, sticky='EW')
        self.button_velChangeSens_dec.grid(row=11, column=0)
        self.entry_velChangeSensitivity.grid(row=11, column=1)
        self.button_velChangeSens_inc.grid(row=11, column=2)

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
        self.label_trajecNum.grid(row=2, column=10, columnspan = 3, sticky='EW')
        self.entry_trajecNum.grid(row=3, column=10, columnspan = 3, sticky='EW')
        self.button_tasks.grid(row=4, column=10, columnspan = 3, sticky='EW')
        self.check_filterShow.grid(row=5, column=10, columnspan = 3, sticky='EW')

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

    def check_filterShow_callback(self):
        self.showFilter = 1 - self.showFilter
        self.updatePlot_derivatives()

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

    def incAcellSens_callback(self):
        val = float(self.entry_acellSensitivity.get())
        self.entry_acellSensitivity.delete(0, END)
        self.entry_acellSensitivity.insert(0, str(val+0.0001))
    
    def decAcellSens_callback(self):
        val = float(self.entry_acellSensitivity.get())
        self.entry_acellSensitivity.delete(0, END)
        self.entry_acellSensitivity.insert(0, str(val-0.0001))

    def incIterativeError_callback(self):
        val = float(self.entry_iterativeErrorThreshold.get())
        self.entry_iterativeErrorThreshold.delete(0, END)
        self.entry_iterativeErrorThreshold.insert(0, str(val+0.0001))
    
    def decIterativeError_callback(self):
        val = float(self.entry_iterativeErrorThreshold.get())
        self.entry_iterativeErrorThreshold.delete(0, END)
        self.entry_iterativeErrorThreshold.insert(0, str(val-0.0001))

    def incVelChangeSens_callback(self):
        val = float(self.entry_velChangeSensitivity.get())
        self.entry_velChangeSensitivity.delete(0, END)
        self.entry_velChangeSensitivity.insert(0, str(val+0.1))

    def decVelChangeSens_callback(self):
        val = float(self.entry_velChangeSensitivity.get())
        self.entry_velChangeSensitivity.delete(0, END)
        self.entry_velChangeSensitivity.insert(0, str(val-0.1))

    def incdisplayIndexRow_callback(self):
        val = int(self.entry_displayIndexRow.get())
        val = val + 1
        if val > self.dof_vel - 1:
            val = self.dof_vel - 1
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
        if val > (self.dof_pos + self.dof_vel) - 1:
            val = (self.dof_pos + self.dof_vel) - 1
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

        # positions
        if(self.stateDisplayNumber == 0):
            if(self.stateDisplayDof > self.dof_pos - 1):
                self.stateDisplayDof = self.dof_pos - 1

        # velocities, accelerations, jerks
        elif(self.stateDisplayNumber >= 1 and self.stateDisplayNumber <= 3):
            if(self.stateDisplayDof > self.dof_vel - 1):
                self.stateDisplayDof = self.dof_vel - 1

        # controls
        elif(self.stateDisplayNumber == 4):
            if(self.stateDisplayNumber > self.num_ctrl):
                self.stateDisplayDof = self.num_ctrl

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
        self.trajectoryNumber = int(self.entry_trajecNum.get())
        self.interpolator = interpolator(self.task, self.trajectoryNumber)
        self.dof_pos = self.interpolator.dof_pos
        self.dof_vel = self.interpolator.dof_vel
        self.num_ctrl = self.interpolator.num_ctrl

        defaultDynParamsForTask = self.startingDynParamsDict[self.task]

        self.dynParams = []
        self.entry_minN.delete(0, END)  
        self.entry_minN.insert(0, str(defaultDynParamsForTask[0]))
        self.entry_maxN.delete(0, END)
        self.entry_maxN.insert(0, str(defaultDynParamsForTask[1]))
        self.entry_jerkSensitivity.delete(0, END)
        self.entry_jerkSensitivity.insert(0, float(defaultDynParamsForTask[2]))
        self.entry_acellSensitivity.delete(0, END)
        self.entry_acellSensitivity.insert(0, float(defaultDynParamsForTask[3]))
        self.entry_iterativeErrorThreshold.delete(0, END)
        self.entry_iterativeErrorThreshold.insert(0, float(defaultDynParamsForTask[4]))
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

        highlightedIndices = []
        
        #Position
        if(self.stateDisplayNumber == 0):
            self.plot_trajecInfo.plot(states[:,displayDof], label='Position', color = self.black)
            displayKeypoints = self.keyPoints[self.interpTypeNum]
            displayKeypoints = displayKeypoints[displayDof]
            highlightedIndices = np.copy(states[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof], s=10, color = self.yellow, zorder=10)
        #Velocity
        elif(self.stateDisplayNumber == 1):
            self.plot_trajecInfo.plot(states[:,displayDof+self.dof_pos], label='Velocity', color = self.black)
            displayKeypoints = self.keyPoints[4]
            displayKeypoints = displayKeypoints[displayDof]
            highlightedIndices = np.copy(states[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof + self.dof_pos], s=10, color = self.yellow, zorder=10)
        #Acceleration
        elif(self.stateDisplayNumber == 2):
            self.plot_trajecInfo.plot(accelProfile[:,displayDof], label='Acceleration', color = self.black)
            displayKeypoints = self.keyPoints[1]
            displayKeypoints = displayKeypoints[displayDof]
            highlightedIndices = np.copy(accelProfile[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof], s=10, color = self.yellow, zorder=10)
            # draw a horizontal line at y = jerk threshold
            self.plot_trajecInfo.axhline(y=self.dynParams[3].acellThreshold, color=self.yellow, linestyle='--')
            self.plot_trajecInfo.axhline(y=-self.dynParams[3].acellThreshold, color=self.yellow, linestyle='--')
            # set y limits to be the same as jerk
            # self.plot_trajecInfo.set_ylim([-self.dynParams[2] * 2, self.dynParams[2] * 2])
        #Jerk
        elif(self.stateDisplayNumber == 3):
            self.plot_trajecInfo.plot(jerkProfile[:,displayDof], label='Jerk', color = self.black)
            displayKeypoints = self.keyPoints[2]
            displayKeypoints = displayKeypoints[displayDof]
            highlightedIndices = np.copy(jerkProfile[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof], s=10, color = self.yellow, zorder=10)
            # draw a horizontal line at y = jerk threshold
            self.plot_trajecInfo.axhline(y=self.dynParams[2].jerkThreshold, color=self.yellow, linestyle='--')
            self.plot_trajecInfo.axhline(y=-self.dynParams[2].jerkThreshold, color=self.yellow, linestyle='--')
            # set y limits to be the same as jerk
            # self.plot_trajecInfo.set_ylim([-self.dynParams[2] * 2, self.dynParams[2] * 2])
        #Control
        elif(self.stateDisplayNumber == 4):
            self.plot_trajecInfo.plot(controls[:,displayDof], label='Control', color = self.black)
            displayKeypoints = self.keyPoints[self.interpTypeNum]
            displayKeypoints = displayKeypoints[displayDof]
            highlightedIndices = np.copy(controls[displayKeypoints, ])
            self.plot_trajecInfo.scatter(displayKeypoints, highlightedIndices[:, displayDof], s=10, color = self.yellow, zorder=10)
        
        self.plot_trajecInfo.legend()
        self.plot_trajecInfo.set_title('trajec Info', fontsize=15, color= self.white, fontweight='bold')
        self.canvas_trajecInfo.draw()

    # ------------------ Update left plot - trajectory derivatives ---------------------
    def updatePlot_derivatives(self):
        dynParams = self.returnDynParams()

        # if dyn params are the same, don't recompute
        if dynParams == self.dynParams:
            pass
        else:
            self.dynParams = dynParams
            self.trueTrajec, self.interpolatedTrajec, self.unfilteredTrajec, self.errors, self.keyPoints, self.key_points_w = self.interpolator.interpolateTrajectory(0, self.dynParams)

        index = (int(self.entry_displayIndexRow.get()) * (self.dof_pos + self.dof_vel)) + int(self.entry_displayIndexCol.get())

        row = int(self.entry_displayIndexRow.get())
        col = int(self.entry_displayIndexCol.get())

        # get the column
        # check it against size of dof vel
        #TODO - fix this for multiple quaternions
        if(len(self.key_points_w) and col == self.dof_pos - 1):
            displayKeypoints = self.key_points_w
            highlightedIndices = np.copy(self.unfilteredTrajec[displayKeypoints, row, col])
        else:
            displayKeypoints = self.keyPoints[self.interpTypeNum]
            # 0 -> dof_pos - 1, dof_pos -> dof_pos + dof_vel - 1
            if(col >= self.dof_pos):
                col = col - self.dof_pos
            # displayKeypoints = displayKeypoints[col % self.dof_vel]
            displayKeypoints = displayKeypoints[col]
            highlightedIndices = np.copy(self.unfilteredTrajec[displayKeypoints, row, col])

        self.numEvals = len(displayKeypoints)

        self.plot_AB.clear()

        if(self.showFilter):
            print("showing filtered trajectory")
            self.plot_AB.plot(self.trueTrajec[:, row, col], color = 'orange', label='Ground truth')

        self.plot_AB.plot(self.unfilteredTrajec[:, row, col], color = self.black, label='Unfiltered')

        # Plot keypoints
        self.plot_AB.scatter(displayKeypoints, highlightedIndices, s=10, color = self.yellow, zorder=10)

        self.plot_AB.plot(self.interpolatedTrajec[self.interpTypeNum,:, row, col], color = self.yellow, label = 'Interpolated')
        self.plot_AB.legend(loc='upper right')
        self.plot_AB.set_title('A matrix val over trajectory', fontsize=15, color= self.white, fontweight='bold')

        # set y lims
        minVal = np.min(self.unfilteredTrajec[:,row, col])
        maxVal = np.max(self.unfilteredTrajec[:,row, col])

        if (maxVal - minVal < 0.1):
            self.plot_AB.set_ylim([minVal - 0.05, maxVal + 0.05])
        

        evalsString = "Evals: " + str(self.numEvals)

        # Anchor text above plot - offset from plot by 10%
        at = AnchoredText(evalsString, loc='upper left', prop=dict(size=8), frameon=True, bbox_to_anchor=(0.9, 1.1), bbox_transform=self.plot_AB.transAxes)
        
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
        acellSensitivity = float(self.entry_acellSensitivity.get())
        iterativeErrorThreshold = float(self.entry_iterativeErrorThreshold.get())
        velChangeSensitivity = float(self.entry_velChangeSensitivity.get())

        dynParams = [None] * len(self.interpolationTypes)
        for i in range(len(self.interpolationTypes)):
            dynParams[i] = derivative_interpolator(self.interpolationTypes[i], minN, maxN, acellSensitivity, jerkSensitivity, iterativeErrorThreshold, 2)

        return dynParams
    
if __name__ == "__main__":
    root = Tk()
    myGUI = dynamicsGUI(root)
    root.mainloop()
    



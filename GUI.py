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
        self.stateTypes = ["Position", "Velocity", "Acceleration", "Jerk", "Control"]
        self.stateDisplayNumber = 0
        self.stateDisplayDof = 0

        self.darkBlue = '#103755'
        self.lightBlue = '#90CBCB'
        self.teal = '#59CBCB'
        self.yellow = '#EEF30D'
        self.white = '#FFFFFF'
        self.black = '#000000'

        self.numEvals = 0

        self.setupGUI()

        self.taskNumber = 1

        self.interpolator = interpolator(0, self.taskNumber)
        self.numDOFs = self.interpolator.dof
        self.dynParams = []
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

        # # creating the Matplotlib toolbar
        # toolbar = NavigationToolbar2Tk(self.canvas,
        #                             self.master)
        # toolbar.update()
    
        # # placing the toolbar on the Tkinter window
        # self.canvas.get_tk_widget().pack(side=LEFT)

        # frame = Frame(self.master, bg='#f25252')
        # frame.pack(expand=True)

    def setupButtons(self):
        settingsWidth = 10

        # ------ A/B matrix widgets ------
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

        # ------ state widgets ------
        #label for state type
        self.label_stateType = tk.Label(self.state_widgetFrame, text="state type", width=settingsWidth)
        self.entry_stateType = tk.Entry(self.state_widgetFrame, width=settingsWidth)
        self.entry_stateType.insert(0, self.stateTypes[self.stateDisplayNumber])
        self.button_stateType_inc = tk.Button(self.state_widgetFrame, text="+", command=self.incStateType_callback)
        self.button_stateType_dec = tk.Button(self.state_widgetFrame, text="-", command=self.decStateType_callback)
        
        self.label_dofIndex = tk.Label(self.state_widgetFrame, text="dof index", width=settingsWidth)
        self.entry_dofIndex = tk.Entry(self.state_widgetFrame, width=settingsWidth)
        self.entry_dofIndex.insert(0, "0")
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

    def displayMode_callback(self):

        self.updatePlot()

    def updatePlot_trajecInfo(self):

        jerkProfile, accelProfile, states, controls = self.interpolator.returnTrajecInformation()

        self.plot_trajecInfo.clear()
        displayDof = int(self.entry_dofIndex.get())

        #Position
        if(self.stateDisplayNumber == 0):
            self.plot_trajecInfo.plot(states[:,displayDof], label='Position', color = self.black)
        #Velocity
        elif(self.stateDisplayNumber == 1):
            self.plot_trajecInfo.plot(states[:,displayDof+self.numDOFs], label='Velocity', color = self.black)
        #Acceleration
        elif(self.stateDisplayNumber == 2):
            self.plot_trajecInfo.plot(accelProfile[:,displayDof+self.numDOFs], label='Acceleration', color = self.black)
        #Jerk
        elif(self.stateDisplayNumber == 3):
            self.plot_trajecInfo.plot(jerkProfile[:,displayDof+self.numDOFs], label='Jerk', color = self.black)
        #Control
        elif(self.stateDisplayNumber == 4):
            self.plot_trajecInfo.plot(controls[:,displayDof], label='Control', color = self.black)
        
        self.plot_trajecInfo.legend()
        self.plot_trajecInfo.set_title('trajec Info', fontsize=15, color= self.white, fontweight='bold')
        self.canvas_trajecInfo.draw()

    def updatePlot(self):
        dynParams = self.returnDynParams()

        # if dyn params are the same, don't recompute
        if dynParams == self.dynParams:
            pass
        else:
            self.dynParams = dynParams
            self.trueTrajec, self.interpolatedTrajec, self.unfilteredTrajec, self.errors, self.reEvaluationIndices, self.iterativeKeyPoints = self.interpolator.interpolateTrajectory(0, self.dynParams)

        index = int(self.entry_displayIndex.get())

        highlightedIndices = np.copy(self.unfilteredTrajec[self.reEvaluationIndices, ])
        highlightedIndicesIterative = np.copy(self.unfilteredTrajec[self.iterativeKeyPoints, ])
        self.numEvals = len(self.reEvaluationIndices)

        self.plot_AB.clear()

        self.plot_AB.plot(self.trueTrajec[:,index], color = self.black, label='Ground truth')
        self.plot_AB.plot(self.unfilteredTrajec[:,index], color = 'orange', label='Unfiltered')

        if(self.interpTypeNum == 1):
            self.plot_AB.scatter(self.iterativeKeyPoints, highlightedIndicesIterative[:, index], s=10, color = self.yellow, zorder=10)
        else:
            self.plot_AB.scatter(self.reEvaluationIndices, highlightedIndices[:, index], s=10, color = self.yellow, zorder=10)

        self.plot_AB.plot(self.interpolatedTrajec[self.interpTypeNum,:,index], color = self.yellow, label = 'Interpolated')
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
    



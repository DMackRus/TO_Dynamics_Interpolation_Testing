from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from GUI import *

trajecLength = 3000

def main():

    root = Tk()
    myGUI = dynamicsGUI(root)

    mainloop()

    pass

if __name__=="__main__":
    main()
# Programmer: Jayde Spiegel (spiegel.85@buckeyemail.osu.edu)
# File: analysis.py
# Latest rev: 18-Mar-2025
# Desc: Automatic analysis of Milikan Oil drops 

import numpy as np
import glob
from where_linear import LinearDomainFinder

class analysis:

    def __init__(self):
        self.directory = ""

    def setDirectory(self, dire: str):
        """
        """
        self.directory = dire
    
    def Analyze(self):
        
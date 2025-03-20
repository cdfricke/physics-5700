# Programmer: Jayde Spiegel (spiegel.85@buckeyemail.osu.edu)
# File: analysis.py
# Latest rev: 18-Mar-2025
# Desc: Automatic analysis of Milikan Oil drops 

import numpy as np
import glob
from where_linear import LinearDomainFinder
import pandas as pd

def add_in_quad(data: np.ndarray) -> float:
        sumsqr = 0.0
        for val in data:
            sumsqr += val*val
        return np.sqrt(sumsqr)
def q(ve,vg):
        d = 4.56e-3 # m
        V = 400 # V
        eta = 18.13e-6 #N*s/m^2
        b = 6.17e-6 # m*cmHg
        P = 76 #cmHg
        rho = 0.86e3 #kg/m^3
        g = 9.801 #m/s^2
        a = np.sqrt((9*eta*vg)/(2*g*rho))
        return [a,((6*np.pi*d/V)*np.sqrt((9*eta**3)/(2*rho*g))*(ve+vg)*(np.sqrt(vg))*(1+(b/(a*P)))**(-3/2))]

class analysis:

    def __init__(self):
        self.files = []
        self.Q = []
        self.A = []
        self.VG = []
        self.VE = []
        self.VE_err = []
        self.VG_err = []

    def Analyze(self,PATH:str,verb:int,win_size,cut):
        """
        """
        self.files.append(PATH[-11:])
        LDF = LinearDomainFinder()
        LDF.setMethod(1)
        LDF.setVerbosity(verb)
        df = pd.read_csv(PATH)
        LDF.setX(df["t"].to_numpy(),label="Time")
        LDF.setY(df["y"].to_numpy(),label="Y Position")

        LDF.slidingWindowFind(WIN_SIZE=win_size,FDEV_CUT=cut)
        if verb > 0:
            print(LDF._slopes)
            print(LDF._slope_errors)
            userInput = input("Press ENTER to continue.")

        domainSelection = range(1,len(LDF._slopes)-1)
        vg_data = []
        vg_err = []
        ve_data = []
        ve_err = []
        for i in domainSelection:
            slope = LDF._slopes[i]
            err = LDF._slope_errors[i]
            if slope > 0.0:
                vg_data.append(slope)
                vg_err.append(err)
            elif slope < 0.0:
                ve_data.append(abs(slope))
                ve_err.append(err)

        V_g = np.mean(vg_data)
        V_g_err = add_in_quad(vg_err)
        V_E = np.mean(ve_data)
        V_E_err = add_in_quad(ve_err)
        if verb > 0:
            print("V_g =", V_g, "+-", V_g_err, "(mm/s)")
            print("V_E =", V_E, "+-", V_E_err, "(mm/s)")

        radius, totalCharge = q(V_E*1.0E-3, V_g*1.0E-3)
        if verb > 0:
            print("a =", radius)
            print("q =", totalCharge)

        numElectrons = totalCharge / 1.602E-19
        if verb > 0:
            print("# of Electrons =", numElectrons)
        
        self.VE.append(V_E)
        self.VE_err.append(V_E_err)
        self.VG.append(V_g)
        self.VG_err.append(V_g_err)


    def export(self, name:str):
        data = pd.DataFrame()
        data["q"] = self.Q
        data["a"] = self.A
        data["VG"] = self.VG
        data["VE"] = self.VE
        data["VE_err"] = self.VE_err
        data["VG_err"] = self.VG_err
        data["Particle"] = self.files
        data.to_csv(name)
        

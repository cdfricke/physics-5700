import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

df = pd.read_csv('prelim-3-data.csv')

print(df.head())

def V_r(R, Z, V_0):
    return ((R-Z)/(R+Z)) * V_0

def dV_r(R, Z, V_0):
    term1 = V_0 / (R - Z)
    term2 = V_0 * (R + Z) / ((R - Z)**2)
    return term1-term2

xdata = df['R'].to_numpy()
ydata = df['V'].to_numpy()
xerr = df['R_err'].to_numpy()
yerr = df['V_err'].to_numpy()

popt, pcov = curve_fit(V_r, xdata, ydata)
sigmas = np.sqrt( yerr ** 2 + (xerr**2 * dV_r(xdata, popt[0], popt[1])**2 ) )
popt, pcov = curve_fit(V_r, xdata, ydata, sigma=sigmas)
print(popt)

xdata_fit = np.logspace(-1, 4.0, 100)
ydata_fit = V_r(xdata_fit, popt[0], popt[1])

fig, ax = plt.subplots()

plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, ls='none', c='k', capsize=1)
plt.plot(xdata_fit, ydata_fit)
plt.title('Reflection Amplitude vs. Resistance')
plt.xlabel('Resistance (Ohms)')
plt.ylabel('Reflection Amplitude (V)')
plt.xscale('log')
plt.show()
# Preliminary Lab 2 Physics 5700
# Connor Fricke
# fricke.59@osu.edu

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def y(x, m, b):
    return m*x + b

# store data
dist = np.array([6.756, 12.903, 19.05, 20.269, 26.416])
dist_err = np.array([.003, .006, .009, .013, .016])
time = np.array([28.6, 56.84, 84.92, 91.04, 119.08])
time_err = np.array([0.07, 0.22, 0.11, 0.33, 0.23])

# run initial fit
init_fit = np.polyfit(dist, time, deg=1)
m = init_fit[0]
b = init_fit[1]
fit_y = y(x=dist, m=m, b=b)

# calculate true yerr for weighted fit based off of slope and error propagation of xerr to yerr
s_tot = np.sqrt(time_err**2 + (m**2)*(dist_err**2))
linfit, lincov = curve_fit(y, dist, time, sigma=s_tot)
m = linfit[0]
b = linfit[1]
final_fit_y = y(dist, m=m, b=b)

N=5
sx=np.sum(dist)
sy=np.sum(time)
sxy=np.sum(dist*time)
sxx=np.sum(dist**2)
delta=N*sxx-sx*sx

m=(N*sxy-sx*sy)/delta
b=(sxx*sy-sx*sxy)/delta

sigma_m=N/delta
sigma_b=sxx/delta
print(m, np.sqrt(sigma_m), b, np.sqrt(sigma_b))

chi_sq = np.sum((time - final_fit_y)**2 / (s_tot**2))
print("chi squared:", chi_sq)
print("reduced chi squared:", chi_sq / 3.0)
print()

# generate graph
fig, ax = plt.subplots()
fig = plt.errorbar(x=dist, y=time, xerr=dist_err, yerr=time_err, fmt='.k', capsize=2, label='Measurements')
fig = plt.plot(dist, final_fit_y, label=f'Fit: y={round(m,3)}x + {round(b,3)}')
plt.legend()
plt.title('Reflected Pulses: Time Delay vs. Travel Distance')
plt.ylabel('t (ns)')
plt.xlabel('dist (m)')
plt.grid()
plt.show()





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def n(N, theta, t, wl):
    return (2.0*t - (N*wl))*(1.0 - np.cos(theta))/((2.0*t*(1 - np.cos(theta))) - (N*wl))

df = pd.read_csv("C:\\users\\cdfri\\dev\\physics-5700\\data\\refractive_index_data.csv")
df['theta'] = -(df['marking'] - 4.5) * (2.0*np.pi / 50.0)
df['N_avg'] = df['N1'] + df['N2'] + df['N3'] / 3.0
df['n'] = n(df['N_avg'].to_numpy(), df['theta'].to_numpy(), t=0.001, wl=632.8e-9)
df = df.iloc[0:12]
print(df.head(15))

# Given parameters
t = 1.0  # Thickness of the glass plate (arbitrary units)
lambda_ = 0.0005  # Laser wavelength (arbitrary units)



# Sample measured data (replace with actual measurements)
theta_values = df['theta'].to_numpy()  # Angles from -pi/4 to pi/4
N_values =  df['N_avg'].to_numpy() # Example number of fringes counted

# Compute the index of refraction for each measurement
n_values = ( (2*t - N_values*lambda_) * (1 - np.cos(theta_values)) ) / ( 2*t*(1 - np.cos(theta_values)) - N_values*lambda_ )

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(theta_values, n_values, 'bo-', label='Measured n')
plt.xlabel("Rotation Angle (radians)")
plt.ylabel("Index of Refraction")
plt.title("Interferometric Measurement of Index of Refraction")
plt.legend()
plt.grid(True)
plt.show()
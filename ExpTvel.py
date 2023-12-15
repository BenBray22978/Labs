import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

num = 3000
particles = 473

column_names = ['1', '2', '3', '4', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', '8', '9', '10', '11']
dflist = []

for i in range(1, num + 1):
    format_i = str(i).zfill(5)
    filename = f'aftermath6.{format_i}.csv'
    df = pd.read_csv(filename, header=None, names=column_names, sep=' ')
    dflist.append(df)

comx = []
comy = []
comz = []

for df in dflist:
    CoMx = df['Vx'].sum() / particles
    CoMy = df['Vy'].sum() / particles
    CoMz = df['Vz'].sum() / particles
    comx.append(CoMx)
    comy.append(CoMy)
    comz.append(CoMz)

count = 0
rsum = np.zeros(3000, dtype=float)

for df in dflist:
    rlist = np.zeros(particles, dtype=float)
    for x in range(0, particles):
        r = np.sqrt((df['Vx'][x] - comx[count]) ** 2 + (df['Vy'][x] - comy[count]) ** 2 + (
                df['Vz'][x] - comz[count]) ** 2)
        rlist[x] = r

    rsum[count] = np.sum(rlist) / particles
    count += 1

xaxis = np.arange(1, 3001)
yaxis = rsum

plt.plot(xaxis, yaxis)
plt.xlabel('time step (5E-6 years/2pi)')
plt.ylabel('Average radial velocity from CoM (AU/s)')
plt.show()

xaxis2 = np.arange(1, 3001)
grad = np.gradient(rsum)
max_index = np.where(grad == grad.max())
plt.plot(xaxis2, grad)
plt.xlabel('time step (5E-6 years/2pi)')
plt.ylabel('Average radial acceleration from CoM (AU/s/s)')
plt.show()

window_size = 100
avgrad = np.convolve(grad, np.ones(window_size) / window_size, mode='same')  # Use 'same' mode

plt.plot(xaxis2, avgrad)
plt.xlabel('time step (5E-6 years/2pi)')
plt.ylabel('Smoothed Average radial acceleration from CoM (AU/s/s)')
plt.show()

threshold = 0.5e-8  # Add your desired threshold value here

# Define the region of interest
start_index = 300
end_index = 800

# Find the index where the smoothed graph crosses the threshold within the specified region
crossing_indices = np.where((avgrad[start_index:end_index] >= threshold))[0]

# If no crossing is found, print a message
if len(crossing_indices) == 0:
    print(f"No threshold crossing found within the specified region ({start_index} to {end_index})")
else:
    # Get the first crossing index within the region
    first_crossing_index = crossing_indices[0] + start_index
    print(f'Threshold Crossing Index: {first_crossing_index}')
    print(f'Threshold Crossing Time: {xaxis2[first_crossing_index]}')
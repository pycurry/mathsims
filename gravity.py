# Setting the simulation parameters

planetmass = 15e16              # Mass of the planet (kg) mass of earth=5.972e24 original = 15e15
# objectmass = 10               # Mass of the satellite (kg)
p0 = [200, 50]                    # Initial position of the satellite (X-meters, Y-meters) [0, 50]
v0 = 200                        # Initial veloity of the satellite (m/s) 100
theta0 = 180                    # Direction of velocity of satellite with the +ve X-axis (degrees)180
gconst = 6.674e-11              # Gravitational constant (N.m**2/kg**2)
time = 10                       # Total time period of simulation (seconds)
dt = 0.01                       # Temporal resolution for integration function (seconds)
axis_limits = 200

# Importing required libraries

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Initializing the simulation parameters

theta0 = np.radians(theta0)
vx0, vy0 = v0*np.cos(theta0), v0*np.sin(theta0) 
vector0 = np.array([p0[0], vx0, p0[1], vy0], dtype=np.int64)      # initial state vector of the satellite [x-position , X-velocity, Y-position, Y-velocity]

# Defining the gravitional function, a second order differentil equation that returns dp(change in position) a vector

def gravity(vector, t, gconst, planetmass):

    xt, vxt, yt, vyt = vector          # vector = [x-position , X-velocity, Y-position, Y-velocity] at time t
    theta = np.arctan2(yt, xt)
    r = np.sum(np.square(xt) + np.square(yt))

    dp = [vxt, vxt-(((gconst*planetmass)/r)*np.cos(theta)),
          vyt, vyt-(((gconst*planetmass)/r)*np.sin(theta))]

    return dp

t = np.arange(0.0, time, dt)         # defining the temporal list

# Solving the differential equation with scipy.integrate.odeint whose back end is FORTRAN odepack

sol = odeint(gravity, vector0, t, args=(gconst, planetmass))   # storing the solution to a variable 'sol'

# Plotting the solution with matplotlib

fig, ax = plt.subplots()                                      # initializing the plot window
ln1, = ax.plot([], [])                                        # empty array to plot the trail of the satellite
ln2, = ax.plot([], [], 'o-')                                  # empty array to plot the real position of satellite
ln3, = ax.plot([0], [0], 'o-', markersize=10)                 # plotting the marker of planet at origin
ax.grid()                                                     # initializing the grid lines in the plot
ax.set_aspect('equal')                                        # plot aspect ratio as equal (square grid)
ax.set_xlim(-axis_limits, axis_limits)                        # limits on x-axis
ax.set_ylim(-axis_limits, axis_limits)                        # limits on y-axis
ax.set_title('Gravity')                                       # plot title 
time_template = 'time = %.2fs'                                # template for time text
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)    # position for the time text
xdata, ydata = [], []                                         # empty data list for the trailing line              

# Defining the function for the animation of the plot

def update(i):
    xdata.append(sol[i, 0])                        # appending x-data for the trail line
    ydata.append(sol[i, 2])                        # appending y-data for trail line
    ln1.set_data(xdata, ydata)                     # updating data for trail line
    ln2.set_data(sol[i, 0], sol[i, 2])             # updating the position of the satellite
    time_text.set_text(time_template % (i*dt))     # updating the time text
    return ln1, ln2, time_text

# passing the update function to the animation function of matplotlib

animation = ani.FuncAnimation(fig, update, np.arange(0, len(t)-1), interval=dt*1000, repeat=True)
plt.show()    # displaying the plot

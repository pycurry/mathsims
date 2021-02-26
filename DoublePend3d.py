# Import the required libraries

import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D         # This import registers the 3D projection, but is otherwise unused.

# setting the simulation parameters

G = 9.8   # acceleration due to gravity, in m/s^2
L1 = 1.5  # length of 1st arm in meters
L2 = 1.0  # length of 2nd arm in meters
M1 = 1.0  # 1st mass in kg
M2 = 1.0  # 2nd mass in kg

# Defining initial conditions of the pendulum

th1 = 120.0      # initial angle of 1st arm (theta1)
w1 = 0.0         # initial angular velocity of 1st mass (omega1)
th2 = 50.0       # initial angle of 2nd arm (theta2)
w2 = 0.0         # initial angular velocity of end mass (omega2)

# Equation is referred from https://www.myphysicslab.com/pendulum/double-pendulum-en.html
def derivs(state, t):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(del_) * cos(del_)
    dydx[1] = (M2 * L1 * state[1] * state[1] * sin(del_) * cos(del_) +
               M2 * G * sin(state[2]) * cos(del_) +
               M2 * L2 * state[3] * state[3] * sin(del_) -
               (M1 + M2) * G * sin(state[0])) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1
    dydx[3] = (-M2 * L2 * state[3] * state[3] * sin(del_) * cos(del_) +
               (M1 + M2) * G * sin(state[0]) * cos(del_) -
               (M1 + M2) * L1 * state[1] * state[1] * sin(del_) -
               (M1 + M2) * G * sin(state[2])) / den2

    return dydx


# create a time array from sampled at 0.05 second steps 
dt = 0.05                     # temporial resolution
t = np.arange(0.0, 20, dt)

# initializing the initial state (at t=0)
state = np.radians([th1, w1, th2, w2])

# integrate the ODE function using scipy.integrate
y = integrate.odeint(derivs, state, t)

# Converting polar position to Cartesian 
x1 = L1 * sin(y[:, 0])
y1 = -L1 * cos(y[:, 0])
x2 = L2 * sin(y[:, 2]) + x1
y2 = -L2 * cos(y[:, 2]) + y1

# Creating data structures for plotting
# NOTE: we re plotting time on the y-axis of the graph
dataset0 = np.array([x1, t, y1])        # for inner trail
dataset1 = np.array([x2, t, y2])        # for end tril

datasetSw = np.empty((3, 3, len(t)))    # for the pendulum arms
datasetSw[0, 0, :] = 0
datasetSw[2, 0, :] = 0
datasetSw[0, 1, :] = x1
datasetSw[0, 2, :] = x2
datasetSw[2, 1, :] = y1
datasetSw[2, 2, :] = y2
datasetSw[1, 0, :] = datasetSw[1, 1, :] = datasetSw[1, 2, :] = t

# Defining this fuction for updating frame data with the animation module
def update(num, dataset0, dataset1, datasetSw, line0, line1, line2, ax):
    # NOTE: there is no .set_data() for 3 dimensional data...
    line0.set_data(dataset0[0:2, :num])    
    line0.set_3d_properties(dataset0[2, :num])
    line1.set_data(dataset1[0:2, :num])    
    line1.set_3d_properties(dataset1[2, :num]) 
    line2.set_data(datasetSw[0:2, :, num])
    line2.set_3d_properties(datasetSw[2, :, num])
    ax.set_ylim3d((dataset0[1, num]-5), (dataset0[1, num]+1))   


# initializing the plotting window
# plt.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = Axes3D(fig)


# ax = fig.gca(projection='3d')

line0 = plt.plot(dataset0[0], dataset0[1], dataset0[2], lw=1, c='b', )[0]                      # Plotting Trail of inner mass
line1 = plt.plot(dataset1[0], dataset1[1], dataset1[2], lw=1, c='g', )[0]                      # Plotting Trail of end mass
line2 = plt.plot(datasetSw[0, :, 0], datasetSw[1, :, 0], datasetSw[2, :, 0], 'o-', lw=2 )[0]   # Plotting Swing-arms of pendulum


# AXES PROPERTIES
# ax.set_xlim3d([limit0, limit1])
ax.view_init(elev=0, azim=-90)
ax.set_xlabel('X(t)')
ax.set_ylabel('Time')
ax.set_zlabel('Y(t)')
ax.set_title('Trajectory of end mass')
 
# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update, frames=len(t), fargs=(dataset0, dataset1, datasetSw, line0, line1, line2, ax), 
                                    interval=50, blit=False)
# line_ani.save('AnimationNew.mp4', fps=24)

plt.show()

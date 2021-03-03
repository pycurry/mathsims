# Import the required libraries

import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

# setting the simulation parameters

G = 9.8   # acceleration due to gravity, in m/s^2
L1 = 1.5  # length of 1st arm in meters
L2 = 1.0  # length of 2nd arm in meters
M1 = 1.0  # 1st mass in kg
M2 = 1.0  # 2nd mass in kg

# Defining initial conditions of the pendulum

th1 = 120.0      # initial angle of 1st arm (theta1)
w1 = 0.0         # initial angular velocity of 1st mass (omega1)
th2 = -20.0      # initial angle of 2nd arm (theta2)
w2 = 0.0         # initial angular velocity of end mass (omega2)
mu = 0.05        # coeff of dampning

# Equation is referred from http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

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
dataset = np.empty((3, 3, len(t)))
dataset[0, 0, :] = 0
dataset[2, 0, :] = 0
dataset[0, 1, :] = x1
dataset[0, 2, :] = x2
dataset[2, 1, :] = y1
dataset[2, 2, :] = y2
dataset[1, 0, :] = dataset[1, 1, :] = dataset[1, 2, :] = t

# Defining this fuction for updating frame data with the animation module
def update(i, dataset, line0, line1, line2, ax):
    # NOTE: there is no .set_data() for 3 dimensional data...
    line0.set_data(dataset[0:2, 1, :i])    
    line0.set_3d_properties(dataset[2, 1, :i])
    line1.set_data(dataset[0:2, 2, :i])    
    line1.set_3d_properties(dataset[2, 2, :i]) 
    line2.set_data(dataset[0:2, :, i])
    line2.set_3d_properties(dataset[2, :, i])
    ax.set_ylim3d((dataset[1, 0, i]-5), (dataset[1, 0, i]))   


# initializing the plotting window
# plt.rcParams['legend.fontsize'] = 10
fig= plt.figure()
ax = fig.gca(projection='3d')

line0 = ax.plot(dataset[0, 1, :], dataset[1, 1, :], dataset[2, 1, :], lw=1, c='b', )[0]   # Plotting Trail of inner mass
line1 = ax.plot(dataset[0, 2, :], dataset[1, 2, :], dataset[2, 2, :], lw=1, c='g', )[0]   # Plotting Trail of end mass
line2 = ax.plot(dataset[0, :, 0], dataset[1, :, 0], dataset[2, :, 0], 'o-', lw=2 )[0]     # Plotting Swing-arms of pendulum


# AXES PROPERTIES
# ax.set_xlim3d([limit0, limit1])
ax.view_init(elev=0, azim=90)
ax.set_xlabel('X(t)')
ax.set_ylabel('Time')
ax.set_zlabel('Y(t)')
ax.set_title('Double Pendulum vs. Time')
 
# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update, frames=len(t), fargs=(dataset, line0, line1, line2, ax), 
                                    interval=50, blit=False)
# line_ani.save('AnimationNew.mp4', fps=24)

plt.show()

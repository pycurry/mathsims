import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Create the pendulum function, a second order differential equation that returns 'dydt' a vector

def pend(y, t, mu, g, length):
    theta, omega = y
    dydt = [omega, -mu * omega - (g / length) * np.sin(theta)]
    return dydt

length = 2
mu = 0.05
g = 9.8
theta0 = np.pi - 0.03
omega = 1.0
time = 20

y0 = [theta0, omega]
dt = 0.01
t = np.arange(0.0, time, dt)

# Simple Harmonic Motion
th = y0[0] * np.cos(np.sqrt(g / length) * t)

# Ordinary differential equation
sol = odeint(pend, y0, t, args=(mu, g, length))

x1 = -length * np.cos(sol[:, 0])
y1 = length * np.sin(sol[:, 0])

x2 = -length * np.cos(th)
y2 = length * np.sin(th)

fig, (ax1, ax2) = plt.subplots(1, 2)
xdata, ydata = [], []
x1data, y1data = [], []

ln1, = ax1.plot([], [], 'o-', color='blue')
ln2, = ax2.plot([], [], 'o-', color='red')
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_aspect('equal')
ax1.grid()
ax1.set_title('A')
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.set_aspect('equal')
ax2.grid()
ax2.set_title('B')

# plt.plot(t, sol[:, 0], 'b', label='theta(t)',)
# # plt.plot(t, sol[:, 1], 'g', label='omega(t)')
# # plt.plot(t, th, 'r:', label='harmonic')
# plt.legend(loc='best')
# plt.grid()
# plt.show()


# def init():
#     ax1.set_xlim(-4, 4)
#     ax1.set_ylim(-4, 4)
#     ax1.set_aspect('equal')
#     ax1.grid()
#     ax1.set_title('A')
#     ax2.set_xlim(-4, 4)
#     ax2.set_ylim(-4, 4)
#     ax2.set_aspect('equal')
#     ax2.grid()
#     ax2.set_title('B')
#     return ln1, ln2


def update(i):
    xdata = [0, y1[i]]
    ydata = [0, x1[i]]
    x1data = [0, y2[i]]
    y1data = [0, x2[i]]
    # xdata.append([0, x1[i]])
    # ydata.append([0, y1[i]])
    ln1.set_data(xdata, ydata)
    ln2.set_data(x1data, y1data)
    return ln1, ln2

animation = ani.FuncAnimation(fig, update, np.arange(0, len(t)), blit=True, interval=dt*1000,
                              repeat=False)
plt.show()
# , init_func=init
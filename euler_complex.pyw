#! D:/Python_Projects/Mathematics/.venv/Scripts/pythonw.exe

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, Button 

def exp(th):
    e = [1 +0j]

    for n in range(1, 20):
        e.append(e[-1] + (complex(0, th)**n)/np.math.factorial(n))

    X, Y = [x.real for x in e], [y.imag for y in e]
    return X, Y

th = 1
X, Y = exp(th)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.30)
ax.axis([-5, 5, -5, 5])
ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')
ax.set_aspect('equal')
ax.grid()
plt.title(r'$e^{i\theta}$')
# plt.style.use('dark_background')

angle = np.linspace(0, 2*np.pi, 100)
x1 = np.cos(angle)
y1 = np.sin(angle)

l1, = plt.plot(X, Y)
l2, = plt.plot(x1, y1, '--')
p3, = plt.plot(X[-1], Y[-1], '.', color= 'k')

axcolor = 'lightgoldenrodyellow'
axth = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
sth = Slider(axth, r'$\theta$', 0.1, 7.0, valinit=th, valstep=0.001)

zoom = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
zoomval = Slider(zoom, 'Axes Limits', 1.5, 20, valinit=5, valstep=0.01)

def update(value):
    th = sth.val
    zval = zoomval.val
    X, Y = exp(th)
    l1.set_xdata(X)
    l1.set_ydata(Y)
    p3.set_xdata(X[-1])
    p3.set_ydata(Y[-1])
    ax.axis([-zval, zval, -zval, zval])
    fig.canvas.draw_idle()

sth.on_changed(update)
zoomval.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

circleax = plt.axes([0.6, 0.025, 0.1, 0.04])
circle_button = Button(circleax, 'Circle', color=axcolor, hovercolor='0.975')

def reset(event):
    sth.reset()
    zoomval.reset()

button.on_clicked(reset)

def circle_toggle(event):
    l2.set_visible(not l2.get_visible())

circle_button.on_clicked(circle_toggle)

plt.show()
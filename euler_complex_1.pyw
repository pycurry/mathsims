#! D:/Python_Projects/Mathematics/.venv/Scripts/pythonw.exe

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, Button 

def exp(th):
    e = [complex(1, 0)]

    for n in range(1, 20):
        e.append(e[-1] + (complex(0, th)**n)/np.math.factorial(n))

    X, Y = [x.real for x in e], [y.imag for y in e]
    return X, Y

def stringop(X, Y):
    string1 = str()
    for tx1, tx2 in zip(X, Y):
        sign = '-' if np.sign(tx2) == -1 else '+'
        string2= f'{tx1:^10.3f} {sign} {abs(tx2):>7.3f}i\n'
        string1 += string2
    return string1

th = np.pi
X, Y = exp(th)
string1 = stringop(X, Y)

fig, ax = plt.subplots(1, 2)
plt.subplots_adjust(left=0.0, bottom=0.3)
ax[1].axis([-3, 3, -3, 3])
ax[1].set_xlabel('Real axis')
ax[1].set_ylabel('Imaginary axis')
ax[1].set_aspect('equal')
ax[1].grid()
fig.suptitle(r'$e^{i\theta}$', fontsize=18, fontweight='bold')
# plt.style.use('dark_background')

angle = np.linspace(0, 2*np.pi, 100)
x1 = np.cos(angle)
y1 = np.sin(angle)

l1, = ax[1].plot(X, Y)
l2, = ax[1].plot(x1, y1, '--')
p3, = ax[1].plot(X[-1], Y[-1], '.', color= 'k')

ax[0].axis([0, 5, 0, 10])
ax[0]. axis('off')
text = ax[0].text(1.5, -1.0, r'$\sum_{i=0}^{20} \frac{i\theta^n}{n!}$'+' =\n' + string1)
# , style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})


axcolor = 'lightgoldenrodyellow'

axth = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
sth = Slider(axth, r'$\theta$', 0.0, 7.0, valinit=th, valstep=0.001)

zoom = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
zoomval = Slider(zoom, 'Axes Limits', 1.5, 20, valinit=3, valstep=0.01)

def update(value):
    th = sth.val
    zval = zoomval.val
    X, Y = exp(th)
    l1.set_xdata(X)
    l1.set_ydata(Y)
    p3.set_xdata(X[-1])
    p3.set_ydata(Y[-1])
    ax[1].axis([-zval, zval, -zval, zval])
    string1 = stringop(X, Y)
    text.set_text(r'$\sum_{i=0}^{20} \frac{i\theta^n}{n!}$'+'=\n' + string1)
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
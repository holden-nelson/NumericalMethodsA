import numpy as np 
import matplotlib.pyplot as plt

from eulers import eulers

# comparing y' = ysinx, y(0) = 1, 0 <= x <= 4pi

# compute y with eulers method
x = np.linspace(0, 4*np.pi, 50)
y0 = 1
f = lambda x,y: y*np.sin(x)
y = eulers(f, y0, x)

# known y for comparison
y_true = np.exp(-1*np.cos(x)+1)

# print y50 to compare
print("Exact Value of y50: " + str(y_true[49]))
print("Approx Value of y50: " + str(y[49]))

# build graph and show
plt.plot(x, y, 'b.-', x, y_true, 'r-')
plt.legend(['Euler', 'True'])
plt.axis([0, 4*np.pi, 0, 8])
plt.grid(True)
plt.title("Solutions of $y'=ysinx , y(0)=1$")
plt.show()
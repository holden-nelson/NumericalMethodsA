import numpy as np 
import matplotlib.pyplot as plt

from eulers import eulers, improved_eulers, runge_kutta

# comparing y' = ycosx + x, y(0) = 1, 0 <= x <= 4pi

# compute y with eulers method
x = np.linspace(0, 4*np.pi, 50)
y0 = 1
f = lambda x,y: y*np.cos(x) + x
y_eulers = eulers(f, y0, x)
y_improved = improved_eulers(f, y0, x)
y_runge_kutta = runge_kutta(f, y0, x)

# print comparison
print("Comparison Table")
print("----------------")
print("Eulers\t\t\t\tImproved Eulers\t\t\tRunge-Kutta")
print("------\t\t\t\t---------------\t\t\t-----------")
for i in range(40, 50):
	print(str(y_eulers[i]) + '\t\t' 
	    + str(y_improved[i]) + '\t\t'
	    + str(y_runge_kutta[i]) + '\t\t')

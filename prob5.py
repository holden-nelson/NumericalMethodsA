import numpy as np 
import matplotlib.pyplot as plt

from eulers import eulers, improved_eulers, runge_kutta

# comparing y' = x^2 + y^2, y(0) = 1, 0 <= x <= 0.5

# compute y with eulers method
y0 = 1
f = lambda x,y: x**2 + y**2

# h = 50
x = np.linspace(0, 0.5, 50)
y_h50 = runge_kutta(f, y0, x)

# h = 200
x = np.linspace(0, 0.5, 200)
y_h200 = runge_kutta(f, y0, x)

# h = 500
x = np.linspace(0, 0.5, 500)
y_h500 = runge_kutta(f, y0, x)

# print comparison
print("Comparison Table")
print("----------------")
print("h\t\tapprox")
print("-\t\t------")
print("50\t\t" + str(y_h50[-1]))
print("200\t\t" + str(y_h200[-1]))
print("500\t\t" + str(y_h500[-1]))


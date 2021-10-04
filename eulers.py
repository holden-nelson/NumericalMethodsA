import numpy as np

def eulers(f, y0, x):
	'''
	Euler's method implementation in Python
	:param f: function. right hand side of diff eq y' = f(x,y)
	:param y0: init value y(x0) = y0
	:param x: 1D NumPy array of x values we use to approx
	                y values.

	:return: 1D numpy array
	         Approximation y[n] of the solution y(x_n)

	'''
	y = np.zeros(len(x))
	y[0] = y0
	for n in range(0, len(x)-1):
		h = x[n+1] - x[n]
		y[n+1] = y[n] + f(x[n], y[n])*h
	return y


def improved_eulers(f, y0, x):
	'''
	Improved Euler's method implementation in Python
	:param f: function. right hand side of diff eq y' = f(x,y)
	:param y0: init value y(x0) = y0
	:param x: 1D NumPy array of x values we use to approx
	                y values.

	:return: 1D numpy array
	         Approximation y[n] of the solution y(x_n)

	'''

	y = np.zeros(len(x))
	y[0] = y0
	for n in range(0, len(x)-1):
		h = x[n+1] - x[n]
		k1 = f(x[n], y[n])
		u = y[n] + h*k1
		k2 = f(x[n+1], u)
		y[n+1] = y[n] + h*((k1+k2)/2)

	return y


def runge_kutta(f, y0, x):
	'''
	Runge-Kutta method implementation in Python
	:param f: function. right hand side of diff eq y' = f(x,y)
	:param y0: init value y(x0) = y0
	:param x: 1D NumPy array of x values we use to approx
	                y values.

	:return: 1D numpy array
	         Approximation y[n] of the solution y(x_n)

	'''

	y = np.zeros(len(x))
	y[0] = y0
	for n in range(0, len(x)-1):
		h = x[n+1] - x[n]
		k1 = f(x[n], y[n])
		k2 = f(x[n] + h/2, y[n] + h*k1/2)
		k3 = f(x[n] + h/2, y[n] + h*k2/2)
		k4 = f(x[n+1], y[n] + h*k3)
		y[n+1] = y[n] + h*(k1 + 2*k2 + 2*k3 + k4)/6

	return y








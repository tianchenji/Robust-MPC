#!/usr/bin/env python

# variables in this file are in the form of np.matrix

import numpy as np
import matplotlib.pyplot as plt

def kf_predict(X, P, A, Q, B, U):
	'''
	Prediction Step
	'''
	X = A*X + B*U
	P = A*P*A.T + Q
	return (X, P)

def kf_update(X, P, Y, H, R):
	'''
	Update Step
	'''
	IM = H*X
	IS = R + H*P*H.T
	K = P*H.T*np.linalg.inv(IS)
	X = X + K*(Y - IM)
	P = P - K*IS*K.T
	return(X, P)

def single_var_ex():
	n_iter = 500
	x      = -0.37727 # real value
	Q      = np.matrix([[0.0577**2]])
	R      = np.matrix([[2**2]])

	xhat         = [0] * n_iter
	measurements = [0] * n_iter
	xreal        = [0] * n_iter
	xreal[0]     = x
	X = x # initial guess

	P = np.matrix([[1.0]])
	A = np.matrix([[1.0]])
	B = np.matrix([[0.0]])
	U = np.matrix([[0.0]])
	H = np.matrix([[1.0]])

	'''
	measurements start from k = 1
	xhat_0_plus = E(x_0)
	'''
	for i in range(1, n_iter):
		xreal[i] = xreal[i - 1] + np.random.normal(0, 0.0577) # real states
		y = np.asmatrix(xreal[i] + np.random.normal(0, 2)) # observations
		measurements[i] = float(y)
		(X, P) = kf_predict(X, P, A, Q, B, U)
		(X, P) = kf_update(X, P, y, H, R)
		print(P)
		xhat[i] = float(X)

	xreal = xreal[1:len(xreal)]
	xhat  = xhat[1:len(xhat)]
	measurements = measurements[1:len(measurements)]
	plt.figure()
	plt.plot(measurements, 'k+', label='noisy measurements')
	plt.plot(xhat, 'b.-', label='a posteri estimate')
	plt.plot(xreal, 'r.-', label='real states')
	plt.axhline(x, color='g', label='truth value')
	plt.legend()
	plt.grid()
	plt.show()

single_var_ex()
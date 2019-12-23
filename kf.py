#!/usr/bin/env python

# variables in this file are in the form of np.array

import numpy as np
import matplotlib.pyplot as plt

def kf_predict(X, P, A, Q, B, U):
	'''
	Prediction Step
	'''
	X = np.dot(A, X) + np.dot(B, U)
	P = np.dot(A, np.dot(P, A.T)) + Q
	return (X, P)

def kf_update(X, P, Y, H, R):
	'''
	Update Step
	'''
	IM = np.dot(H, X)
	IS = R + np.dot(H, np.dot(P, H.T))
	K = np.dot(P, np.dot(H.T, np.linalg.inv(IS)))
	X = X + np.dot(K, (Y - IM))
	P = P - np.dot(K, np.dot(IS, K.T))
	return (X, P)

def single_var_ex():
	n_iter = 500
	x      = -0.37727 # real value
	Q      = np.array([[0.0577**2]])
	R      = np.array([[2**2]])

	xhat         = [0] * n_iter
	measurements = [0] * n_iter
	xreal        = [0] * n_iter
	xreal[0]     = x
	X = x # initial guess

	P = np.array([[1.0]])
	A = np.array([[1.0]])
	B = np.array([[0.0]])
	U = np.array([[0.0]])
	H = np.array([[1.0]])

	'''
	measurements start from k = 1
	xhat_0_plus = E(x_0)
	'''
	for i in range(1, n_iter):
		xreal[i] = xreal[i - 1] + np.random.normal(0, 0.0577) # real states
		y = xreal[i] + np.random.normal(0, 2) # observations
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

def multi_var_ex():
	n_iter = 50
	x      = np.array([[1.2], [2.7]]) # real value
	Q      = np.array([[0.0577**2, 0], [0, 0.0577**2]])
	R      = np.array([[2**2, 0], [0, 2**2]])

	xreal           = [0] * n_iter
	xreal[0]        = x
	# we visualize the first state
	x1_hat          = [0] * n_iter
	x1_measurements = [0] * n_iter
	x1_real         = [0] * n_iter
	x1_real[0]      = float(x[0])
	X = x # initial guess

	P = np.array([[1, 0], [0, 1]])
	A = np.array([[1, 0], [0, 1]])
	B = np.array([[0], [0]])
	U = np.array([[0]])
	H = np.array([[1, 0], [0, 1]])

	for i in range(1, n_iter):
		xreal[i] = xreal[i - 1] + np.random.normal(0, 0.0577, (2, 1))
		x1_real[i] = float(xreal[i][0])
		y = xreal[i] + np.random.normal(0, 2, (2, 1))
		x1_measurements[i] = float(y[0])
		(X, P) = kf_predict(X, P, A, Q, B, U)
		(X, P) = kf_update(X, P, y, H, R)
		print(P)
		x1_hat[i] = float(X[0])

	x1_real = x1_real[1:len(x1_real)]
	x1_hat  = x1_hat[1:len(x1_hat)]
	x1_measurements = x1_measurements[1:len(x1_measurements)]
	plt.figure()
	plt.plot(x1_measurements, 'k+', label='noisy measurements')
	plt.plot(x1_hat, 'b.-', label='a posteri estimate')
	plt.plot(x1_real, 'r.-', label='real states')
	plt.axhline(x[0], color='g', label='nominal value without noise')
	plt.legend()
	plt.grid()
	plt.show()

multi_var_ex()
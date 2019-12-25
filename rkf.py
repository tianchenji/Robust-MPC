#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Robust Kalman Filter (RKF)
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2019-12-19
# ---------------------------------------------------------------------------

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def rkf_update(A, B, C, Q, R, Sigma, z, xhat, delta):
	'''
	robust filtering algorithm for energy constraint
	rkf_update returns state estimate xhat, ellipsoid shape matrix Sigma and shrinkage delta at time k
	Inputs: 	A, B, C: system dynamics
				Q, R: energy constraints of process noise and measurement noise respectively
				Sigma: a posteri error covariance at time k - 1
				z: measurement at time k
				xhat: a posteri state estimate at time k - 1
				delta: shrinkage at time k - 1
	Outputs:	Sigma: a posteri error covariance at time k
				xhat: a posteri state estimate at time k
				delta: shrinkage at time k
	'''

	# update Sigma
	Sigma_priori  = np.dot(A, np.dot(Sigma, A.T)) + np.dot(B, np.dot(Q, B.T))
	Sigma_posteri = np.linalg.inv(np.linalg.inv(Sigma_priori) + np.dot(C.T, np.dot(np.linalg.inv(R), C)))

	# update xhat
	IM = z - np.dot(C, np.dot(A, xhat))
	xhat = np.dot(A, xhat) + np.dot(Sigma_posteri, np.dot(C.T, np.dot(np.linalg.inv(R), IM)))

	# update shrinkage delta
	IS = np.linalg.inv(np.dot(C, np.dot(Sigma_priori, C.T)) + R)
	delta += np.dot(IM.T, np.dot(IS, IM))

	return (Sigma_posteri, xhat, delta)

def rbe_update(A, B, C, Q, R, Sigma, z, xhat, delta):
	'''
	robust filtering algorithm for instantaneous constraint
	rbe_update returns state estimate xhat, ellipsoid shape matrix Sigma and shrinkage delta at time k
	Inputs: 	A, B, C: system dynamics
				Q, R: energy constraints of process noise and measurement noise respectively
				Sigma: a posteri error covariance at time k - 1
				z: measurement at time k
				xhat: a posteri state estimate at time k - 1
				delta: shrinkage at time k - 1
	Outputs:	Sigma: a posteri error covariance at time k
				xhat: a posteri state estimate at time k
				delta: shrinkage at time k
	'''

	# set filtering parameters
	beta = 0.5
	rho  = 0.5

	# update Sigma
	Sigma_priori  = (1 / (1 - beta)) * np.dot(A, np.dot(Sigma, A.T)) + (1 / beta) * np.dot(B, np.dot(Q, B.T))
	Sigma_posteri = np.linalg.inv((1 - rho) * np.linalg.inv(Sigma_priori) + rho * np.dot(C.T, np.dot(np.linalg.inv(R), C)))

	# update xhat
	IM = z - np.dot(C, np.dot(A, xhat))
	xhat = np.dot(A, xhat) + rho * np.dot(Sigma_posteri, np.dot(C.T, np.dot(np.linalg.inv(R), IM)))

	# update shrinkage delta
	IS = np.linalg.inv((1 / (1 - rho)) * np.dot(C, np.dot(Sigma_priori, C.T)) + (1 / rho) * R)
	delta = (1 - beta) * (1 - rho) * delta + np.dot(IM.T, np.dot(IS, IM))

	return (Sigma_posteri, xhat, delta)


def rbe_project(Sigma, xhat, delta):
	'''
	rkf_project returns lowerbound and upperbound of the state estimates
	Inputs:		Sigma: a posteri error covariance at time k
				xhat: a posteri state estimate at time k
				delta: shrinkage at time k
	Outputs:	s_i_min: the lowerbound of the ith state estimate at time k
				s_i_max: the upperbound of the ith state estimate at time k
	'''

	# cholesky decomposition of shape matrix
	L = np.linalg.cholesky(np.linalg.inv(Sigma))
	x0 = np.array([[0], [0]])

	# state estimates start at index 0
	v0 = np.array([[1], [0]])
	v1 = np.array([[0], [1]])

	# the center of ellipsoid
	c = xhat

	# projection of the center
	s0_0 = np.dot(np.transpose(v0), c - x0) / np.dot(np.transpose(v0), v0)
	s1_0 = np.dot(np.transpose(v1), c - x0) / np.dot(np.transpose(v1), v1)

	# projection of the bounds
	w0 = np.dot(np.linalg.inv(L), v0) / np.dot(np.transpose(v0), v0)
	w1 = np.dot(np.linalg.inv(L), v1) / np.dot(np.transpose(v1), v1)

	norm_w0 = np.linalg.norm(w0) * sqrt(1 - delta)
	norm_w1 = np.linalg.norm(w1) * sqrt(1 - delta)
	s0_min = float(s0_0 - norm_w0)
	s0_max = float(s0_0 + norm_w0)
	s1_min = float(s1_0 - norm_w1)
	s1_max = float(s1_0 + norm_w1)

	return (s0_min, s0_max, s1_min, s1_max)

# bivariate example
n_iter = 50
x      = np.array([[12], [27]]) # real value
Q      = np.array([[0.02, 0], [0, 0.02]])
R      = np.array([[0.02, 0], [0, 0.02]])

xreal           = [0] * n_iter
xreal[0]        = x
# we visualize the first state
x1_hat          = [0] * n_iter
x1_measurements = [0] * n_iter
x1_real         = [0] * n_iter
x1_lowerbound   = [0] * n_iter
x1_upperbound   = [0] * n_iter
x1_real[0]      = float(x[0])
xhat  = x # initial guess
delta = 0

Sigma = np.array([[1, 0], [0, 1]])
A     = np.array([[1, 0], [0, 1]])
B     = np.array([[1, 0], [0, 1]])
C     = np.array([[1, 0], [0, 1]])

for i in range(1, n_iter):
	xreal[i] = xreal[i - 1] + np.random.uniform(-0.1, 0.1, (2, 1))
	x1_real[i] = float(xreal[i][0])
	y = xreal[i] + np.random.uniform(-0.1, 0.1, (2, 1))
	x1_measurements[i] = float(y[0])
	(Sigma, xhat, delta) = rbe_update(A, B, C, Q, R, Sigma, y, xhat, delta)
	(s0_min, s0_max, s1_min, s1_max) = rbe_project(Sigma, xhat, delta)
	x1_lowerbound[i] = s0_min
	x1_upperbound[i] = s0_max
	x1_hat[i] = float(xhat[0])

x1_real = x1_real[1:len(x1_real)]
x1_hat  = x1_hat[1:len(x1_hat)]
x1_measurements = x1_measurements[1:len(x1_measurements)]
x1_lowerbound = x1_lowerbound[1:len(x1_lowerbound)]
x1_upperbound = x1_upperbound[1:len(x1_upperbound)]
plt.figure()
plt.plot(x1_measurements, 'k+', label='noisy measurements')
plt.plot(x1_hat, 'b.-', label='a posteri estimate')
plt.plot(x1_real, '.-', label='real states')
plt.plot(x1_lowerbound, 'r.-', label='lowerboud of state estimate')
plt.plot(x1_upperbound, 'r.-', label='upperboud of state estimate')
plt.axhline(x[0], label='nominal value without noise')
plt.legend()
plt.grid()
plt.show()
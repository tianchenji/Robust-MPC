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
import time

# set filtering parameters
beta = 0.5
rho  = 0.5

def rkf_update(A, B, E, C, Q, R, u, Sigma, z, xhat, delta):
	'''
	robust filtering algorithm for energy constraint
	rkf_update returns state estimate xhat, ellipsoid shape matrix Sigma and shrinkage delta at time k
	Inputs: 	A, B, E, C: system dynamics
				Q, R: energy constraints of process noise and measurement noise respectively
				u: control input at time k - 1
				Sigma: a posteri error covariance at time k - 1
				z: measurement at time k
				xhat: a posteri state estimate at time k - 1
				delta: shrinkage at time k - 1
	Outputs:	Sigma: a posteri error covariance at time k
				xhat: a posteri state estimate at time k
				delta: shrinkage at time k
	'''

	# update Sigma
	Sigma_priori  = np.dot(A, np.dot(Sigma, A.T)) + np.dot(E, np.dot(Q, E.T))
	Sigma_posteri = np.linalg.inv(np.linalg.inv(Sigma_priori) + np.dot(C.T, np.dot(np.linalg.inv(R), C)))

	# update xhat
	IM = z - np.dot(C, (np.dot(A, xhat) + np.dot(B, u)))
	xhat = np.dot(A, xhat) + np.dot(B, u) + np.dot(Sigma_posteri, np.dot(C.T, np.dot(np.linalg.inv(R), IM)))

	# update shrinkage delta
	IS = np.linalg.inv(np.dot(C, np.dot(Sigma_priori, C.T)) + R)
	delta += np.dot(IM.T, np.dot(IS, IM))

	return (Sigma_posteri, xhat, delta)

def rbe_update(A, B, E, C, Q, R, u, Sigma, z, xhat, delta):
	'''
	robust filtering algorithm for instantaneous constraint
	rbe_update returns state estimate xhat, ellipsoid shape matrix Sigma and shrinkage delta at time k
	Inputs: 	A, B, E, C: system dynamics
				Q, R: energy constraints of process noise and measurement noise respectively
				u: control input at time k - 1
				Sigma: a posteri error covariance at time k - 1
				z: measurement at time k
				xhat: a posteri state estimate at time k - 1
				delta: shrinkage at time k - 1
	Outputs:	Sigma: a posteri error covariance at time k
				xhat: a posteri state estimate at time k
				delta: shrinkage at time k
	'''

	# update Sigma
	Sigma_priori  = (1 / (1 - beta)) * np.dot(A, np.dot(Sigma, A.T)) + (1 / beta) * np.dot(E, np.dot(Q, E.T))
	Sigma_posteri = np.linalg.inv((1 - rho) * np.linalg.inv(Sigma_priori) + rho * np.dot(C.T, np.dot(np.linalg.inv(R), C)))

	# update xhat
	IM = z - np.dot(C, (np.dot(A, xhat) + np.dot(B, u)))
	xhat = np.dot(A, xhat) + np.dot(B, u) + rho * np.dot(Sigma_posteri, np.dot(C.T, np.dot(np.linalg.inv(R), IM)))

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

def rbe_stable(A, E, C, Q, R, sigma):
	'''
	rbe_stable returns upperbounds and lowerbounds of state estimates error until convergence
	Inputs:		A, E, C: system dynamics
				Q, R: energy constraints of process noise and measurement noise respectively
				sigma: initial constraints for state estimates error
	Outputs:	a list containing upperbounds and lowerbounds of state estimates error
	'''

	num_of_iter = 100
	xhat = np.array([[0],[0]])
	delta = 0
	lb_eps = []
	ub_eps = []

	# iterate 100 times to get steady state shape matrix simga_inf
	sigma_tmp = sigma
	for i in range(num_of_iter):
		sigma_priori  = (1 / (1 - beta)) * np.dot(A, np.dot(sigma_tmp, A.T)) + (1 / beta) * np.dot(E, np.dot(Q, E.T))
		sigma_posteri = np.linalg.inv((1 - rho) * np.linalg.inv(sigma_priori) + rho * np.dot(C.T, np.dot(np.linalg.inv(R), C)))
		sigma_tmp = sigma_posteri
	sigma_inf = sigma_tmp

	# store the upperbounds and lowerbounds until convergence
	while np.linalg.norm(sigma - sigma_inf) > 1e-5:
		(xreal1_min, xreal1_max, xreal2_min, xreal2_max) = rbe_project(sigma, xhat, delta)
		err1 = (xreal1_max - xreal1_min) / 2
		err2 = (xreal2_max - xreal2_min) / 2
		lb_eps.append([- err1, - err2])
		ub_eps.append([err1, err2])

		sigma_priori  = (1 / (1 - beta)) * np.dot(A, np.dot(sigma, A.T)) + (1 / beta) * np.dot(E, np.dot(Q, E.T))
		sigma_posteri = np.linalg.inv((1 - rho) * np.linalg.inv(sigma_priori) + rho * np.dot(C.T, np.dot(np.linalg.inv(R), C)))
		sigma = sigma_posteri

	# store the steady state bounds in the end
	(xreal1_min, xreal1_max, xreal2_min, xreal2_max) = rbe_project(sigma_inf, xhat, delta)
	err1 = (xreal1_max - xreal1_min) / 2
	err2 = (xreal2_max - xreal2_min) / 2
	lb_eps.append([- err1, - err2])
	ub_eps.append([err1, err2])

	return (lb_eps, ub_eps)

if __name__ == '__main__':
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
	B     = np.array([[1.5], [0.0]])
	E     = np.array([[1, 0], [0, 1]])
	C     = np.array([[1, 0], [0, 1]])
	u     = np.array([[0.02]])

	start = time.clock()

	for i in range(1, n_iter):
		xreal[i] = np.dot(A, xreal[i - 1]) + np.dot(B, u) + np.random.uniform(-0.1, 0.1, (2, 1))
		x1_real[i] = float(xreal[i][0])
		y = np.dot(C, xreal[i]) + np.random.uniform(-0.1, 0.1, (2, 1))
		x1_measurements[i] = float(y[0])
		(Sigma, xhat, delta) = rbe_update(A, B, E, C, Q, R, u, Sigma, y, xhat, delta)
		print(Sigma)
		(s0_min, s0_max, s1_min, s1_max) = rbe_project(Sigma, xhat, delta)
		x1_lowerbound[i] = s0_min
		x1_upperbound[i] = s0_max
		x1_hat[i] = float(xhat[0])

	end = time.clock()

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
	print(end-start)
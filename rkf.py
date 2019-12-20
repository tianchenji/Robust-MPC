#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Robust Kalman Filter (RKF)
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2019-12-19
# ---------------------------------------------------------------------------

import numpy as np

def rkf_update(A, B, C, Q, R, Sigma, z, xhat, delta):
	'''
	rkf returns state estimate xhat, ellipsoid shape matrix Sigma and shrinkage delta at time k
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

	return (Sigma, xhat, delta)
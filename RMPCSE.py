#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Robust Model Predictive Control (RMPC)
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2019-11-06
# ---------------------------------------------------------------------------

from casadi import *
import numpy as np
from scipy.linalg import solve_discrete_are
from rkf import rbe_update, rbe_project, rbe_stable
import matplotlib.pyplot as plt
import time

class FirstStateIndex:
	'''
	FirstStateIndex aims for readability
	Note: the length of horizon includes initial states
	'''
	def __init__(self, A, B, N):
		'''
		A, B: system dynamic matrices
		N: the prediction horizon
		'''
		self.s = [0] * np.shape(A)[0]
		self.v = [0] * np.shape(B)[1]
		self.s[0] = 0
		self.v[0] = np.shape(A)[0] * N
		for i in range(np.shape(A)[0] - 1):
			self.s[i + 1] = self.s[i] + N
		for i in range(np.shape(B)[1] - 1):
			self.v[i + 1] = self.v[i] + N - 1

class RMPC:

	def __init__(self, A, B, D, F, G, P, K, V_w, V_eps, f, lb_w, ub_w, lb_eps_0, ub_eps_0, r, N):
		'''
		A, B, D: system dynamic matrices
		F, G: constriant matrices
		P: terminal cost
		K: fixed stabilizing feedback gain
		V_w: the matrix bounding W
		V_eps: the matrix bounding eps_0
		f: states and input constraints
		lb_w: lowerbound of the system noise
		ub_w: upperbound of the system noise
		lb_eps_0: lowerbound of eps_0
		ub_eps_0: upperbound of eps_0
		r: parameters in approximating mRPI
		N: the prediction horizon
		'''

		self.A = A
		self.B = B
		self.D = D
		self.F = F
		self.G = G
		self.P = P
		self.K = K
		self.V_w = V_w
		self.V_eps = V_eps
		self.f = f
		self.w_lb = lb_w
		self.w_ub = ub_w
		self.lb_eps_0 = lb_eps_0
		self.ub_eps_0 = ub_eps_0
		self.r        = r
		self.horizon  = N
		self.first_state_index = FirstStateIndex(A=A, B=B, N=N)
		# number of optimization variables
		self.num_of_x = np.shape(self.A)[0] * self.horizon + np.shape(self.B)[1] * (self.horizon - 1)
		self.num_of_g = np.shape(self.A)[0] * self.horizon + np.shape(self.F)[0] * self.horizon

	def mRPI(self):
		'''
		mRPI returns the degree by which constraints are tightened
		'''

		n_x   = np.shape(self.A)[0]
		n_w   = np.shape(self.D)[1]
		n_h   = np.shape(self.F)[0]
		h_w   = [0]*n_h
		h_eps = [0]*n_h
		h     = [0]*n_h

		# calculate vector h_w
		# calculating rho_w given r
		phi = self.A + np.dot(self.B, self.K)
		n_rho_w = np.shape(self.V_w)[0]
		mrho_w = [None]*n_rho_w

		# define optimization variables
		w = SX.sym('w', n_w)

		# define costs for linear programs in matrix form
		tmp = np.dot(self.V_w, np.dot(np.linalg.pinv(self.D), np.dot(np.linalg.matrix_power(phi, self.r), self.D)))
		rhocost_w = - mtimes(tmp, w)

		# solve n_rho_w linear programs
		for i in range(n_rho_w):
			nlp = {'x':w, 'f':rhocost_w[i]}
			opts = {}
			opts["ipopt.print_level"] = 0
			opts["print_time"] = 0
			solver = nlpsol('solver', 'ipopt', nlp, opts)
			x0 = [0] * n_w
			res = solver(x0=x0, lbx=self.w_lb, ubx=self.w_ub)
			mrho_w[i] = - res['f']
		rho_w = max(mrho_w)

		# calculate vector h_w by solving r * n_h linear programs
		for j in range(self.r):
			tmp = self.F + np.dot(self.G, self.K)
			hcost_w = - mtimes(np.dot(tmp, np.dot(np.linalg.matrix_power(phi, j), self.D)), w)
			for k in range(n_h):
				nlp = {'x':w, 'f':hcost_w[k]}
				opts = {}
				opts["ipopt.print_level"] = 0
				opts["print_time"] = 0
				solver = nlpsol('solver', 'ipopt', nlp, opts)
				x0 = [0] * n_w
				res = solver(x0=x0, lbx=self.w_lb, ubx=self.w_ub)
				h_w[k] += - res['f']
		h_w = [i/(1 - rho_w) for i in h_w]


		# calculate vector h_eps
		# calculating rho_eps given r
		psi = np.dot(self.B, self.K)
		n_rho_eps = np.shape(self.V_eps)[0]
		mrho_eps = [None]*n_rho_eps

		# define optimization variables
		eps = SX.sym('eps', n_x)

		# define costs for linear programs in matrix form
		tmp = np.dot(self.V_eps, np.dot(np.linalg.pinv(psi), np.dot(np.linalg.matrix_power(phi, self.r), psi)))
		rhocost_eps = - mtimes(tmp, eps)

		# solve n_rho_eps linear programs
		for i in range(n_rho_eps):
			nlp = {'x':eps, 'f':rhocost_eps[i]}
			opts = {}
			opts["ipopt.print_level"] = 0
			opts["print_time"] = 0
			solver = nlpsol('solver', 'ipopt', nlp, opts)
			x0 = [0] * n_x
			res = solver(x0=x0, lbx=self.lb_eps_0, ubx=self.ub_eps_0)
			mrho_eps[i] = - res['f']
		rho_eps = max(mrho_eps)

		# calculate vector h_eps by solving r * n_h linear programs
		for j in range(self.r):
			tmp = self.F + np.dot(self.G, self.K)
			hcost_eps = - mtimes(np.dot(tmp, np.dot(np.linalg.matrix_power(phi, j), psi)), eps)
			for k in range(n_h):
				nlp = {'x':eps, 'f':hcost_eps[k]}
				opts = {}
				opts["ipopt.print_level"] = 0
				opts["print_time"] = 0
				solver = nlpsol('solver', 'ipopt', nlp, opts)
				x0 = [0] * n_x
				res = solver(x0=x0, lbx=self.lb_eps_0, ubx=self.ub_eps_0)
				h_eps[k] += - res['f']
		h_eps = [i/(1 - rho_eps) for i in h_eps]

		h = [h_w[i] + h_eps[i] for i in range(len(h_w))]

		return h

	def SEerr(self, lb_eps, ub_eps):
		'''
		SEerr returns the degree by which constraints are tightened by measurement noise
		'''

		n_eps  = np.shape(self.A)[0]
		n_p    = np.shape(self.G)[0]
		p      = [0]*n_p
		p_list = []

		# define optimization variables
		eps = SX.sym('eps', n_eps)

		# define costs for linear programs in matrix form
		tmp = np.dot(self.G, self.K)
		pcost = - mtimes(tmp, eps)

		# calculate vector p
		for i in range(len(lb_eps)):
			for j in range(n_p):
				nlp = {'x':eps, 'f':pcost[j]}
				opts = {}
				opts["ipopt.print_level"] = 0
				opts["print_time"] = 0
				solver = nlpsol('solver', 'ipopt', nlp, opts)
				x0 = [0] * n_eps
				res = solver(x0=x0, lbx=lb_eps[i], ubx=ub_eps[i])
				p[j] = - res['f']
			p = list(map(float, p))
			p_list.append(p)
			p = [0] * n_p

		return p_list

	def RMPC(self, h, s_0, p_list, time_index):
		'''
		RMPC returns optimal control sequence
		'''

		num_of_ss = len(p_list)

		# initial variables
		x_0 = [0] * self.num_of_x
		for i in range(len(self.first_state_index.s)):
			x_0[self.first_state_index.s[i]] = s_0[i]

		# define optimization variables
		x = SX.sym('x', self.num_of_x)

		states = [0] * self.horizon
		aux_input = [0] * (self.horizon - 1)

		ineq_cons_index = np.shape(self.A)[0] * self.horizon

		# define lowerbound and upperbound of g constraints
		g_lowerbound = [0] * self.num_of_g
		g_upperbound = [0] * self.num_of_g

		for i in range(len(self.first_state_index.s)):
			g_lowerbound[self.first_state_index.s[i]] = s_0[i]
			g_upperbound[self.first_state_index.s[i]] = s_0[i]

		for i in range(np.shape(self.A)[0] * self.horizon, self.num_of_g):
			g_lowerbound[i] = -exp(10)
		#for i in range(self.horizon):
			#for j in range(np.shape(self.F)[0]):
				#g_upperbound[ineq_cons_index + j * self.horizon + i] = self.f[j] - h[j]
		if time_index < num_of_ss - 1:
			for i in range(num_of_ss - time_index - 1):
				for j in range(np.shape(self.F)[0]):
					g_upperbound[ineq_cons_index + j * self.horizon + i] = self.f[j] - h[j] - p_list[time_index + i][j]
			for i in range(num_of_ss - time_index - 1, self.horizon):
				for j in range(np.shape(self.F)[0]):
					g_upperbound[ineq_cons_index + j * self.horizon + i] = self.f[j] - h[j] - p_list[num_of_ss - 1][j]
		else:
			for i in range(self.horizon):
				for j in range(np.shape(self.F)[0]):
					g_upperbound[ineq_cons_index + j * self.horizon + i] = self.f[j] - h[j] - p_list[num_of_ss - 1][j]
		# no constraints on input at time step N - 1
		g_upperbound[self.num_of_g - 1] = exp(10)
		g_upperbound[self.num_of_g - self.horizon - 1] = exp(10)

		# define cost functions
		cost = 0.0
		# penalty on states
		for i in range(len(self.first_state_index.s)):
			for j in range(self.horizon - 1):
				#cost += fabs(x[self.first_state_index.s[i] + j])
				cost += (x[self.first_state_index.s[i] + j]**2)
		## penalty on terminal states
		#for i in range(len(self.first_state_index.s)):
			##cost += 10 * fabs(x[self.first_state_index.s[i] + self.horizon - 1])
			#cost += 10 * (x[self.first_state_index.s[i] + self.horizon - 1]**2)
		# penalty on terminal states
		terminal_states = x[self.first_state_index.s[0] + self.horizon - 1:self.first_state_index.v[0]:self.horizon]
		cost += mtimes(terminal_states.T, mtimes(self.P, terminal_states))
		# penalty on control inputs
		for i in range(len(self.first_state_index.v)):
			for j in range(self.horizon - 1):
				#cost += 10 * fabs(x[self.first_state_index.v[i] + j])
				cost += 10 * (x[self.first_state_index.v[i] + j]**2)

		# define g constraints
		g = [None] * self.num_of_g
		for i in range(len(self.first_state_index.s)):
			g[self.first_state_index.s[i]] = x[self.first_state_index.s[i]]

		# constraints based on system dynamic equations
		for i in range(self.horizon):
			states[i] = x[self.first_state_index.s[0] + i:self.first_state_index.v[0]:self.horizon]
		for i in range(self.horizon - 1):
			aux_input[i] = x[self.first_state_index.v[0] + i::(self.horizon - 1)]
		
		# equality constraints
		for i in range(self.horizon - 1):
			for j in range(len(self.first_state_index.s)):
				g[1 + self.first_state_index.s[j] + i] = \
				    (states[1 + i] - mtimes(self.A, states[i]) - mtimes(self.B, aux_input[i]))[j]

		# inequality constraints
		for i in range(self.horizon - 1):
			for j in range(np.shape(self.F)[0]):
				g[ineq_cons_index + j * self.horizon + i] = \
				    (mtimes(self.F, states[i]) + mtimes(self.G, aux_input[i]))[j]
		for j in range(np.shape(self.F)[0]):
			g[ineq_cons_index + j * self.horizon + self.horizon - 1] = \
			    (mtimes(self.F, states[self.horizon - 1]))[j]

		# create the NLP
		nlp = {'x':x, 'f':cost, 'g':vertcat(*g)}

		# solver options
		opts = {}
		opts["ipopt.print_level"] = 0
		opts["print_time"] = 0

		solver = nlpsol('solver', 'ipopt', nlp, opts)

		# solve the NLP
		#print(g[ineq_cons_index + 3 * self.horizon + 1])
		#print(g_lowerbound[ineq_cons_index + 3 * self.horizon + 1])
		#print(g_upperbound[ineq_cons_index + 3 * self.horizon + 1])
		
		res = solver(x0=x_0, lbg=g_lowerbound, ubg=g_upperbound)
		return res

def lqr(A, B, Q, R):
	'''
	lqr solves the discrete time lqr controller.
	'''

	P   = solve_discrete_are(A, B, Q, R)
	tmp = np.linalg.inv(R + np.dot(B.T, np.dot(P, B)))
	K   = - np.dot(tmp, np.dot(B.T, np.dot(P, A)))
	return (P, K)


# system dynaimcs
A = np.array([[0.5,0],[0.5,1]])
B = np.array([[1],[0]])
D = np.array([[-1,0],[0,-1]])

# observer matrix
H = np.array([[1, 0], [0, 1]])

# states and input constraints
F = np.array([[-10/3,0],[10/7,0],[0,-2],[0,2],[0,0],[0,0]])
G = np.array([[0],[0],[0],[0],[-10/3],[5]])
f = np.array([[1],[1],[1],[1],[1],[1]])

# bounds for process noise
V_w  = np.array([[20,0],[-20,0],[0,20],[0,-20]])
lb_w = [-0.05] * 2
ub_w = [0.05] * 2

# bounds for measurement noise
lb_zeta = [-0.01] * 2
ub_zeta = [0.01] * 2

# initial constraints and initial guess
sigma = np.array([[0.001, 0], [0, 0.001]])
x_hat = np.array([[0.58],[-0.18]])
delta = 0

# instantaneous constraints in filtering
Q = np.array([[2*ub_w[0]**2, 0], [0, 2*ub_w[0]**2]])
R = np.array([[2*ub_zeta[0]**2, 0], [0, 2*ub_zeta[0]**2]])

# calculate LQR gain matrix
Q_lqr  = np.array([[1, 0], [0, 1]])
R_lqr  = np.array([[0.01]])
(P, K) = lqr(A, B, Q_lqr, R_lqr)

# mRPI parameters
r = 6

# prediction horizon
N = 10

s_0 = np.array([[0.6],[-0.2]])
x_ori_0 = s_0
threshold = pow(10, -5)
vis_x = []
vis_y = []
vis_x.append(list(map(float,x_ori_0[0])))
vis_y.append(list(map(float,x_ori_0[1])))

# calculate epsilon_0
(xreal1_min, xreal1_max, xreal2_min, xreal2_max) = rbe_project(sigma, x_hat, delta)
err1 = (xreal1_max - xreal1_min) / 2
err2 = (xreal2_max - xreal2_min) / 2
lb_eps_0 = [- err1, - err2]
ub_eps_0 = [err1, err2]

V_eps = np.array([[1/err1,0],[-1/err1,0],[0,1/err2],[0,-1/err2]])

rmpc = RMPC(A=A, B=B, D=D, F=F, G=G, P=P, K=K, V_w=V_w, V_eps=V_eps, f=f, lb_w=lb_w, ub_w=ub_w, \
																	lb_eps_0=lb_eps_0, ub_eps_0=ub_eps_0, r=r, N=N)

#(lb_eps, ub_eps) = rbe_stable(A, D, H, Q, R, sigma)
#p_list = rmpc.SEerr(lb_eps, ub_eps)

h = list(map(float, rmpc.mRPI()))
if max(h) >= 1:
	print("Robustly positively invariant set is empty! Cannot achieve robustness!")
	sys.exit()

# calculate states estimates error
(lb_eps, ub_eps) = rbe_stable(A, D, H, Q, R, sigma)
p_list = rmpc.SEerr(lb_eps, ub_eps)

time_index = 0
start = time.clock()
sol = rmpc.RMPC(h, s_0, p_list, time_index)
end = time.clock()

# constraints visualization variables
constraints_varlist = [0] * (rmpc.horizon - 1)
constraint_var      = [0] * np.shape(F)[0]
vis_flag            = 0

# keep iterating until the cost is less than the threshold
while sol["f"] > threshold:
	# calculate optimal control
	v_opt = np.asarray(sol["x"][rmpc.first_state_index.v[0]::(rmpc.horizon - 1)])
	u_opt = np.dot(K, (x_ori_0 - s_0)) + v_opt

	# visualize the constraints
	if vis_flag == 0:
		for i in range(rmpc.horizon - 1):
			constraints_varlist[i] = \
				np.dot(F, np.asarray(sol["x"][rmpc.first_state_index.s[0] + i:rmpc.first_state_index.v[0]:rmpc.horizon])) \
				+ np.dot(G, np.asarray(sol["x"][rmpc.first_state_index.v[0] + i::(rmpc.horizon - 1)]))

		for i in range(np.shape(F)[0]):
			tmp_list = [0] * (rmpc.horizon - 1)
			for j in range(rmpc.horizon - 1):
				tmp_list[j] = float(constraints_varlist[j][i])
			constraint_var[i] = tmp_list
	vis_flag = 1

	# simulate forward
	# we assume that all disturbances have the same range
	disturbance = np.random.uniform(lb_w[0], ub_w[0], (np.shape(D)[1], 1))
	x_ori_0_next = np.dot(A, x_ori_0) + np.dot(B, u_opt) + np.dot(D, disturbance)
	s_0_next = np.dot(A, s_0) + np.dot(B, v_opt)
	x_ori_0 = x_ori_0_next
	s_0 = s_0_next
	time_index += 1

	vis_x.append(list(map(float,x_ori_0[0])))
	vis_y.append(list(map(float,x_ori_0[1])))

	sol = rmpc.RMPC(h, s_0, p_list, time_index)
	print(sol["f"])

# plot state trajectory
plt.figure()
plt.plot(vis_x, vis_y, 'o-', label='state trajectory')
plt.legend()
plt.grid()

'''
# plot constraints and corresponding bounds (indirect way)
plt.figure()
index = 4
plt.plot(constraint_var[index], 'k.-', label='control input')
plt.hlines(float(f[index]) - h[index], 0, N - 2, colors='r', label='input bounds')
plt.legend()
plt.grid()
'''

# plot constraints and corresponding bounds (direct way)
plt.figure()
plt.plot([i * float(1/G[4]) for i in constraint_var[4]], 'k.-', label='auxiliary control input')
plt.hlines(float(1/G[4])*(float(f[4]) - h[4]), 0, N - 2, colors='r')
plt.hlines(float(1/G[5])*(float(f[5]) - h[5]), 0, N - 2, colors='r')
plt.legend()
plt.grid()

plt.show()
print(end-start)
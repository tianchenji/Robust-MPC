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

	def __init__(self, A, B, D, F, G, P, K, V, f, lb, ub, r, N):
		'''
		A, B, D: system dynamic matrices
		F, G: constriant matrices
		K: fixed stabilizing feedback gain
		V: the matrix bounding W
		f: states and input constraints
		lb: lowerbound of the system noise
		ub: upperbound of the system noise
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
		self.V = V
		self.f = f
		self.w_lb = lb
		self.w_ub = ub
		self.r = r
		self.horizon = N
		self.first_state_index = FirstStateIndex(A=A, B=B, N=N)
		# number of optimization variables
		self.num_of_x = np.shape(self.A)[0] * self.horizon + np.shape(self.B)[1] * (self.horizon - 1)
		self.num_of_g = np.shape(self.A)[0] * self.horizon + np.shape(self.F)[0] * self.horizon

	def mRPI(self):
		'''
		mRPI returns the degree by which constraints are tightened
		'''

		n_x = np.shape(self.A)[0]
		n_w = np.shape(self.D)[1]
		n_h = np.shape(self.F)[0]
		h = [0]*n_h

		# calculating rho given r
		phi = self.A + np.dot(self.B, self.K)
		n_rho = np.shape(self.V)[0]
		mrho = [None]*n_rho

		# define optimization variables
		w = SX.sym('w', n_w)

		# define costs for linear programs in matrix form
		tmp = np.dot(self.V, np.dot(np.linalg.pinv(self.D), np.dot(np.linalg.matrix_power(phi, self.r), self.D)))
		rhocost = - mtimes(tmp, w)

		# solve n_rho linear programs
		for i in range(n_rho):
			nlp = {'x':w, 'f':rhocost[i]}
			opts = {}
			opts["ipopt.print_level"] = 0
			opts["print_time"] = 0
			solver = nlpsol('solver', 'ipopt', nlp, opts)
			x0 = [0] * n_w
			res = solver(x0=x0, lbx=self.w_lb, ubx=self.w_ub)
			mrho[i] = - res['f']
		rho = max(mrho)

		# calculate vector h by solving r * n_h linear programs
		for j in range(self.r):
			tmp = self.F + np.dot(self.G, self.K)
			hcost = - mtimes(np.dot(tmp, np.dot(np.linalg.matrix_power(phi, j), self.D)), w)
			for k in range(n_h):
				nlp = {'x':w, 'f':hcost[k]}
				opts = {}
				opts["ipopt.print_level"] = 0
				opts["print_time"] = 0
				solver = nlpsol('solver', 'ipopt', nlp, opts)
				x0 = [0] * n_w
				res = solver(x0=x0, lbx=self.w_lb, ubx=self.w_ub)
				h[k] += - res['f']
		h = [i/(1 - rho) for i in h]
		return h

	def RMPC(self, h, s_0):
		'''
		RMPC returns optimal control sequence
		'''

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
		for i in range(self.horizon):
			for j in range(np.shape(self.F)[0]):
				g_upperbound[ineq_cons_index + j * self.horizon + i] = self.f[j] - h[j]
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
A = np.array([[1.2,1.5],[0,1.3]])
B = np.array([[0],[1]])
D = np.array([[1,0],[0,1]])

# states and input constraints
F = np.array([[-0.1,0],[0.1,0],[0,-0.1],[0,0.1],[0,0],[0,0]])
G = np.array([[0],[0],[0],[0],[-1],[1]])
f = np.array([[1],[1],[1],[1],[1],[1]])

# bounds on noise
V = np.array([[10,0],[-10,0],[0,10],[0,-10]])
lb=[-0.1] * 2
ub=[0.1] * 2

# calculate LQR gain matrix
Q      = np.array([[1, 0], [0, 1]])
R      = np.array([[10]])
(P, K) = lqr(A, B, Q, R)

# mRPI parameters
r = 25

# prediction horizon
N = 20

s_0 = np.array([[-6.7],[1.4]])
x_ori_0 = s_0
threshold = pow(10, -8)
u_realized = []
J_value = []
vis_x = []
vis_y = []
vis_x.append(list(map(float,x_ori_0[0])))
vis_y.append(list(map(float,x_ori_0[1])))
rmpc = RMPC(A=A, B=B, D=D, F=F, G=G, P=P, K=K, V=V, f=f, lb=lb, ub=ub, r=r, N=N)
start = time.clock()
h = list(map(float, rmpc.mRPI()))
if max(h) >= 1:
	print("Robustly positively invariant set is empty! Cannot achieve robustness!")
	sys.exit()
sol = rmpc.RMPC(h, s_0)
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
	u_realized.append(list(map(float,u_opt)))
	J_value.append(list(map(float,np.asarray(sol["f"]))))

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
	disturbance = np.random.uniform(lb[0], ub[0], (np.shape(D)[1], 1))
	x_ori_0_next = np.dot(A, x_ori_0) + np.dot(B, u_opt) + np.dot(D, disturbance)
	s_0_next = np.dot(A, s_0) + np.dot(B, v_opt)
	x_ori_0 = x_ori_0_next
	s_0 = s_0_next

	vis_x.append(list(map(float,x_ori_0[0])))
	vis_y.append(list(map(float,x_ori_0[1])))

	sol = rmpc.RMPC(h, s_0)
	print(sol["f"])

# plot state trajectory
plt.figure()
plt.plot(vis_x, vis_y, '.-', label='realized closed-loop trajectory')
#plt.axis([-7, 0.5, -0.2, 1.6])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
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

# plot constraints and corresponding bounds on control inputs (direct way)
plt.figure()
plt.plot([i * float(1/G[4]) for i in constraint_var[4]], 'k.-', label='auxiliary control input planned at $t=0$')
plt.axhline(float(1/G[4])*(float(f[4]) - h[4]), color='r')
plt.axhline(float(1/G[5])*(float(f[5]) - h[5]), color='r')
plt.axis([0, N-2, -1.2, 1.2])
plt.xlabel('time steps ($t$)')
plt.legend()
plt.grid()

# plot realized optimal control inputs
plt.figure()
plt.plot(u_realized, '.-', label='realized optimal control inputs')
plt.axhline(f[4]/G[4], color='r')
plt.axhline(f[5]/G[5], color='r')
plt.axis([0, len(u_realized)-1, -1.4, 1.4])
plt.xlabel('time steps ($t$)')
plt.legend()
plt.grid()

# plot optimal cost
plt.figure()
plt.plot(J_value, '.-', label='optimal cost value')
plt.xlabel('time steps ($t$)')
plt.ylabel(r'$J^*$')
plt.legend()
plt.grid()

plt.show()
print(end-start)
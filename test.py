from casadi import *
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
'''
V = np.matrix([[20,0],[-20,0],[0,20],[0,-20]])
D = np.matrix([[-1,0],[0,-1]])
A = np.matrix([[0.5,0],[0.5,1]])
B = np.matrix([[1],[0]])
K = np.matrix([[-0.89,-0.78]])
F = np.matrix([[-10/3,0],[-10/7,0],[0,-2],[0,2],[0,0],[0,0]])
G = np.matrix([[0],[0],[0],[0],[-10/3],[5]])
phi = A + B*K
r = 6
n_rho = np.shape(V)[0]
n_h = np.shape(F)[0]
mrho = [None]*n_rho
h = [0]*n_h

start = time.clock()
w = SX.sym('w', 2)
mcost = - mtimes((V * np.linalg.pinv(D) * numpy.linalg.matrix_power(phi, r) * D), w)
for i in range(n_rho):
	nlp = {'x':w, 'f':mcost[i]}
	solver = nlpsol('solver', 'ipopt', nlp)
	x0=[0] * 2
	lbx=[-0.05] * 2
	ubx=[0.05] * 2
	res = solver(x0=x0, lbx=lbx, ubx=ubx)
	mrho[i] = - res['f']
rho = max(mrho)
end = time.clock()

for j in range(r):
	hcost = - mtimes(((F + G*K)*numpy.linalg.matrix_power(phi, j)*D), w)
	for k in range(n_h):
		nlp = {'x':w, 'f':hcost[k]}
		solver = nlpsol('solver', 'ipopt', nlp)
		x0=[0] * 2
		lbx=[-0.05] * 2
		ubx=[0.05] * 2
		res = solver(x0=x0, lbx=lbx, ubx=ubx)
		h[k] += - res['f']
h = [i/(1 - rho) for i in h]
print(end-start)
print(h)
'''
'''
class Mytry:
	def __init__(self, n):
		self.index = n

class Test:
	def __init__(self, n):
		self.a = 0
		self.b = self.a + n
		self.mytry = Mytry(n=n)
	def add(self, x):
		self.b = self.mytry.index + x
test = Test(n = 17)
test.add(2)
print(test.b)
'''

# projection
A = np.array([[0.8, 0.3], [0.3, 2.7]])
L = np.linalg.cholesky(A)
x0 = np.array([[0], [0]])

v1 = np.array([[1], [0]])
v2 = np.array([[0], [1]])
c = np.array([[1.2], [0.4]])

s1 = np.dot(np.transpose(v1), c - x0) / np.dot(np.transpose(v1), v1)
s2 = np.dot(np.transpose(v2), c - x0) / np.dot(np.transpose(v2), v2)
w1 = np.dot(np.linalg.inv(L), v1) / np.dot(np.transpose(v1), v1)
w2 = np.dot(np.linalg.inv(L), v2) / np.dot(np.transpose(v2), v2)
norm_w1 = np.linalg.norm(w1) * (3 / sqrt(10))
norm_w2 = np.linalg.norm(w2) * (3 / sqrt(10))
s1_min = float(s1 - norm_w1)
s1_max = float(s1 + norm_w1)
s2_min = float(s2 - norm_w2)
s2_max = float(s2 + norm_w2)

# plot
ell = Ellipse(xy=[1.2,0.4], width=1.1449, height=2.1854, angle=81.2395, edgecolor='k', facecolor='none')
fig, ax = plt.subplots()
ax.add_patch(ell)
ax.set_aspect('equal')
ax.autoscale()
plt.grid()
plt.plot([s1_min, s1_max], [0, 0], 'or')
plt.plot([0, 0], [s2_min, s2_max], 'or')
plt.vlines(s1_min, -0.3, 1.1, colors='r', linestyles='dashed')
plt.vlines(s1_max, -0.3, 1.1, colors='r', linestyles='dashed')
plt.hlines(s2_min, 0.0, 2.5, colors='r', linestyles='dashed')
plt.hlines(s2_max, 0.0, 2.5, colors='r', linestyles='dashed')
plt.show()
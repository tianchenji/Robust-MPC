import matplotlib.pyplot as plt
import numpy as np
import pickle

# plot 1
RMPC_traj = pickle.load(open("RMPC_traj.pkl", "rb"))
RMPCSE_traj = pickle.load(open("RMPCSE_traj.pkl", "rb"))

# plot 2
RMPC_planned_input = pickle.load(open("RMPC_planned_input.pkl", "rb"))
RMPCSE_planned_input = pickle.load(open("RMPCSE_planned_input.pkl", "rb"))

# plot 3
u_realized_RMPC = pickle.load(open("RMPC_realized_input.pkl", "rb"))
u_realized_RMPCSE = pickle.load(open("RMPCSE_realized_input.pkl", "rb"))

# plot 4
J_value_average_RMPC = pickle.load(open("J_value_average_RMPC.pkl", "rb"))
J_value_average_RMPCSE = pickle.load(open("J_value_average_RMPCSE.pkl", "rb"))

RMPC_traj = [list(i) for i in zip(*RMPC_traj)]
RMPCSE_traj = [list(i) for i in zip(*RMPCSE_traj)]
RMPC_planned_input = [list(i) for i in zip(*RMPC_planned_input)]
RMPCSE_planned_input = [list(i) for i in zip(*RMPCSE_planned_input)]

# realized state trajectory
plt.figure()
plt.plot(RMPC_traj[0], RMPC_traj[1], color='royalblue', marker='.', label='no measurement noise')
plt.plot(RMPCSE_traj[0], RMPCSE_traj[1], color='black', marker='.', label=r'measurement noise $|| \xi ||_\infty \leq 0.01$')
plt.title('realized state trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis([-7, 1, -0.2, 2.0])
plt.legend()
plt.grid()

# planned auxiliary inputs at t = 0
plt.figure()
plt.plot(RMPC_planned_input[0], color='royalblue', marker='.', label='no measurement noise')
plt.plot(RMPC_planned_input[1], RMPC_planned_input[2], color='red')
plt.plot(RMPC_planned_input[1], RMPC_planned_input[3], color='red')
plt.plot(RMPCSE_planned_input[0], color='black', linestyle='--', marker='.', label=r'measurement noise $|| \xi ||_\infty \leq 0.01$')
plt.plot(RMPCSE_planned_input[1], RMPCSE_planned_input[2], linestyle='--', color='red')
plt.plot(RMPCSE_planned_input[1], RMPCSE_planned_input[3], linestyle='--', color='red')
plt.fill_between(RMPCSE_planned_input[1], RMPCSE_planned_input[2], RMPCSE_planned_input[3], color='grey', alpha='0.2')
plt.xlabel('time steps ($t$)')
plt.xticks(np.arange(0,19,2))
plt.axis([0, 19, -1.2, 1.2])
plt.legend(loc=1)
plt.title(r'planned auxiliary inputs at $t=0$')
plt.grid()

# realized input u
while len(u_realized_RMPC) < len(u_realized_RMPCSE):
	u_realized_RMPC.append([0])

plt.figure()
plt.plot(u_realized_RMPC, color='royalblue', marker='.', label='no measurement noise')
plt.plot(u_realized_RMPCSE, color='black', marker='.', label=r'measurement noise $|| \xi ||_\infty \leq 0.01$')
plt.axhline(-1, color='r', linestyle='--', label='control input constraints')
plt.axhline(1, color='r', linestyle='--')
plt.xlabel('time steps ($t$)')
plt.title(r'realized input $u(t)$')
plt.axis([0, 25, -1.2, 1.2])
plt.legend()
plt.grid()

# average optimal cost value
plt.figure()
plt.plot(J_value_average_RMPC[0:11], color='royalblue', marker='.', label='no measurement noise')
plt.plot(J_value_average_RMPCSE[0:11], color='black', marker='.', label=r'measurement noise $|| \xi ||_\infty \leq 0.01$')
plt.xlabel('time steps ($t$)')
plt.ylabel(r'$J^*$')
plt.title("average optimal cost value over 100 sample trajectories")
plt.axis([0, 10, -5, 160])
plt.legend()
plt.grid()

plt.show()

'''
#vis_traj = list(zip(vis_x, vis_y))
#pickle.dump(vis_traj, open( "vis_RMPC_success.pkl", "wb" ))
'''
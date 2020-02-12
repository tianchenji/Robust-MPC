import matplotlib.pyplot as plt
import pickle

# plot 1
vis_traj_fail_1 = pickle.load(open("vis_RMPC_fail_1.pkl", "rb"))
vis_traj_fail_2 = pickle.load(open("vis_RMPC_fail_2.pkl", "rb"))
vis_traj_fail_3 = pickle.load(open("vis_RMPC_fail_3.pkl", "rb"))
vis_traj_success = pickle.load(open("vis_RMPC_success.pkl", "rb"))

# plot 2
u_realized_fail_03 = pickle.load(open("u_realized_fail_0.3.pkl", "rb"))
u_realized_fail_05 = pickle.load(open("u_realized_fail_0.5.pkl", "rb"))
u_realized_success = pickle.load(open("u_realized_success.pkl", "rb"))

# plot 3
RMPC_traj = pickle.load(open("RMPC_traj.pkl", "rb"))
RMPCSE_traj = pickle.load(open("RMPCSE_traj.pkl", "rb"))

# plot 4
RMPC_planned_input = pickle.load(open("RMPC_planned_input.pkl", "rb"))
RMPCSE_planned_input = pickle.load(open("RMPCSE_planned_input.pkl", "rb"))

# plot 5
u_realized_RMPC = pickle.load(open("RMPC_realized_input.pkl", "rb"))
u_realized_RMPCSE = pickle.load(open("RMPCSE_realized_input.pkl", "rb"))

# plot 6
J_value_average_RMPC = pickle.load(open("J_value_average_RMPC.pkl", "rb"))
J_value_average_RMPCSE = pickle.load(open("J_value_average_RMPCSE.pkl", "rb"))

vis_traj_fail_1 = [list(i) for i in zip(*vis_traj_fail_1)]
vis_traj_fail_2 = [list(i) for i in zip(*vis_traj_fail_2)]
vis_traj_fail_3 = [list(i) for i in zip(*vis_traj_fail_3)]
vis_traj_success = [list(i) for i in zip(*vis_traj_success)]
RMPC_traj = [list(i) for i in zip(*RMPC_traj)]
RMPCSE_traj = [list(i) for i in zip(*RMPCSE_traj)]
RMPC_planned_input = [list(i) for i in zip(*RMPC_planned_input)]
RMPCSE_planned_input = [list(i) for i in zip(*RMPCSE_planned_input)]

# failure case of state trajectory
plt.figure()
plt.plot(vis_traj_success[0], vis_traj_success[1], color='green', marker='.', label='no noise')
plt.plot(vis_traj_fail_1[0], vis_traj_fail_1[1], color='black', marker='.', label='failure case 1 with noise')
plt.plot(vis_traj_fail_2[0], vis_traj_fail_2[1], color='dimgray', marker='.', label='failure case 2 with noise')
plt.plot(vis_traj_fail_3[0], vis_traj_fail_3[1], color='saddlebrown', marker='.', label='failure case 3 with noise')
plt.axvline(-10, color='r', linestyle='--', label='state constraints')
plt.axvline(10, color='r', linestyle='--')
plt.title('realized state trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis([-12, 14, -4, 4])
plt.legend()
plt.grid()

# failure case of control inputs
plt.figure()
plt.plot(u_realized_success, color='green', marker='.', linewidth=1.0, label=r'measurement noise $|| \xi ||_\infty \leq 0.1$')
plt.plot(u_realized_fail_03, color='royalblue', marker='.', linewidth=1.0, label=r'measurement noise $|| \xi ||_\infty \leq 0.3$')
plt.plot(u_realized_fail_05, color='black', marker='.', linewidth=1.0, label=r'measurement noise $|| \xi ||_\infty \leq 0.5$')
plt.axhline(-1, color='r', linestyle='--', label='control input constraints')
plt.axhline(1, color='r', linestyle='--')
plt.xlabel('time steps ($t$)')
plt.title(r'realized input $u(t)$')
plt.axis([0, 24, -1.8, 1.8])
plt.legend()
plt.grid()

# realized state trajectory
plt.figure()
plt.plot(RMPC_traj[0], RMPC_traj[1], color='royalblue', marker='.', label='no measurement noise')
plt.plot(RMPCSE_traj[0], RMPCSE_traj[1], color='black', marker='.', label=r'measurement noise $|| \xi ||_\infty \leq 0.01$')
plt.title('realized state trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis([-7, 1, -0.2, 1.8])
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
plt.axis([0, 18, -0.8, 0.8])
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
plt.axis([0, 27, -1.2, 1.2])
plt.legend()
plt.grid()

# average optimal cost value
plt.figure()
plt.plot(J_value_average_RMPC[0:11], color='royalblue', marker='.', label='no measurement noise')
plt.plot(J_value_average_RMPCSE[0:11], color='black', marker='.', label=r'measurement noise $|| \xi ||_\infty \leq 0.01$')
plt.xlabel('time steps ($t$)')
plt.ylabel(r'$J^*$')
plt.title("average optimal cost value over 100 sample trajectories")
plt.axis([0, 10, -5, 180])
plt.legend()
plt.grid()

plt.show()

'''
#vis_traj = list(zip(vis_x, vis_y))
#pickle.dump(vis_traj, open( "vis_RMPC_success.pkl", "wb" ))

from RMPC import main_loop
import pickle

J_value_sum = [[0]] * 20
num = 100

for i in range(num):
	J_value_tmp = main_loop()[0:20]
	J_value_sum = [[sum(x) for x in zip(J_value_sum[i], J_value_tmp[i])] for i in range(len(J_value_sum))]

J_value_average = [[x / num for x in J_value_sum[i]] for i in range(len(J_value_sum))]

pickle.dump(J_value_average, open( "J_value_average_RMPC.pkl", "wb"))
'''
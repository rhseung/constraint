import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import inv
from numpy.typing import NDArray, ArrayLike

# FIXME: 제대로 시뮬레이션 안됨

g = np.array([0, -9.81])

class Object:
	def __init__(self, p: ArrayLike, theta: float, m: float, r: float) -> None:
		self.p = np.array(p)
		self.v = np.zeros((2,))
		self.theta = theta
		self.omega = 0
		self.m = m
		self.r = r
		self.f = m * g
		self.tau = 0

	@property
	def I(self) -> float:
		return (1 / 6) * self.m * (2 * self.r) ** 2

	@property
	def M(self) -> NDArray[np.float32]:
		return np.diag((self.m, self.m, self.I))

	@property
	def q(self) -> NDArray[np.float32]:
		return np.array([*self.p, self.theta])

	@property
	def q_dot(self) -> NDArray[np.float32]:
		return np.array([*self.v, self.omega])

	@property
	def F(self) -> NDArray[np.float32]:
		return np.array([*self.f, self.tau])


obj1 = Object(
	p=(6, -4),
	theta=np.pi / 3,
	m=3,
	r=0.5
)

obj2 = Object(
	p=(1, 4),
	theta=np.pi / 4,
	m=2,
	r=0.4
)

Q = np.array([*obj1.q, *obj2.q])
Q_dot = np.array([*obj1.q_dot, *obj2.q_dot])
M = np.zeros((6, 6))
M[:3, :3] = obj1.M
M[3:, 3:] = obj2.M
F_ext = np.array([*obj1.F, *obj2.F])

L1 = 5
L2 = 4


def C1(Q: NDArray[np.float32]) -> float:
	return Q[0] ** 2 + Q[1] ** 2 - L1 ** 2


def J1(Q: NDArray[np.float32]) -> NDArray[np.float32]:
	ret = np.zeros((len(Q),))
	ret[0] = 2 * Q[0]
	ret[1] = 2 * Q[1]
	return ret


def J1_dot(Q: NDArray[np.float32]) -> NDArray[np.float32]:
	ret = np.zeros((len(Q),))
	ret[0] = 2
	ret[1] = 2
	return ret


def C2(Q: NDArray[np.float32]) -> float:
	return Q[3] ** 2 + Q[4] ** 2 - L2 ** 2


def J2(Q: NDArray[np.float32]) -> NDArray[np.float32]:
	ret = np.zeros((len(Q),))
	ret[3] = 2 * Q[3]
	ret[4] = 2 * Q[4]
	return ret


def J2_dot(Q: NDArray[np.float32]) -> NDArray[np.float32]:
	ret = np.zeros((len(Q),))
	ret[3] = 2
	ret[4] = 2
	return ret


def C(Q: NDArray[np.float32]) -> NDArray[np.float32]:
	return np.array([C1(Q), C2(Q)])


def J(Q: NDArray[np.float32]) -> NDArray[np.float32]:
	return np.array([J1(Q), J2(Q)])


def J_dot(Q: NDArray[np.float32]) -> NDArray[np.float32]:
	return np.array([J1_dot(Q), J2_dot(Q)])


def lamb(Q: NDArray[np.float32], Q_dot: NDArray[np.float32], F_ext: NDArray[np.float32]) -> NDArray[np.float32]:
	j = J(Q)
	W = inv(M)
	j_dot = J_dot(Q)

	lhs = j @ W @ j.T
	rhs = -(j_dot @ Q_dot + j @ W @ F_ext)

	return np.linalg.solve(lhs, rhs)


def F_c(Q: NDArray[np.float32], Q_dot: NDArray[np.float32], F_ext: NDArray[np.float32]) -> NDArray[np.float32]:
	j = J(Q)
	lambda_ = lamb(Q, Q_dot, F_ext)

	return j.T @ lambda_


# Simulation parameters
dt = 0.01
num_steps = 3000
substeps = 1

# To store the simulation results
positions1 = []
positions2 = []
angles1 = []
angles2 = []

# Simulation loop
for _ in range(num_steps):
	for _ in range(substeps):
		f_c = F_c(Q, Q_dot, F_ext)
		Q_dot += inv(M) @ (F_ext + f_c) * dt
		Q += Q_dot * dt

	# Store positions and angles
	positions1.append(Q[:2].copy())
	positions2.append(Q[3:5].copy())
	angles1.append(Q[2])
	angles2.append(Q[5])

positions1 = np.array(positions1)
positions2 = np.array(positions2)
angles1 = np.array(angles1)
angles2 = np.array(angles2)

# Animation using Matplotlib
fig, ax = plt.subplots()
ax.set_xlim(-12, 12)
ax.set_ylim(-16, 8)
ax.set_aspect('equal')
line1, = ax.plot([], [], 'bo-', lw=2)
line2, = ax.plot([], [], 'ro-', lw=2)
rect1, = ax.plot([], [], 's-', lw=2, markersize=2)
rect2, = ax.plot([], [], 's-', lw=2, markersize=2)
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top', fontsize=7,
			   bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))


def init():
	line1.set_data([], [])
	line2.set_data([], [])
	rect1.set_data([], [])
	rect2.set_data([], [])
	text.set_text('')
	return line1, line2, rect1, rect2, text


def update(frame):
	x1, y1 = positions1[frame]
	x2, y2 = positions2[frame]
	theta1 = angles1[frame]
	theta2 = angles2[frame]

	line1.set_data([0, x1], [0, y1])
	line2.set_data([x1, x2], [y1, y2])

	r1, r2 = 0.5, 0.4
	corners1 = np.array([[r1, r1], [-r1, r1], [-r1, -r1], [r1, -r1], [r1, r1]])
	corners2 = np.array([[r2, r2], [-r2, r2], [-r2, -r2], [r2, -r2], [r2, r2]])

	R1 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])
	R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])

	rotated_corners1 = (R1 @ corners1.T).T + np.array([x1, y1])
	rotated_corners2 = (R2 @ corners2.T).T + np.array([x2, y2])

	rect1.set_data(rotated_corners1[:, 0], rotated_corners1[:, 1])
	rect2.set_data(rotated_corners2[:, 0], rotated_corners2[:, 1])

	length1 = np.sqrt(x1 ** 2 + y1 ** 2)
	length2 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	text.set_text(f'String Lengths: L1={length1:.2f}, L2={length2:.2f}')

	return line1, line2, rect1, rect2, text


ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=2)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Double Pendulum Animation with Rotation')
plt.grid()
plt.show()

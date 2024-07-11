import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
L1 = 5.0  # Length of first rod
L2 = 5.0  # Length of second rod
M1 = 1.0  # Mass of first pendulum
M2 = 1.0  # Mass of second pendulum
G = -9.8  # Gravitational acceleration

# Initial conditions
theta1 = np.pi / 3  # Initial angle for first pendulum
theta2 = 0.0        # Initial angle for second pendulum
omega1 = 0.0        # Initial angular velocity for first pendulum
omega2 = 0.0        # Initial angular velocity for second pendulum

# Time step
dt = 0.01

# Equations of motion
def derivatives(state, t):
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1
    
    denom1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    denom2 = (L2 / L1) * denom1

    dtheta1_dt = omega1
    dtheta2_dt = omega2
    
    domega1_dt = (M2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
                  M2 * G * np.sin(theta2) * np.cos(delta) +
                  M2 * L2 * omega2 * omega2 * np.sin(delta) -
                  (M1 + M2) * G * np.sin(theta1)) / denom1

    domega2_dt = (- M2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
                  (M1 + M2) * G * np.sin(theta1) * np.cos(delta) -
                  (M1 + M2) * L1 * omega1 * omega1 * np.sin(delta) -
                  (M1 + M2) * G * np.sin(theta2)) / denom2

    return dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt

# Solve the equations of motion using the 4th order Runge-Kutta method
def solve_double_pendulum(theta1, omega1, theta2, omega2, dt, steps):
    state = np.array([theta1, omega1, theta2, omega2])
    states = np.zeros((steps, 4))
    for i in range(steps):
        states[i] = state
        k1 = np.array(derivatives(state, i * dt))
        k2 = np.array(derivatives(state + 0.5 * dt * k1, i * dt + 0.5 * dt))
        k3 = np.array(derivatives(state + 0.5 * dt * k2, i * dt + 0.5 * dt))
        k4 = np.array(derivatives(state + dt * k3, i * dt + dt))
        state += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return states

# Number of time steps
steps = 5000

# Get the solutions
states = solve_double_pendulum(theta1, omega1, theta2, omega2, dt, steps)

# Extract angles
theta1s = states[:, 0]
theta2s = states[:, 2]

# Convert to Cartesian coordinates
x1 = L1 * np.sin(theta1s)
y1 = L1 * np.cos(theta1s)
x2 = x1 + L2 * np.sin(theta2s)
y2 = y1 + L2 * np.cos(theta2s)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], 'r-', lw=1)  # Trace for the path of the second pendulum bob

def init():
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace

def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    trace.set_data(x2[:frame], y2[:frame])
    return line, trace

ani = FuncAnimation(fig, update, frames=range(steps), init_func=init, blit=True, interval=2)

plt.show()

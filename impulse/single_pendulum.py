import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import inv

# 초기 값 설정
p = np.array([6, -4])
theta = np.pi / 3

q = np.array([*p, theta])

m = 3
r = 0.5  # 사각형의 한 변의 절반 길이

# 사각형의 관성 모멘트 계산
I = (1/6) * m * (2*r)**2

M = np.diag((m, m, I))

g = np.array([0.0, -9.8])
f = m * g
tau = 0

F = np.array([*f, tau])

L = 5

def C(q):
    x, y, theta = q
    return x**2 + y**2 - L**2

def J(q):
    x, y, theta = q
    return np.array([2*x, 2*y, 0])

dt = 0.01
b = 0.5

def lamb(q, q_dot):
    c = C(q)
    j = J(q)
    
    lhs = j @ inv(M) @ j.T
    rhs = -(j @ (q_dot + inv(M) @ F * dt) + b * c / dt)
    
    return rhs / lhs

def P_c(q, q_dot):
    j = J(q)
    lambda_ = lamb(q, q_dot)
    
    return j.T * lambda_

p_dot = np.array([0, 1])
omega = np.pi / 2
q_dot = np.array([*p_dot, omega])

# 위치 기록을 위한 리스트 초기화
positions = []
angles = []

# 시뮬레이션 루프
for i in range(5000):
    positions.append(q[:2].copy())  # 현재 위치 기록
    angles.append(q[2])  # 현재 각도 기록
    P = P_c(q, q_dot)
    q_dot = q_dot + inv(M) @ (F * dt + P)
    q = q + q_dot * dt
    
    if i % 100 == 0:
        print(f'Iteration {i}: q = {q}, q_dot = {q_dot}')

# 위치 리스트를 배열로 변환
positions = np.array(positions)
angles = np.array(angles)

# 애니메이션 설정
fig, ax = plt.subplots()
ax.set_xlim(-6, 6)
ax.set_ylim(-8, 3)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2)  # 진자의 줄
rect, = ax.plot([], [], 's-', lw=2, markersize=2)  # 사각형 모양 진자
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

def init():
    line.set_data([], [])
    rect.set_data([], [])
    text.set_text('')
    return line, rect, text,

def update(frame):
    x = positions[frame, 0]
    y = positions[frame, 1]
    theta = angles[frame]
    
    # 진자의 줄 설정
    line.set_data([0, x], [0, y])
    
    # 사각형의 네 꼭지점 계산
    corners = np.array([
        [r, r],
        [-r, r],
        [-r, -r],
        [r, -r],
        [r, r]
    ])
    
    # 회전 행렬 적용
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    rotated_corners = (R @ corners.T).T
    
    # 위치 이동
    rotated_corners += np.array([x, y])
    
    rect.set_data(rotated_corners[:, 0], rotated_corners[:, 1])
    
    # 줄 길이 계산 및 표시
    length = np.sqrt(x**2 + y**2)
    text.set_text(f'String Length: {length:.2f}')
    
    return line, rect, text,

ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, interval=2)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Pendulum Animation with Rotation')
plt.grid()
plt.show()

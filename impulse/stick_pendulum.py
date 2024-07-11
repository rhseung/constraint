import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import inv

# 초기 값 설정
theta = np.pi / 3  # 초기 각도
omega = 0.0  # 초기 각속도
L = 5  # 막대의 길이
r = 0.5  # 막대의 반 두께

m = 3
I = (1/3) * m * L**2  # 막대의 관성 모멘트 (고정된 끝점을 기준으로)

q = np.array([0, -L, theta])
q_dot = np.array([0.0, 0.0, omega])

g = 9.8  # 중력 가속도
F_ext = np.array([0, -m * g, 0])  # 중력에 의한 힘 (타우는 0으로 초기화)

dt = 0.01
b = 0.0  # 감쇠 계수

def C(q):
    x, y, theta = q
    return np.array([x**2 + y**2 - L**2])

def J(q):
    x, y, theta = q
    return np.array([[2*x, 2*y, 0]])

def lamb(q, q_dot):
    c = C(q)
    j = J(q)
    lhs = j @ inv(M) @ j.T
    rhs = -(j @ (q_dot + inv(M) @ F_ext * dt) + b * c / dt)
    return rhs / lhs

def P_c(q, q_dot):
    j = J(q)
    lambda_ = lamb(q, q_dot)
    return j.T * lambda_

p_dot = np.array([0, 0])  # 초기 선속도
omega = 0.0  # 초기 각속도
q_dot = np.array([*p_dot, omega])

# 위치 기록을 위한 리스트 초기화
positions = []
angles = []

# 시뮬레이션 루프
for i in range(1000):
    positions.append(q[:2].copy())  # 현재 위치 기록
    angles.append(q[2])  # 현재 각도 기록
    P = P_c(q, q_dot)
    q_dot = q_dot + inv(M) @ (F_ext * dt + P)
    q = q + q_dot * dt
    
    if i % 100 == 0:
        print(f'Iteration {i}: q = {q}, q_dot = {q_dot}')

# 위치 리스트를 배열로 변환
positions = np.array(positions)
angles = np.array(angles)

# 애니메이션 설정
fig, ax = plt.subplots()
ax.set_xlim(-L - 1, L + 1)
ax.set_ylim(-L - 1, L + 1)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2)  # 막대의 줄
mass_start, = ax.plot([], [], 'o', lw=2)  # 시작점의 질량
mass_end, = ax.plot([], [], 'o', lw=2)  # 끝점의 질량
rect, = ax.plot([], [], lw=2)  # 사각형 모양 막대
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

def init():
    line.set_data([], [])
    mass_start.set_data([], [])
    mass_end.set_data([], [])
    rect.set_data([], [])
    text.set_text('')
    return line, mass_start, mass_end, rect, text,

def update(frame):
    x_end = positions[frame, 0]
    y_end = positions[frame, 1]
    theta = angles[frame]
    
    # 막대의 줄 설정
    line.set_data([0, x_end], [0, y_end])
    
    # 시작점의 질량 위치 설정
    mass_start.set_data([0], [0])
    
    # 끝점의 질량 위치 설정
    mass_end.set_data([x_end], [y_end])
    
    # 사각형의 네 꼭지점 계산
    corners = np.array([
        [-L / 2, -r],
        [L / 2, -r],
        [L / 2, r],
        [-L / 2, r],
        [-L / 2, -r]
    ])
    
    # 회전 행렬 적용
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    rotated_corners = (R @ corners.T).T
    
    # 위치 이동
    rotated_corners[:, 0] += x_end
    rotated_corners[:, 1] += y_end
    
    rect.set_data(rotated_corners[:, 0], rotated_corners[:, 1])
    
    # 줄 길이 계산 및 표시
    length = np.sqrt(x_end**2 + y_end**2)
    text.set_text(f'String Length: {length:.2f}')
    
    return line, mass_start, mass_end, rect, text,

ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, interval=20)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Pendulum Animation with Rotation')
plt.grid()
plt.show()

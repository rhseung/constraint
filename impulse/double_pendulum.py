import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import inv

# FIXME: 부자연스러움

# 초기 값 설정
p1 = np.array([6, -4])
p2 = np.array([1, 4])
theta1 = np.pi / 3
theta2 = np.pi / 4

q1 = np.array([*p1, theta1])
q2 = np.array([*p2, theta2])  # q2는 q1의 끝점을 기준으로 상대 좌표를 저장

m1 = 3
m2 = 2
r1 = 1  # 사각형의 한 변의 절반 길이
r2 = 0.8  # 사각형의 한 변의 절반 길이

# 사각형의 관성 모멘트 계산
I1 = (1/6) * m1 * (2*r1)**2
I2 = (1/6) * m2 * (2*r2)**2

M1 = np.diag((m1, m1, I1))
M2 = np.diag((m2, m2, I2))

g = np.array([0.0, -9.8])
f1 = m1 * g
f2 = m2 * g
tau1 = 0
tau2 = 0

F1 = np.array([*f1, tau1])
F2 = np.array([*f2, tau2])

L1 = 5
L2 = 4

def C1(q1):
    x, y, theta = q1
    return x**2 + y**2 - L1**2

def C2(q2):
    x, y, theta = q2
    return x**2 + y**2 - L2**2

def J1(q1):
    x, y, theta = q1
    return np.array([2*x, 2*y, 0])

def J2(q2):
    x, y, theta = q2
    return np.array([2*x, 2*y, 0])

dt = 0.01
b = 0.5
substeps = 4

def lamb(q, q_dot, M, F, C, J, b):
    c = C(q)
    j = J(q)
    
    lhs = j @ inv(M) @ j.T
    rhs = -(j @ (q_dot + inv(M) @ F * dt) + b * c / dt)
    
    return rhs / lhs

def P_c(q, q_dot, M, F, C, J, b):
    j = J(q)
    lambda_ = lamb(q, q_dot, M, F, C, J, b)
    
    return j.T * lambda_

p1_dot = np.array([0, 1])
p2_dot = np.array([0, 1])
omega1 = np.pi / 2
omega2 = np.pi / 3
q1_dot = np.array([*p1_dot, omega1])
q2_dot = np.array([*p2_dot, omega2])

# 위치 기록을 위한 리스트 초기화
positions1 = []
positions2 = []
angles1 = []
angles2 = []

# 시뮬레이션 루프
for i in range(3000):
    for _ in range(substeps):
        # 첫 번째 진자의 외력과 구속 충격량 적용
        P1 = P_c(q1, q1_dot, M1, F1, C1, J1, b)
        q1_dot = q1_dot + inv(M1) @ (F1 * dt + P1)
        q1 = q1 + q1_dot * dt
        
        # 두 번째 진자의 외력과 구속 충격량 적용
        P2 = P_c(q2, q2_dot, M2, F2, C2, J2, b)
        q2_dot = q2_dot + inv(M2) @ (F2 * dt + P2)
        q2 = q2 + q2_dot * dt
        
    # 두 번째 진자의 위치를 첫 번째 진자의 끝점 기준으로 저장
    positions1.append(q1[:2].copy())  # 현재 위치 기록
    positions2.append((q1[:2] + q2[:2]).copy())  # 현재 위치 기록
    angles1.append(q1[2])  # 현재 각도 기록
    angles2.append(q2[2])  # 현재 각도 기록

    if i % 100 == 0:
        print(f'Iteration {i}')

# 위치 리스트를 배열로 변환
positions1 = np.array(positions1)
positions2 = np.array(positions2)
angles1 = np.array(angles1)
angles2 = np.array(angles2)

# 애니메이션 설정
fig, ax = plt.subplots()
ax.set_xlim(-12, 12)
ax.set_ylim(-16, 8)
ax.set_aspect('equal')
line1, = ax.plot([], [], 'o-', lw=2)  # 첫 번째 진자의 줄
line2, = ax.plot([], [], 'o-', lw=2)  # 두 번째 진자의 줄
rect1, = ax.plot([], [], 's-', lw=2, markersize=2)  # 첫 번째 사각형 모양 진자
rect2, = ax.plot([], [], 's-', lw=2, markersize=2)  # 두 번째 사각형 모양 진자
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top', fontsize=7, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    rect1.set_data([], [])
    rect2.set_data([], [])
    text.set_text('')
    return line1, line2, rect1, rect2, text,

def update(frame):
    x1 = positions1[frame, 0]
    y1 = positions1[frame, 1]
    x2 = positions2[frame, 0]
    y2 = positions2[frame, 1]
    theta1 = angles1[frame]
    theta2 = angles2[frame]
    
    # 첫 번째 진자의 줄 설정
    line1.set_data([0, x1], [0, y1])
    # 두 번째 진자의 줄 설정
    line2.set_data([x1, x2], [y1, y2])
    
    # 첫 번째 사각형의 네 꼭지점 계산
    corners1 = np.array([
        [r1, r1],
        [-r1, r1],
        [-r1, -r1],
        [r1, -r1],
        [r1, r1]
    ])
    
    # 두 번째 사각형의 네 꼭지점 계산
    corners2 = np.array([
        [r2, r2],
        [-r2, r2],
        [-r2, -r2],
        [r2, -r2],
        [r2, r2]
    ])
    
    # 첫 번째 회전 행렬 적용
    R1 = np.array([
        [np.cos(theta1), -np.sin(theta1)],
        [np.sin(theta1), np.cos(theta1)]
    ])
    rotated_corners1 = (R1 @ corners1.T).T
    
    # 두 번째 회전 행렬 적용
    R2 = np.array([
        [np.cos(theta2), -np.sin(theta2)],
        [np.sin(theta2), np.cos(theta2)]
    ])
    rotated_corners2 = (R2 @ corners2.T).T
    
    # 위치 이동
    rotated_corners1 += np.array([x1, y1])
    rotated_corners2 += np.array([x2, y2])
    
    rect1.set_data(rotated_corners1[:, 0], rotated_corners1[:, 1])
    rect2.set_data(rotated_corners2[:, 0], rotated_corners2[:, 1])
    
    # 줄 길이 계산 및 표시
    length1 = np.sqrt(x1**2 + y1**2)
    length2 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    text.set_text(f'String Lengths: L1={length1:.2f}, L2={length2:.2f}')
    
    return line1, line2, rect1, rect2, text,

ani = FuncAnimation(fig, update, frames=len(positions1), init_func=init, blit=True, interval=2)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Double Pendulum Animation with Rotation')
plt.grid()
plt.show()
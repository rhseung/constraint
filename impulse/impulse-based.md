# Impulse Based Dynamics

위치 상태 벡터 $\mathbf{q}$는 물체의 변위와 관련된 정보를 가진다.

```math
\mathbf{q} = \begin{pmatrix} \mathbf{p} \\ \theta \end{pmatrix}
```

$\mathbf{{p}}$는 d차원에서 원소가 d개이므로, $\mathbf{{q}}$는 d+1 차원이다.

---

질량 행렬 $\mathbf{{M}}$은 물체의 질량과 관련된 정보를 가진다. 2차원인 경우 아래와 같다.

```math
\mathbf{M} = \begin{bmatrix} m \cdot \mathbf{1_{d \times d}} & 0 \\ 0 & I \end{bmatrix} = \begin{bmatrix} m & 0 & 0 \\ 0 & m & 0 \\ 0 & 0 & I \end{bmatrix}
```

$I$는 각 물체의 회전 관성이고, $m$은 질량이다.  
3차원부터는 회전 관성이 스칼라가 아니게 되어 다르게 할 필요가 있으나 조사가 필요하다.

---

힘 상태 벡터 $\mathbf{{F}}$는 n개의 물체의 힘과 관련된 정보를 가진다.

```math
\mathbf{F} = \begin{pmatrix} \mathbf{f} \\ \tau \end{pmatrix}
```

$\mathbf{q}$와 동일하게 $\mathbf{F}$는 n+1 차원이다.

---

구속조건 $C$는 구속조건과 관련된 정보를 가진다. 위치 상태 벡터 $\mathbf{q}$에 대한 함수이므로, $C$는 위치 차원이고, $\dot{C}$는 속도 차원이다.

```math
\begin{align*}
C(\mathbf{q}) &= 0 \\
\dot{C}(\mathbf{q}) &= \frac{\partial C}{\partial \mathbf{q}} \dot{\mathbf{q}} = \mathbf{J} \dot{\mathbf{q}} + b = 0 \quad \text{where} \quad b = \text{velocity bias}
\end{align*}
```

$\dot{C}$은 야코비안을 통해 기술될 수 있는데, 이 때 야코비안 $\mathbf{J}$은 다음과 같이 1 * (n+1) 차원의 행렬이다.

```math
\mathbf{J} = \frac{\partial C}{\partial \mathbf{q}} = \begin{bmatrix} \frac{\partial C}{\partial \mathbf{p}} & \frac{\partial C}{\partial \theta} \end{bmatrix}
```

---

충격량(Impulse)이란, $\mathbf{P} = \mathbf{F} \Delta t = m \Delta \mathbf{v}$로 정의된다. 즉, 충격량은 속도를 변화시키는 물리량이다.

물체에 외력이 작용함으로써 구속조건을 만족하지 않게 되면 물체에 구속조건을 만족시킬 수 있게 구속력이 작용하게 하는 방식으로 구속조건을 유지시킬 수 있다. 이 dynamics는 Impulse Based이므로, 구속력 대신 구속충격량 $\mathbf{P_c}$를 사용한다.

```math
\begin{align*}
\dot{\mathbf{q}}_{i+1}^* &= \dot{\mathbf{q}}_i + \mathbf{M^{-1}}\mathbf{F_{ext}} \Delta t \\
\dot{\mathbf{q}}_{i+1} &= \dot{\mathbf{q}}_{i+1}^* + \mathbf{M^{-1}}\mathbf{P_c} \\
\Rightarrow \dot{\mathbf{q}}_{i+1} &= \dot{\mathbf{q}}_i + \mathbf{M^{-1}}(\mathbf{F_{ext}} \Delta t  + \mathbf{P_c})
\end{align*}
```

또한, Force Based Dynamics와 비슷하게 유도하면, 구속충격량은 다음과 같이 구할 수 있다.

```math
\mathbf{P_c} = \mathbf{J}^T \lambda
```

$\lambda$를 구하기 위해 미분된 구속조건에 전부 대입해주자.

```math
\begin{align*}
\dot{C}(\mathbf{q}_{i+1}) &= \mathbf{J} \dot{\mathbf{q}_{i+1}} + b \\
 &= \mathbf{J}(\dot{\mathbf{q}}_{i+1}^* + \mathbf{M^{-1}}\mathbf{P_c}) + b \\ &= \mathbf{J}(\dot{\mathbf{q}}_{i+1}^* + \mathbf{M^{-1}}\mathbf{J}^T \lambda) + b \\ &= \mathbf{J}\mathbf{M^{-1}}\mathbf{J}^T \lambda + \mathbf{J}\dot{\mathbf{q}}_{i+1}^* + b \\ &= \mathbf{J}\mathbf{M^{-1}}\mathbf{J}^T \lambda + \mathbf{J}(\dot{\mathbf{q}}_i + \mathbf{M^{-1}}\mathbf{F_{ext}} \Delta t) + b  \\ &= 0
\end{align*}
```

```math
\textcolor{red}{\mathbf{J}\mathbf{M^{-1}}\mathbf{J}^T} \lambda = \textcolor{blue}{-\mathbf{J}(\dot{\mathbf{q}}_i + \mathbf{M^{-1}}\mathbf{F_{ext}} \Delta t) - b}
```

위 식을 통해 $\lambda$를 구할 수 있다. 이 때, 좌변의 $\textcolor{red}{\mathbf{J}\mathbf{M^{-1}}\mathbf{J}^T}$와, 우변의 $\textcolor{blue}{-\mathbf{J}(\dot{\mathbf{q}}\_i + \mathbf{M^{-1}}\mathbf{F_{ext}} \Delta t) - b}$는 모두 스칼라이기 때문에 $\lambda$를 쉽게 구할 수 있다.

---

$\lambda$를 구했다면, 구속충격량 $\mathbf{P_c} = \mathbf{J}^T \lambda$를 구할 수 있다. 이를 통해, 물체의 속도를 업데이트할 수 있다.

```math
\dot{\mathbf{q}}_{i+1} = \dot{\mathbf{q}}_i + \mathbf{M^{-1}}(\mathbf{F_{ext}} \Delta t + \mathbf{P_c})
```

이후, 위치는 [Force Based Dynamics](force-based-1.md)와 동일하게 업데이트할 수 있다.

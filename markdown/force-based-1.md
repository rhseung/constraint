# Force Based Dynamics 1

위치 상태 벡터 $\mathbf{{Q}}$는 n개의 물체의 위치와 관련된 정보를 가진다.

```math
\mathbf{{Q}}={\left(\begin{matrix}{\mathbf{{p}}}_{{1}}\\\theta_{{1}}\\{\mathbf{{p}}}_{{2}}\\\theta_{{2}}\\\vdots\\{\mathbf{{p}}}_{{n}}\\\theta_{{n}}\end{matrix}\right)}={\left(\begin{matrix}{\mathbf{{q}}}_{{1}}\\{\mathbf{{q}}}_{{2}}\\\vdots\\{\mathbf{{q}}}_{{n}}\end{matrix}\right)}
```

$\mathbf{{p}}$는 d차원에서 원소가 d개이므로, $\mathbf{{q}}$는 d+1 차원이다. 즉, $\mathbf{{Q}}$는 n(d+1) 차원이다.  
2차원에 대해서만 다루므로, $\mathbf{{Q}}$는 3n 차원이다.

---

질량 행렬 $\mathbf{{M}}$은 n개의 물체의 질량과 관련된 정보를 가진다. 2차원인 경우 아래와 같다.

```math
\mathbf{{M}}={\left[\begin{matrix}{m}_{{1}}&&&&&&\\&{m}_{{1}}&&&&&\\&&{I}_{{1}}&&&&\\&&&\ddots&&&\\&&&&{m}_{{n}}&&\\&&&&&{m}_{{n}}&\\&&&&&&{I}_{{n}}\end{matrix}\right]}={\left[\begin{matrix}{\mathbf{{m}}}_{{1}}&&&\\&{\mathbf{{m}}}_{{2}}&&\\&&\ddots&\\&&&{\mathbf{{m}}}_{{n}}\end{matrix}\right]}
```

$\mathbf{{I}}$는 각 물체의 회전 관성이고, $m$은 질량이다.  
$\mathbf{{m}}$은 3 * 3 행렬이다. $\mathbf{{M}}$은 3n * 3n 차원이라고 할 수 있다. 2차원이기 때문에 각 성분인 x, y에 대해 m을 2번 써주는 것이다.  

---

힘 상태 벡터 $\mathbf{{F}}$는 n개의 물체의 힘과 관련된 정보를 가진다.

```math
{\mathbf{{F}}}={\left(\begin{matrix}{\mathbf{{f}}}_{{1}}\\\tau_{{1}}\\{\mathbf{{f}}}_{{2}}\\\tau_{{2}}\\\vdots\\{\mathbf{{f}}}_{{n}}\\\tau_{{n}}\end{matrix}\right)}={\left(\begin{matrix}{\mathbf{{F}}}_{{1}}\\{\mathbf{{F}}}_{{2}}\\\vdots\\{\mathbf{{F}}}_{{n}}\end{matrix}\right)}
```

$\mathbf{{Q}}$와 동일하게 2차원인 경우 3n 차원이다.

---

구속조건 벡터 $\mathbf{{C}}$는 m개의 구속조건과 관련된 정보를 가진다. 위치 상태 벡터 $\mathbf{{Q}}$에 대한 함수이므로, $\mathbf{{C}}$는 위치 차원이다.

```math
{\mathbf{{C}}}{\left({\mathbf{{Q}}}\right)}={\left(\begin{matrix}{C}_{{1}}{\left({\mathbf{{Q}}}\right)}\\{C}_{{2}}{\left({\mathbf{{Q}}}\right)}\\\vdots\\{C}_{{m}}{\left({\mathbf{{Q}}}\right)}\end{matrix}\right)}
```

---

우리는 구속조건이 0이 되도록 해야 한다. 따라서, ${C}={0}\Rightarrow\dot{{C}}={0}\Rightarrow\ddot{{C}}={0}$이다.  
$\mathbf{{C}}$를 시간에 대해 미분해 속도 차원의 구속조건을 구할 수 있다.

```math
\dot{{\mathbf{{C}}}}=\frac{{\partial{\mathbf{{C}}}}}{{\partial{\mathbf{{Q}}}}}\cdot\dot{{\mathbf{{Q}}}}={\mathbf{{J}}}\dot{{\mathbf{{Q}}}}={0}\,\,\Rightarrow\,\,{\mathbf{{J}}}^T\bot\dot{{\mathbf{{Q}}}}
```

야코비안을 활용해서 구속조건의 시간에 대한 미분형을 써줄 수 있다. 또한, $\dot{{C}}={0}$이므로, 직교 조건도 확인할 수 있다. 이 때, 야코비안 행렬 $\mathbf{{J}}$는 다음과 같다.  
  
```math
\mathbf{J} = \begin{bmatrix}
\frac{\partial C_1}{\partial q_1} & \frac{\partial C_1}{\partial q_2} & \cdots & \frac{\partial C_1}{\partial q_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial C_m}{\partial q_1} & \frac{\partial C_m}{\partial q_2} & \cdots & \frac{\partial C_m}{\partial q_n}
\end{bmatrix}
```

```math
\frac{\partial x}{\partial \mathbf{y}} = \begin{bmatrix} \frac{\partial x}{\partial y_1}, & \frac{\partial x}{\partial y_2}, & \cdots, & \frac{\partial x}{\partial y_n} \end{bmatrix}
```
  
즉, 야코비안 행렬은 구속조건의 각 성분에 대해 위치 벡터에 대한 편미분을 구한 것이다. 차원은 2차원인 경우 m * 3n이다.

한 번 더 시간에 대해 미분해 가속도 차원의 구속조건을 구해보자.

```math
{\mathbf{\dot{{C}}}}={\mathbf{{J}}}{\mathbf{\dot{{Q}}}}={0}\,\,\Rightarrow\,\,{\mathbf{\ddot{{C}}}}={\mathbf{\dot{{J}}}}{\mathbf{\dot{{Q}}}}+{\mathbf{{J}}}{\mathbf{\ddot{{Q}}}}={0}
```

여기서, 뉴턴의 운동 제 2법칙 ${\mathbf{{F}}}={m}{\mathbf{{a}}}$을 적용하여 $\mathbf{\ddot{{Q}}}$를 구할 수 있다. 또한, 물체에 작용하는 $\mathbf{{F_{tot}}}$은 외력 $\mathbf{{F_{ext}}}$와 구속력 $\mathbf{{F_c}}$에 의해 결정된다.

```math
{\mathbf{\ddot{{Q}}}}={\mathbf{{M}}}^{ -{{1}}}{\mathbf{{F}}}={\mathbf{{M}}}^{ -{{1}}}{\left({\mathbf{{{F}_{{{e}{x}{t}}}}}}+{\mathbf{{{F}_{{c}}}}}\right)}
```

또한, 구속력 $\mathbf{{F_c}}$는 일을 하지 않으므로, $\mathbf{{F_c}}$와 $\mathbf{{\dot{{Q}}}}$는 직교한다. 따라서, 아까 구한 $\mathbf{{J}}^T\bot\dot{{\mathbf{{Q}}}}$를 이용하면 다음과 같다.  
$\mathbf{{\lambda}}$는 각 구속조건에 대한 라그랑지 승수를 가지고 있는 m차원 벡터이다.

```math
{\mathbf{{{F}_{{c}}}}}\bot{\mathbf{\dot{{Q}}}},\,\,{\mathbf{{J}}}^{T}\bot{\mathbf{\dot{{Q}}}}\,\Rightarrow\,{\mathbf{{{F}_{{c}}}}}={\mathbf{{J}}}^{T}{\mathbf{{\lambda}}}
```

두 식을 합치면, 다음과 같다.

```math
\dot{\mathbf{J}}\dot{\mathbf{Q}} + \mathbf{J}\ddot{\mathbf{Q}} = 0 \implies \textcolor{red}{\mathbf{J}\mathbf{M}^{-1}\mathbf{J}^T} \mathbf{\lambda} = \textcolor{blue}{-\dot{\mathbf{J}}\dot{\mathbf{Q}} - \mathbf{J}\mathbf{M}^{-1} \mathbf{{F}_{\text{ext}}}}
```

$\mathbf{{\lambda}}$앞의 $\textcolor{red}{\mathbf{J}\mathbf{M}^{-1}\mathbf{J}^T}$은 m * m 차원이고, 뒤의 $\textcolor{blue}{-\dot{\mathbf{J}}\dot{\mathbf{Q}} - \mathbf{J}\mathbf{M}^{-1} \mathbf{{F}_{\text{ext}}}}$은 m차원 벡터이다.  
위 식은 라그랑지 승수법을 이용한 방정식이다. 이 방정식을 풀어서 물체의 위치와 속도를 구할 수 있다.  

${\color{red}{{A}}}\mathbf{\lambda}={\color{blue}{b}}$ 꼴의 선형 방정식을 푸는 방법은 [PGS](https://ko.wikipedia.org/wiki/%EA%B0%80%EC%9A%B0%EC%8A%A4-%EC%9E%90%EC%9D%B4%EB%8D%B8_%EB%B0%A9%EB%B2%95)를 사용해서 풀 수 있다.

---

$\mathbf{{\lambda}}$를 구했으면, ${\mathbf{{{F}_{{c}}}}}={\mathbf{{J}}}^{T}{\mathbf{{\lambda}}}$를 구할 수 있다. 이를 이용해 속도를 갱신할 수 있다.

```math
{\mathbf{\dot{{Q}}}}_{{{i}+{1}}}={\mathbf{\dot{{Q}}}}_{{i}}+{\mathbf{{M}}}^{ -{{1}}}{\left({\mathbf{{{F}_{{{e}{x}{t}}}}}}+{\mathbf{{{F}_{{c}}}}}\right)}
```

이후, 위치 갱신은 반-암시적 오일러 방법([Semi-implicit Euler method](https://ko.wikipedia.org/wiki/%EB%B0%98-%EC%95%94%EC%8B%9C%EC%A0%81_%EC%98%A4%EC%9D%BC%EB%9F%AC_%EB%B0%A9%EB%B2%95)) 혹은, 룽게-쿠타 방법([Runge-Kutta method](https://ko.wikipedia.org/wiki/%EB%A3%BD%EA%B2%8C-%EC%BF%A0%ED%83%80_%EB%B0%A9%EB%B2%95))을 사용할 수 있다.

- 반-암시적 오일러 방법: $\mathbf{Q}\_{i+1}=\mathbf{Q}\_i +\mathbf{{\dot{{Q}}}}_{i+1}\Delta{t}$
- 룽게-쿠타 방법(RK4):

```math
\dot{\mathbf{Q}} = f(t, \mathbf{Q})
```

```math
\mathbf{k_1} = f(t_i, \mathbf{Q}_i)
```

```math
\mathbf{k_2} = f\left(t_i + \frac{\Delta t}{2}, \mathbf{Q}_i + \frac{\mathbf{k_1} \Delta t}{2}\right)
```

```math
\mathbf{k_3} = f\left(t_i + \frac{\Delta t}{2}, \mathbf{Q}_i + \frac{\mathbf{k_2} \Delta t}{2}\right)
```

```math
\mathbf{k_4} = f(t_i + \Delta t, \mathbf{Q}_i + \mathbf{k_3} \Delta t)
```

```math
\mathbf{Q}_{i+1} = \mathbf{Q}_i + \frac{\Delta t}{6} \left( \mathbf{k_1} + 2\mathbf{k_2} + 2\mathbf{k_3} + \mathbf{k_4} \right)
```

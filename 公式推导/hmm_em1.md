# HMM 的 EM 算法：公式与代码对照（GitHub 友好版）

> 适用于 GitHub、VS Code、Jupyter，避免公式换行与渲染问题。

---

## 一、公式区（右对齐编号）

### (1) 前向变量 $\alpha_{t}(j)$
$$
\alpha_{t}(j) = 
\begin{cases}
\pi_{j} b_{j}(x_{1}) & t = 1 \\
\left( \sum_{i=1}^{N} \alpha_{t-1}(i) a_{ij} \right) b_{j}(x_{t}) & t > 1
\end{cases}
\tag{1}
$$

### (2) 后向变量 $\beta_{t}(i)$
$$
\beta_{t}(i) = 
\begin{cases}
1 & t = T \\
\sum_{j=1}^{N} a_{ij} b_{j}(x_{t+1}) \beta_{t+1}(j) & t < T
\end{cases}
\tag{2}
$$

### (3) 状态后验 $\gamma_{t}(i)$
$$
\gamma_{t}(i) = \frac{\alpha_{t}(i) \beta_{t}(i)}{\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j)}
\tag{3}
$$

### (4) 转移后验 $\xi_{t}(i,j)$
$$
\xi_{t}(i,j) = \frac{\alpha_{t}(i) a_{ij} b_{j}(x_{t+1}) \beta_{t+1}(j)}{P(x)},\quad P(x) = \sum_{k=1}^{N} \alpha_{T}(k)
\tag{4}
$$

### (5) 参数更新
$$
\pi_{i}^{\text{new}} = \gamma_{1}(i)
\tag{5.1}
$$
$$
a_{ij}^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_{t}(i,j)}{\sum_{t=1}^{T-1} \gamma_{t}(i)}
\tag{5.2}
$$
$$
b_{j}(k)^{\text{new}} = \frac{\sum_{t: x_{t} = k} \gamma_{t}(j)}{\sum_{t=1}^{T} \gamma_{t}(j)}
\tag{5.3}
$$

---

## 二、代码区（公式引用在注释中）

```python
from functools import lru_cache

# 示例参数
π = [0.6, 0.4]
A = [[0.7, 0.3],
     [0.4, 0.6]]
B = [[0.5, 0.5],
     [0.3, 0.7]]
x = [0, 1, 0, 0, 1]  # 观测序列
N, M, T = len(π), len(B[0]), len(x)

# (1) 前向变量 α_t(j)
@lru_cache(maxsize=None)
def alpha(t, j):
    if t == 1:
        return π[j] * B[j][x[0]]  # 公式 (1)
    else:
        sum_term = sum(alpha(t-1, i) * A[i][j] for i in range(N))
        return sum_term * B[j][x[t-1]]  # 公式 (1)

# (2) 后向变量 β_t(i)
@lru_cache(maxsize=None)
def beta(t, i):
    if t == T:
        return 1.0  # 公式 (2)
    else:
        return sum(
            A[i][j] * B[j][x[t]] * beta(t+1, j)
            for j in range(N)
        )  # 公式 (2)

# (3) 状态后验 γ_t(i)
@lru_cache(maxsize=None)
def gamma(t, i):
    numer = alpha(t, i) * beta(t, i)
    denom = sum(alpha(t, j) * beta(t, j) for j in range(N))
    return numer / denom if denom != 0 else 0.0  # 公式 (3)

# (4) 转移后验 ξ_t(i,j)
@lru_cache(maxsize=None)
def xi(t, i, j):
    if t >= T:
        raise ValueError("t must be < T")
    numerator = alpha(t, i) * A[i][j] * B[j][x[t]] * beta(t+1, j)
    Z = sum(alpha(T, k) for k in range(N))  # P(x), 公式 (4)
    return numerator / Z if Z != 0 else 0.0  # 公式 (4)

# (5.1) 更新 π_i
def update_pi(i):
    return gamma(1, i)  # 公式 (5.1)

# (5.2) 更新 a_ij
def update_A(i, j):
    numerator = sum(xi(t, i, j) for t in range(1, T))
    denominator = sum(gamma(t, i) for t in range(1, T))
    return numerator / denominator if denominator != 0 else 0.0  # 公式 (5.2)

# (5.3) 更新 b_j(k)
def update_B(j, k):
    numerator = sum(gamma(t, j) for t in range(1, T+1) if x[t-1] == k)
    denominator = sum(gamma(t, j) for t in range(1, T+1))
    return numerator / denominator if denominator != 0 else 0.0  # 公式 (5.3)

# 执行一次 EM 迭代
def em_step():
    # 触发缓存计算
    for t in range(1, T+1):
        for i in range(N):
            gamma(t, i)
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                xi(t, i, j)
    # 更新参数
    π_new = [update_pi(i) for i in range(N)]
    A_new = [[update_A(i, j) for j in range(N)] for i in range(N)]
    B_new = [[update_B(j, k) for k in range(M)] for j in range(N)]
    return {'π': π_new, 'A': A_new, 'B': B_new}

# 调用
params_new = em_step()
print("更新后的参数：", params_new)

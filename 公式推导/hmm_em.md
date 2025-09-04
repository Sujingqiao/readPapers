# HMM 的 EM 算法：公式与代码对照表

> **文件名建议：`hmm-em-alignment.md`**

本文档采用 **公式区 + 代码区** 结构，所有公式右对齐编号，代码中通过注释 `(1)`、`(2)` 引用，确保逻辑清晰、可维护。

---

## 📐 公式区（右对齐编号）

$$
\alpha_t(j) = 
\begin{cases}
\pi_j b_j(x_1) & t = 1 \\
\left( \sum_{i=1}^N \alpha_{t-1}(i) a_{ij} \right) b_j(x_t) & t > 1
\end{cases}
\tag{1}
$$

$$
\beta_t(i) = 
\begin{cases}
1 & t = T \\
\sum_{j=1}^N a_{ij} b_j(x_{t+1}) \beta_{t+1}(j) & t < T
\end{cases}
\tag{2}
$$

$$
\gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^N \alpha_t(j) \beta_t(j)}
\tag{3}
$$

$$
\xi_t(i,j) = \frac{\alpha_t(i) a_{ij} b_j(x_{t+1}) \beta_{t+1}(j)}{P(x_{1:T})},\quad P(x_{1:T}) = \sum_{k=1}^N \alpha_T(k)
\tag{4}
$$

$$
\pi_i^{\text{new}} = \gamma_1(i)
\tag{5}
$$

$$
a_{ij}^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
\tag{6}
$$

$$
b_j(k)^{\text{new}} = \frac{\sum_{t: x_t = k} \gamma_t(j)}{\sum_{t=1}^T \gamma_t(j)}
\tag{7}
$$

---

## 💻 代码区（公式引用注释）

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
        return π[j] * B[j][x[0]]                    # | (1)
    else:
        sum_term = sum(alpha(t-1, i) * A[i][j] for i in range(N))
        return sum_term * B[j][x[t-1]]              # | (1)

# (2) 后向变量 β_t(i)
@lru_cache(maxsize=None)
def beta(t, i):
    if t == T:
        return 1.0                                  # | (2)
    else:
        return sum(
            A[i][j] * B[j][x[t]] * beta(t+1, j)     # | (2)
            for j in range(N)
        )

# (3) 状态后验 γ_t(i)
@lru_cache(maxsize=None)
def gamma(t, i):
    numer = alpha(t, i) * beta(t, i)
    denom = sum(alpha(t, j) * beta(t, j) for j in range(N))
    return numer / denom if denom != 0 else 0.0     # | (3)

# (4) 转移后验 ξ_t(i,j)
@lru_cache(maxsize=None)
def xi(t, i, j):
    if t >= T:
        raise ValueError("t must be < T")
    numerator = alpha(t, i) * A[i][j] * B[j][x[t]] * beta(t+1, j)
    Z = sum(alpha(T, k) for k in range(N))          # | (4)
    return numerator / Z if Z != 0 else 0.0         # | (4)

# (5) 更新初始概率
def update_pi(i):
    return gamma(1, i)                              # | (5)

# (6) 更新转移概率
def update_A(i, j):
    numerator = sum(xi(t, i, j) for t in range(1, T))       # | (6)
    denominator = sum(gamma(t, i) for t in range(1, T))     # | (6)
    return numerator / denominator if denominator != 0 else 0.0

# (7) 更新发射概率
def update_B(j, k):
    numerator = sum(gamma(t, j) for t in range(1, T+1) if x[t-1] == k)  # | (7)
    denominator = sum(gamma(t, j) for t in range(1, T+1))               # | (7)
    return numerator / denominator if denominator != 0 else 0.0

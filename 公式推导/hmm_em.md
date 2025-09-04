# HMM 的 EM 算法：公式与代码对照（GitHub 友好版）

> 适用于 GitHub、VS Code、Jupyter，无需 MathJax 插件也能清晰阅读。

---

## 一、公式区（右对齐编号）

### (1) 前向变量 $\alpha_t(j)$

$$
\alpha_t(j) = 
\begin{cases}
\pi_j b_j(x_1) & t = 1 \\
\left( \sum_{i=1}^N \alpha_{t-1}(i) a_{ij} \right) b_j(x_t) & t > 1
\end{cases}
\tag{1}
$$

### (2) 后向变量 $\beta_t(i)$

$$
\beta_t(i) = 
\begin{cases}
1 & t = T \\
\sum_{j=1}^N a_{ij} b_j(x_{t+1}) \beta_{t+1}(j) & t < T
\end{cases}
\tag{2}
$$

### (3) 状态后验 $\gamma_t(i)$

$$
\gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^N \alpha_t(j) \beta_t(j)}
\tag{3}
$$

### (4) 转移后验 $\xi_t(i,j)$

$$
\xi_t(i,j) = \frac{\alpha_t(i) a_{ij} b_j(x_{t+1}) \beta_{t+1}(j)}{P(x)},\quad P(x) = \sum_{k=1}^N \alpha_T(k)
\tag{4}
$$

### (5) 参数更新

$$
\pi_i^{\text{new}} = \gamma_1(i)
\tag{5.1}
$$
$$
a_{ij}^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
\tag{5.2}
$$
$$
b_j(k)^{\text{new}} = \frac{\sum_{t: x_t = k} \gamma_t(j)}{\sum_{t=1}^T \gamma_t(j)}
\tag{5.3}
$$

---

## 二、代码区（公式引用在注释中，代码连续不割裂）

```python
from functools import lru_cache

# 示例参数
pi = [0.6, 0.4]          # 初始概率 π
A = [[0.7, 0.3],         # 转移概率 a_ij
     [0.4, 0.6]]
B = [[0.5, 0.5],         # 发射概率 b_j(k)
     [0.3, 0.7]]
x = [0, 1, 0, 0, 1]      # 观测序列 x_t
N, M, T = len(pi), len(B[0]), len(x)  # 状态数、符号数、序列长度

@lru_cache(maxsize=None)
def alpha(t, j):
    """前向变量 α_t(j)"""
    if t == 1:
        return pi[j] * B[j][x[0]]  # 公式 (1)
    sum_term = sum(alpha(t-1, i) * A[i][j] for i in range(N))
    return sum_term * B[j][x[t-1]]  # 公式 (1)

@lru_cache(maxsize=None)
def beta(t, i):
    """后向变量 β_t(i)"""
    if t == T:
        return 1.0  # 公式 (2)
    return sum(A[i][j] * B[j][x[t]] * beta(t+1, j) for j in range(N))  # 公式 (2)

@lru_cache(maxsize=None)
def gamma(t, i):
    """状态后验 γ_t(i)"""
    numer = alpha(t, i) * beta(t, i)
    denom = sum(alpha(t, j) * beta(t, j) for j in range(N))
    return numer / denom if denom != 0 else 0.0  # 公式 (3)

@lru_cache(maxsize=None)
def xi(t, i, j):
    """转移后验 ξ_t(i,j)"""
    if t >= T:
        raise ValueError("t must be < T")
    numerator = alpha(t, i) * A[i][j] * B[j][x[t]] * beta(t+1, j)
    Z = sum(alpha(T, k) for k in range(N))  # P(x) = sum_k α_T(k), 公式 (4)
    return numerator / Z if Z != 0 else 0.0  # 公式 (4)

def update_pi(i):
    """更新初始概率 π_i"""
    return gamma(1, i)  # 公式 (5.1)

def update_A(i, j):
    """更新转移概率 a_ij"""
    numerator = sum(xi(t, i, j) for t in range(1, T))      # 公式 (5.2)
    denominator = sum(gamma(t, i) for t in range(1, T))    # 公式 (5.2)
    return numerator / denominator if denominator != 0 else 0.0

def update_B(j, k):
    """更新发射概率 b_j(k)"""
    numerator = sum(gamma(t, j) for t in range(1, T+1) if x[t-1] == k)  # 公式 (5.3)
    denominator = sum(gamma(t, j) for t in range(1, T+1))                # 公式 (5.3)
    return numerator / denominator if denominator != 0 else 0.0

def em_step():
    """执行一次完整的 EM 迭代"""
    # 触发 E 步：缓存所有 gamma 和 xi
    for t in range(1, T+1):
        for i in range(N):
            gamma(t, i)
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                xi(t, i, j)
    # M 步：更新参数
    pi_new = [update_pi(i) for i in range(N)]                          # 公式 (5.1)
    A_new = [[update_A(i, j) for j in range(N)] for i in range(N)]     # 公式 (5.2)
    B_new = [[update_B(j, k) for k in range(M)] for j in range(N)]     # 公式 (5.3)
    return {'pi': pi_new, 'A': A_new, 'B': B_new}

# 执行并打印结果
params_new = em_step()
print("更新后的参数：")
print("π =", params_new['pi'])
print("A =", params_new['A'])
print("B =", params_new['B'])

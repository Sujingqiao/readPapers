# HMM 的 EM 算法：公式与代码对照（GitHub 稳定版）

> 专为 GitHub 设计，避免下标换行、中文空格、渲染断裂问题。

---

## 一、公式区（右对齐编号，块级公式）

### (1) 前向变量 \(\alpha_t(j)\)
\[
\alpha_t(j) = 
\begin{cases}
\pi_j b_j(x_1) & t = 1 \\
\left( \sum_{i=1}^N \alpha_{t-1}(i) a_{ij} \right) b_j(x_t) & t > 1
\end{cases}
\tag{1}
\]

### (2) 后向变量 \(\beta_t(i)\)
\[
\beta_t(i) = 
\begin{cases}
1 & t = T \\
\sum_{j=1}^N a_{ij} b_j(x_{t+1}) \beta_{t+1}(j) & t < T
\end{cases}
\tag{2}
\]

### (3) 状态后验 \(\gamma_t(i)\)
\[
\gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^N \alpha_t(j) \beta_t(j)}
\tag{3}
\]

### (4) 转移后验 \(\xi_t(i,j)\)
\[
\xi_t(i,j) = \frac{\alpha_t(i) a_{ij} b_j(x_{t+1}) \beta_{t+1}(j)}{P(x)},\quad P(x) = \sum_{k=1}^N \alpha_T(k)
\tag{4}
\]

### (5) 参数更新
\[
\pi_i^{\text{new}} = \gamma_1(i)
\tag{5.1}
\]
\[
a_{ij}^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
\tag{5.2}
\]
\[
b_j(k)^{\text{new}} = \frac{\sum_{t: x_t = k} \gamma_t(j)}{\sum_{t=1}^T \gamma_t(j)}
\tag{5.3}
\]

---

## 二、代码区（公式引用在注释中，不割裂）

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
        return π[j] * B[j][x[0]]  # α_t(j), 公式 (1)
    else:
        sum_term = sum(alpha(t-1, i) * A[i][j] for i in range(N))
        return sum_term * B[j][x[t-1]]  # 公式 (1)

# (2) 后向变量 β_t(i)
@lru_cache(maxsize=None)
def beta(t, i):
    if t == T:
        return 1.0  # β_t(i), 公式 (2)
    else:
        return sum(
            A[i][j] * B[j][x[t]] * beta(t+1, j)
            for j in range(N)
        )  # 公式 (2)

# (3) 状态后验 γ_t(i)
@lru_cache(maxsize=None)
def gamma(t, i):
    numer = alpha(t, i) * beta(t, i)  # α_t(i)β_t(i)
    denom = sum(alpha(t, j) * beta(t, j) for j in range(N))  # ∑α_t(j)β_t(j)
    return numer / denom if denom != 0 else 0.0  # 公式 (3)

# (4) 转移后验 ξ_t(i,j)
@lru_cache(maxsize=None)
def xi(t, i, j):
    if t >= T:
        raise ValueError("t must be < T")
    numerator = alpha(t, i) * A[i][j] * B[j][x[t]] * beta(t+1, j)  # 分子，公式 (4)
    Z = sum(alpha(T, k) for k in range(N))  # P(x) = ∑α_T(k)
    return numerator / Z if Z != 0 else 0.0  # 公式 (4)

# (5.1) 更新初始概率 π_i
def update_pi(i):
    return gamma(1, i)  # π_i = γ_1(i), 公式 (5.1)

# (5.2) 更新转移概率 a_ij
def update_A(i, j):
    numerator = sum(xi(t, i, j) for t in range(1, T))  # ∑ξ_t(i,j)
    denominator = sum(gamma(t, i) for t in range(1, T))  # ∑γ_t(i)
    return numerator / denominator if denominator != 0 else 0.0  # 公式 (5.2)

# (5.3) 更新发射概率 b_j(k)
def update_B(j, k):
    numerator = sum(gamma(t, j) for t in range(1, T+1) if x[t-1] == k)  # ∑_{t:x_t=k} γ_t(j)
    denominator = sum(gamma(t, j) for t in range(1, T+1))  # ∑γ_t(j)
    return numerator / denominator if denominator != 0 else 0.0  # 公式 (5.3)

# EM 迭代主函数
def em_step():
    # 触发 E 步缓存
    for t in range(1, T+1):
        for i in range(N):
            gamma(t, i)
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                xi(t, i, j)
    # M 步更新
    π_new = [update_pi(i) for i in range(N)]
    A_new = [[update_A(i, j) for j in range(N)] for i in range(N)]
    B_new = [[update_B(j, k) for k in range(M)] for j in range(N)]
    return {'π': π_new, 'A': A_new, 'B': B_new}

# 执行
params_new = em_step()
print("更新后的参数：", params_new)

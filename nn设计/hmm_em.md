# HMM 的 EM 算法：函数式递归实现（简洁版）

## 1. 模型设定

- 隐状态数：
- $N$
- 观测序列：
- $x = (x_1, \dots, x_T)$
- 参数：
  - 初始概率：
  - $pi_i$
  - 转移概率： $a_{ij}$
  - 发射概率： $b_j(k)$

## 2. 前向变量 $\alpha_t(j)$

公式：
$$
\alpha_t(j) = 
\begin{cases}
\pi_j b_j(x_1) & t = 1 \
\left( \sum_i \alpha_{t-1}(i) a_{ij} \right) b_j(x_t) & t > 1
\end{cases}
$$

```python
@lru_cache(maxsize=None)
def alpha(t, j):
    if t == 1:
        return π[j] * B[j][x[0]]
    else:
        sum_term = sum(alpha(t-1, i) * A[i][j] for i in range(N))
        return sum_term * B[j][x[t-1]]
```

## 3. 后向变量 $\beta_t(i)$

公式：
$$
\beta_t(i) = 
\begin{cases}
1 & t = T \
\sum_j a_{ij} b_j(x_{t+1}) \beta_{t+1}(j) & t < T
\end{cases}
$$

```python
@lru_cache(maxsize=None)
def beta(t, i):
    if t == T:
        return 1.0
    else:
        return sum(A[i][j] * B[j][x[t]] * beta(t+1, j) for j in range(N))
```

## 4. 状态后验 $\gamma_t(i)$

公式：
$$
\gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{\sum_j \alpha_t(j) \beta_t(j)}
$$

```python
@lru_cache(maxsize=None)
def gamma(t, i):
    numer = alpha(t, i) * beta(t, i)
    denom = sum(alpha(t, j) * beta(t, j) for j in range(N))
    return numer / denom if denom != 0 else 0.0
```

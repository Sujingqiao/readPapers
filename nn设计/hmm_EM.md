# HMM 的 EM 算法：函数式递归实现

> **数学公式 + Python 代码一体化文档**  
> 支持 GitHub 渲染，可读、可复制、可运行

---

## 🎯 目标

将 HMM 的 EM 算法（Baum-Welch）以 **“数学即函数”** 的方式实现：

- 每个数学量 = 一个 `@lru_cache` 函数
- 数学公式与代码一一对应
- GitHub 友好，支持公式渲染

---

## 🧩 模型设定

- 隐状态数：$ N $
- 观测符号数：$ M $
- 观测序列：$ x = (x_1, \dots, x_T) $，$ x_t \in \{0,\dots,M-1\} $
- 参数：
  - 初始概率：$ \pi_i $
  - 转移概率：$ a_{ij} $
  - 发射概率：$ b_j(k) $

---

## 🔁 EM 算法：E步与M步

---

### 1. 前向变量 $\alpha_t(j)$

**📌 数学公式：**
$$
\alpha_t(j) = 
\begin{cases}
\pi_j b_j(x_1) & t = 1 \\
\left( \sum_{i=1}^N \alpha_{t-1}(i) a_{ij} \right) b_j(x_t) & t > 1
\end{cases}
$$

**💻 函数式实现：**

```python
@lru_cache(maxsize=None)
def alpha(t, j):
    if t == 1:
        return π[j] * B[j][x[0]]
    else:
        sum_term = sum(alpha(t-1, i) * A[i][j] for i in range(N))
        return sum_term * B[j][x[t-1]]

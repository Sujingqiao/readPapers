# HMM 的 EM 算法：函数式递归实现（防破碎版）\
\
> **用途：复制后运行还原脚本，得到完整 .md**\
\
## 1. 模型设定\
- 隐状态数：$ N $\
- 观测序列：$ x = (x_1, \dots, x_T) $\
- 参数：$ \pi_i $, $ a_{ij} $, $ b_j(k) $\
\
## 2. 前向变量 $
alpha_t(j)$\
公式：
$
alpha_t(j) = (t==1) ? 
pi_j b_j(x_1) : (
sum_i 
alpha_{t-1}(i) a_{ij}) b_j(x_t)
$\
\
\`\`\`python\
@lru_cache(maxsize=None)\
def alpha(t, j):\
    if t == 1:\
        return π[j] * B[j][x[0]]\
    else:\
        sum_term = sum(alpha(t-1, i) * A[i][j] for i in range(N))\
        return sum_term * B[j][x[t-1]]\
\`\`\`\
\
## 3. 后向变量 $
beta_t(i)$\
公式：
$
beta_t(i) = (t==T) ? 1 : 
sum_j a_{ij} b_j(x_{t+1}) 
beta_{t+1}(j)
$\
\
\`\`\`python\
@lru_cache(maxsize=None)\
def beta(t, i):\
    if t == T:\
        return 1.0\
    else:\
        return sum(A[i][j] * B[j][x[t]] * beta(t+1, j) for j in range(N))\
\`\`\`\
\
## 4. 状态后验 $
gamma_t(i)$\
公式：
$
gamma_t(i) = 
alpha_t(i) 
beta_t(i) / 
sum_j 
alpha_t(j) 
beta_t(j)
$\
\
\`\`\`python\
@lru_cache(maxsize=None)\
def gamma(t, i):\
    numer = alpha(t, i) * beta(t, i)\
    denom = sum(alpha(t, j) * beta(t, j) for j in range(N))\
    return numer / denom if denom != 0 else 0.0\
\`\`\`\
\
## 5. 转移后验 $
xi_t(i,j)$\
公式：
$
xi_t(i,j) = 
alpha_t(i) a_{ij} b_j(x_{t+1}) 
beta_{t+1}(j) / P(x)
$，其中 $P(x) = 
sum_k 
alpha_T(k)$\
\
\`\`\`python\
@lru_cache(maxsize=None)\
def xi(t, i, j):\
    if t >= T: raise ValueError("t < T")\
    numerator = alpha(t, i) * A[i][j] * B[j][x[t]] * beta(t+1, j)\
    Z = sum(alpha(T, k) for k in range(N))\
    return numerator / Z if Z != 0 else 0.0\
\`\`\`\
\
## 6. M 步更新\
- $
pi_i^{
text{new}} = 
gamma_1(i)$\
- $a_{ij}^{
text{new}} = 
sum_t 
xi_t(i,j) / 
sum_t 
gamma_t(i)$\
- $b_j(k)^{
text{new}} = 
sum_{t:x_t=k} 
gamma_t(j) / 
sum_t 
gamma_t(j)$\
\
\`\`\`python\
def update_pi(i): return gamma(1, i)\
def update_A(i, j):\
    num = sum(xi(t, i, j) for t in range(1, T))\
    den = sum(gamma(t, i) for t in range(1, T))\
    return num / den if den != 0 else 0.0\
def update_B(j, k):\
    num = sum(gamma(t, j) for t in range(1, T+1) if x[t-1]==k)\
    den = sum(gamma(t, j) for t in range(1, T+1))\
    return num / den if den != 0 else 0.0\
\`\`\`\
\
> 💡 还原方法：将此文本保存为 .txt，运行替换脚本。\

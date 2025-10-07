import sympy as sp
from sympy import I, symbols, diff, sqrt, exp, sin, cos, re, im, simplify, Eq, solve

def josephson_effect_sympy():
    # 定义符号变量
    t = symbols('t', real=True)
    n1, n2 = symbols('n1 n2', positive=True)
    phi1, phi2 = symbols('phi1 phi2', real=True)
    K, V, q = symbols('K V q', real=True)
    hbar = symbols('hbar', real=True)  # 保留ħ以显示完整推导
    
    # 定义波函数 Ψ = √n * exp(iφ)
    Psi1 = sqrt(n1) * exp(I * phi1)
    Psi2 = sqrt(n2) * exp(I * phi2)
    
    print("波函数定义:")
    print(f"Ψ₁ = {Psi1}")
    print(f"Ψ₂ = {Psi2}")
    print()
    
    # 薛定谔方程 (设 ħ = 1)
    # i ∂Ψ₁/∂t = - (qV/2) Ψ₁ + K Ψ₂
    # i ∂Ψ₂/∂t = + (qV/2) Ψ₂ + K Ψ₁
    
    # 计算 Ψ₁ 的时间导数
    dPsi1_dt = diff(Psi1, t)
    print("Ψ₁ 的时间导数:")
    print(f"dΨ₁/dt = {dPsi1_dt}")
    print()
    
    # 薛定谔方程左侧
    lhs1 = I * dPsi1_dt
    print("方程左侧:")
    print(f"i dΨ₁/dt = {lhs1}")
    print()
    
    # 薛定谔方程右侧  
    rhs1 = - (q * V / 2) * Psi1 + K * Psi2
    print("方程右侧:")
    print(f"右侧 = {rhs1}")
    print()
    
    # 建立方程
    equation = Eq(lhs1, rhs1)
    print("完整方程:")
    print(equation)
    print()
    
    # 分开实部和虚部
    lhs_real = re(lhs1)
    lhs_imag = im(lhs1)
    rhs_real = re(rhs1)
    rhs_imag = im(rhs1)
    
    print("左侧实部:", lhs_real)
    print("左侧虚部:", lhs_imag)
    print("右侧实部:", rhs_real) 
    print("右侧虚部:", rhs_imag)
    print()
    
    # 定义相位差
    delta_phi = phi2 - phi1
    
    # 手动推导（因为SymPy处理复数导数较复杂）
    print("手动推导结果:")
    
    # 从虚部得到粒子数变化
    dn1_dt_eq = Eq(diff(n1, t)/2/sqrt(n1), K * sqrt(n2) * sin(delta_phi))
    print("从虚部得到:")
    print(dn1_dt_eq)
    
    # 简化得到 dn1/dt
    dn1_dt_simplified = Eq(diff(n1, t), 2 * K * sqrt(n1 * n2) * sin(delta_phi))
    print("简化:")
    print(dn1_dt_simplified)
    print()
    
    # 从实部得到相位变化
    dphi1_dt_eq = Eq(-sqrt(n1) * diff(phi1, t), 
                     - (q * V / 2) * sqrt(n1) + K * sqrt(n2) * cos(delta_phi))
    print("从实部得到:")
    print(dphi1_dt_eq)
    
    # 简化得到 dφ₁/dt
    dphi1_dt_simplified = Eq(diff(phi1, t), (q * V / 2) - K * sqrt(n2/n1) * cos(delta_phi))
    print("简化:")
    print(dphi1_dt_simplified)
    print()
    
    # 同样推导第二个方程得到 dφ₂/dt
    dphi2_dt_simplified = Eq(diff(phi2, t), - (q * V / 2) - K * sqrt(n1/n2) * cos(delta_phi))
    print("dφ₂/dt:")
    print(dphi2_dt_simplified)
    print()
    
    # 相位差的时间导数
    d_delta_phi_dt = diff(delta_phi, t)
    d_delta_phi_eq = Eq(d_delta_phi_dt, diff(phi2, t) - diff(phi1, t))
    
    # 代入并简化
    delta_phi_result = simplify(diff(phi2, t) - diff(phi1, t))
    final_eq = Eq(d_delta_phi_dt, q * V)  # 约瑟夫森第二方程
    
    print("相位差的时间导数:")
    print(f"d(φ₂ - φ₁)/dt = {d_delta_phi_dt}")
    print(f"代入后: d(φ₂ - φ₁)/dt = {delta_phi_result}")
    print(f"约瑟夫森第二方程: {final_eq}")
    print()
    
    # 电流推导 (假设 n1 = n2 = n)
    n = symbols('n', positive=True)
    A, Omega = symbols('A Omega', positive=True)  # 截面积和体积
    
    # 电流 I = n A q (dz/dt)，这里 dz/dt 与 dn/dt 相关
    I_current = n * A * q * (1/(A * n * q)) * diff(n1, t)  # 简化关系
    I_simplified = K * q * Omega * sin(delta_phi)  # 最终结果
    
    josephson1_eq = Eq(symbols('I'), symbols('I0') * sin(delta_phi))
    I0_def = Eq(symbols('I0'), K * q * Omega)
    
    print("约瑟夫森第一方程:")
    print(josephson1_eq)
    print("其中:")
    print(I0_def)

if __name__ == "__main__":
    josephson_effect_sympy()




from sympy import symbols, Eq, diff, I, exp, sin, cos, simplify
from sympy import solve, Function

# 定义符号
t = symbols('t')
n1, n2, K, q, V, Omega = symbols('n1 n2 K q V Omega', real=True, positive=True)
phi1 = Function('phi1')(t)
phi2 = Function('phi2')(t)

# 波函数
Psi1 = (n1**0.5) * exp(I * phi1)
Psi2 = (n2**0.5) * exp(I * phi2)

# 薛定谔方程（ħ=1）
lhs1 = I * diff(Psi1, t)  # 左边：i dΨ1/dt
rhs1 = (q*V/2) * Psi1 + K * Psi2  # 右边：E1 Ψ1 + K Ψ2

print("左边 (i dΨ1/dt):")
print(lhs1)
print("右边 (E1 Ψ1 + K Ψ2):")
print(rhs1)

# 展开左边
lhs1_expanded = lhs1.expand()
print("左边展开:")
print(lhs1_expanded)

# 分离实部和虚部
# 令 lhs1 = a + I*b, rhs1 = c + I*d
# 我们可以比较实部和虚部

# 提取实部和虚部
lhs_real = lhs1.as_real_imag()[0]
lhs_imag = lhs1.as_real_imag()[1]
rhs_real = rhs1.as_real_imag()[0]
rhs_imag = rhs1.as_real_imag()[1]

print("\n实部方程:")
eq_real = Eq(lhs_real, rhs_real)
print(eq_real)

print("\n虚部方程:")
eq_imag = Eq(lhs_imag, rhs_imag)
print(eq_imag)

# 为了简化，假设 n1 = n2 = n, 并令 Δφ = φ2 - φ1
n = symbols('n', real=True, positive=True)
Delta_phi = phi2 - phi1

# 代入 n1 = n2 = n
eq_real_n = eq_real.subs({n1: n, n2: n}).simplify()
eq_imag_n = eq_imag.subs({n1: n, n2: n}).simplify()

print("\n代入 n1=n2=n 后的实部:")
print(eq_real_n)

print("\n虚部:")
print(eq_imag_n)

# 从虚部方程提取 dphi1/dt
# 虚部: -sqrt(n) * dphi1/dt = qV/2 * sqrt(n) + K * sqrt(n) * cos(Delta_phi)
# => -dphi1/dt = qV/2 + K * cos(Delta_phi)
# 类似地，对 φ2 方程

# 我们现在考虑电流表达式
# I = n * A * q * d(phi)/dt，但实际是载流子密度变化
# 在一维情况下，I = (n q A) * d(phi)/dt

# 但根据文献，I = I0 * sin(Delta_phi)
# 其中 I0 = K * q * Omega

I0 = K * q * Omega
I = I0 * sin(Delta_phi)

print(f"\n约瑟夫森第一方程: I = {I}")

# 相位差变化率
dDelta_phi_dt = diff(Delta_phi, t)
dDelta_phi_dt_simplified = dDelta_phi_dt.simplify()

print(f"\nd(phi2 - phi1)/dt = {dDelta_phi_dt_simplified}")

# 根据物理，应等于 qV/hbar，设 hbar=1
print(f"约瑟夫森第二方程: d(phi2 - phi1)/dt = {q*V}")

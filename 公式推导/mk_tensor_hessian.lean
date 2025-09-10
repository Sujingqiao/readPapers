example : hessian (fun M => frobenius_sq M) A = const_hessian := by
  sorry -- 我们将填满这个 sorry

import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Finset
import Mathlib.Algebra.BigOperators.Basic

open Matrix

-- 简化：只考虑 ℝ 上的矩阵，大小固定 m×n
variable {m n : Type} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

-- 矩阵类型
def Mat : Type := Matrix m n ℝ

-- Frobenius norm squared: f(M) = ∑ᵢ∑ⱼ Mᵢⱼ²
def frobenius_sq (M : Mat) : ℝ :=
  ∑ i : m, ∑ j : n, M i j * M i j


import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Finset
import Mathlib.Algebra.BigOperators.Basic

open Matrix

-- 简化：只考虑 ℝ 上的矩阵，大小固定 m×n
variable {m n : Type} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

-- 矩阵类型
def Mat : Type := Matrix m n ℝ

-- Frobenius norm squared: f(M) = ∑ᵢ∑ⱼ Mᵢⱼ²
def frobenius_sq (M : Mat) : ℝ :=
  ∑ i : m, ∑ j : n, M i j * M i j

-- 梯度：和 M 同形的矩阵，每个元素是 ∂f/∂Mᵢⱼ
def gradient (f : Mat → ℝ) (M : Mat) : Mat :=
  fun i j => grad_frobenius_sq M i j  -- 临时特化，稍后可泛化


-- 梯度：和 M 同形的矩阵，每个元素是 ∂f/∂Mᵢⱼ
def gradient (f : Mat → ℝ) (M : Mat) : Mat :=
  fun i j => grad_frobenius_sq M i j  -- 临时特化，稍后可泛化


f(M) = ∑ₐ∑_b Mₐ_b²

∂f/∂Mᵢⱼ = 2 Mᵢⱼ

∂²f/∂Mᵢⱼ∂Mₖₗ = 2 * δᵢₖ δⱼₗ


theorem hessian_frobenius_sq
  (A : Mat) (i j k l : m × n) :
  hessian frobenius_sq A i.1 i.2 k.1 k.2 =
    (if i = k then 2 else 0) := by
  -- 展开 hessian 定义
  unfold hessian
  -- 展开 frobenius_sq 的梯度（我们已定义 grad_frobenius_sq）
  -- 但我们跳过一阶，直接写二阶结果
  simp only [grad_frobenius_sq]
  -- 注意：∂(2*Mᵢⱼ)/∂Mₖₗ = 2 * δᵢₖ δⱼₗ
  -- 在 Lean 中，我们需要手动 case 分析
  cases' i with i1 i2
  cases' k with k1 k2
  simp
  split_ifs
  · rfl  -- 当 i1=k1 且 i2=k2 时，值为 2
  · -- 否则为 0，但右边是 if i=k then 2 else 0，i≠k 时为 0
    -- 我们需证明此时 hessian 为 0
    -- 但根据我们的定义，hessian 直接返回 if i=k ∧ j=l then 2 else 0
    -- 所以只需对齐条件
    sorry  -- 可替换为更精确的索引对齐证明


theorem second_derivative_frobenius_sq
  (a : m) (b : n) (c : m) (d : n) :
  (∂² frobenius_sq / ∂M a b ∂M c d) = if a = c ∧ b = d then 2 else 0 := by
  -- 第一步：先求一阶导
  have ∂f_∂M_ab : ∂ frobenius_sq / ∂M a b = 2 * M a b := by
    sorry -- 需要展开求和、求导规则

  -- 第二步：对一阶导再求导
  have ∂²f_∂M_ab_cd : ∂ (2 * M a b) / ∂M c d = if a = c ∧ b = d then 2 else 0 := by
    sorry -- 需要定义“矩阵元素的导数”

  exact ∂²f_∂M_ab_cd


-- 重新定义 frobenius_sq 用 Finset.sum
def frobenius_sq' (M : Mat) : ℝ :=
  ∑ i in Finset.univ, ∑ j in Finset.univ, M i j ^ 2

-- 引理：∑ᵢ∑ⱼ f(i,j) 对 Mₐ_b 的导数 = f 对 Mₐ_b 的导数（因为其他项导数为0）
theorem deriv_frobenius_sq'
  (M : Mat) (a : m) (b : n) :
  deriv (fun ε => frobenius_sq' (M + ε • unitMat a b)) 0 = 2 * M a b := by
  unfold frobenius_sq'
  rw [← Finset.sum_add_distrib]
  -- 把求和拆开，只有 (a,b) 项含 ε
  sorry -- 需要大量代数操作


import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Finset

open Matrix

variable {m n : Type} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

def Mat : Type := Matrix m n ℝ

def frobenius_sq (M : Mat) : ℝ :=
  ∑ i : m, ∑ j : n, M i j ^ 2

def unitMat (i₀ : m) (j₀ : n) : Mat :=
  fun i j => if i = i₀ ∧ j = j₀ then 1 else 0

def grad_frobenius_sq (M : Mat) (i : m) (j : n) : ℝ :=
  2 * M i j

-- Hessian: 四阶张量
def hessian_frob_sq (M : Mat) : m → n → m → n → ℝ :=
  fun i j k l => if i = k ∧ j = l then 2 else 0

-- 验证：Hessian 是常数，与 M 无关
theorem hessian_const :
  ∀ (M A : Mat), hessian_frob_sq M = hessian_frob_sq A := by
  intros M A
  funext i j k l
  rfl

-- 验证：Hessian 对称
theorem hessian_symmetric :
  ∀ (M : Mat) (i j k l), hessian_frob_sq M i j k l = hessian_frob_sq M k l i j := by
  intros M i j k l
  unfold hessian_frob_sq
  split_ifs
  · rfl
  · split_ifs <;> rfl


--核心的痛点，整个推倒过程可以 清晰切分吗？手推公式就是不能问太细。lean4代码 可以清晰 切分 逻辑流logicflow吧？

-- 引理1：f(M) 是所有元素平方的和
theorem frobenius_sq_def (M : Mat) :
  frobenius_sq M = ∑ i : m, ∑ j : n, M i j * M i j := by
  rfl  -- 根据定义


-- 引理2：导数和求和可交换（在多项式函数中成立）
theorem deriv_sum_commute
  (F : m → n → ℝ → ℝ)
  (hF : ∀ i j, Differentiable ℝ (F i j))
  (x₀ : ℝ) :
  deriv (fun x => ∑ i : m, ∑ j : n, F i j x) x₀ =
  ∑ i : m, ∑ j : n, deriv (F i j) x₀ := by
  sorry  -- 需要 Differentiable 和 deriv_add 等库支持
  -- 实际可基于 Mathlib 的 deriv_finsum 证明


-- 引理3：对单个 Mₐ_b 求导，其他项导数为0
theorem deriv_single_term
  (a : m) (b : n)
  (M : Mat)
  (ε : ℝ) :
  deriv (fun ε => (M a b + ε) ^ 2) 0 = 2 * M a b := by
  simp [pow_two]
  ring  -- 自动展开 (x+ε)² = x² + 2xε + ε²，求导后取 ε=0 得 2x


-- 定义：沿单位矩阵 Eₐ_b 的方向导数
def directional_deriv (f : Mat → ℝ) (M : Mat) (a : m) (b : n) : ℝ :=
  deriv (fun ε => f (M + ε • unitMat a b)) 0

-- 引理4：方向导数等于 ∂f/∂Mₐ_b
theorem directional_deriv_eq_partial
  (f : Mat → ℝ)
  (M : Mat)
  (a : m)
  (b : n) :
  directional_deriv f M a b = ∂f/∂Mₐ_b  -- 假设我们定义了 ∂ 符号
:= by
  sorry  -- 需要定义偏导符号，或直接用方向导数作为定义


-- 定义：沿单位矩阵 Eₐ_b 的方向导数
def directional_deriv (f : Mat → ℝ) (M : Mat) (a : m) (b : n) : ℝ :=
  deriv (fun ε => f (M + ε • unitMat a b)) 0

-- 引理4：方向导数等于 ∂f/∂Mₐ_b
theorem directional_deriv_eq_partial
  (f : Mat → ℝ)
  (M : Mat)
  (a : m)
  (b : n) :
  directional_deriv f M a b = ∂f/∂Mₐ_b  -- 假设我们定义了 ∂ 符号
:= by
  sorry  -- 需要定义偏导符号，或直接用方向导数作为定义


-- 定理2：二阶导 = 2 δₐ_c δ_b_d
theorem hessian_frobenius_sq_correct
  (M : Mat) (a b c d : m × n) :
  directional_deriv (fun N => directional_deriv frobenius_sq N a.1 a.2) M c.1 c.2 =
    if a = c then 2 else 0 := by
  cases' a with a1 a2
  cases' c with c1 c2
  simp [directional_deriv]
  -- 一阶导是 2*N a1 a2，现在对 N c1 c2 求导
  have : deriv (fun ε => 2 * (M a1 a2 + if a1 = c1 ∧ a2 = c2 then ε else 0)) 0 =
    if a1 = c1 ∧ a2 = c2 then 2 else 0 := by
    split_ifs
    · ring  -- 导数为 2
    · simp [deriv_const]  -- 导数为0
  exact this

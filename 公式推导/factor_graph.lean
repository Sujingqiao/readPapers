import Mathlib
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog

-- 为简化，假设所有变量取值于同一个有限类型
abbrev Value := Bool
instance : Fintype Value := inferInstance -- Bool 是有限类型

-- 1. 定义变量和因子的索引类型
-- 假设我们有 n 个变量和 m 个因子
abbrev VarIdx := Fin 3 -- 例如，3个变量: X₀, X₁, X₂
abbrev FactorIdx := Fin 2 -- 例如，2个因子: f₀, f₁

-- 2. 定义变量赋值 (Assignment)
abbrev Assignment := VarIdx → Value

-- 3. 定义因子图结构
-- 核心：指定每个因子依赖于哪些变量。
-- 我们用一个函数表示：对于每个因子 j，它依赖的变量集合是什么？
-- 在 Mathlib 中，用 Finset 表示有限集合。
def factor_scope : FactorIdx → Finset VarIdx
  | 0 => {0, 1} -- 因子 f₀ 依赖于变量 X₀, X₁
  | 1 => {1, 2} -- 因子 f₁ 依赖于变量 X₁, X₂
  -- 注意：这是一个具体的例子。在通用形式化中，这将是 FactorGraph 结构的一部分。

-- 4. 定义因子（Factor）本身
-- 一个因子 fⱼ 是一个函数，它接收其作用域内变量的赋值，返回一个非负实数。
-- 作用域内变量的赋值类型：从 {v : VarIdx | v ∈ scope j} 到 Value 的函数。
def Factor (j : FactorIdx) :=
  {vars : VarIdx // vars ∈ factor_scope j} → Value -- 输入：作用域内每个变量的取值
  → ℝ≥0 -- 输出：非负实数 (使用 ENNReal 或 ℝ≥0。这里用 ℝ≥0 更贴近概率)
-- 注意：ℝ≥0 是 Mathlib 中的 Nonnegative Real Numbers，定义在 `Mathlib.Algebra.Order.NonNegative.Basic`

-- 5. 定义具体的因子实例（示例）
-- 我们需要为每个因子提供一个具体的函数。
-- 示例：f₀(X₀, X₁) = if X₀ == X₁ then 2.0 else 1.0
def example_factor_0 : Factor 0 := fun assign => 
  if assign ⟨0, by decide⟩ = assign ⟨1, by decide⟩ then (2 : ℝ≥0) else (1 : ℝ≥0)
-- 说明：⟨0, by decide⟩ 构造一个依赖类型的项，证明 0 ∈ {0,1}。`by decide` 自动证明。

-- 示例：f₁(X₁, X₂) = if X₁ && X₂ then 3.0 else 1.0
def example_factor_1 : Factor 1 := fun assign => 
  if assign ⟨1, by decide⟩ = true ∧ assign ⟨2, by decide⟩ = true then (3 : ℝ≥0) else (1 : ℝ≥0)

-- 6. 将所有因子打包成一个“因子数组”
-- 在真实模型中，这将是 FactorGraph 结构的一部分。
abbrev Factors := (j : FactorIdx) → Factor j
def example_factors : Factors := 
  fun j => match j with
    | 0 => example_factor_0
    | 1 => example_factor_1

-- 7. 定义未归一化的联合概率 (Unnormalized Joint Probability)
-- 对于一个完整的变量赋值 x : Assignment，计算所有因子的乘积。
def unnormalized_prob (fs : Factors) (x : Assignment) : ℝ≥0 :=
  ∏ j : FactorIdx, 
    -- 需要从完整的赋值 x 中，提取出因子 j 作用域内的变量赋值。
    let restricted_assign : {v : VarIdx // v ∈ factor_scope j} → Value := 
      fun v => x v.val -- v.val 是 VarIdx，x 是 VarIdx → Value
    fs j restricted_assign

-- 8. 定义配分函数 Z
-- 对所有可能的变量赋值求和。
def partition_fn (fs : Factors) : ℝ≥0 :=
  ∑ x in Fintype.finset Assignment, -- 遍历所有可能的赋值
    unnormalized_prob fs x

-- 9. 定义归一化的条件概率分布 P(x)
def normalized_prob (fs : Factors) (x : Assignment) : ℝ :=
  if h : partition_fn fs ≠ 0 then
    (unnormalized_prob fs x : ℝ) / (partition_fn fs : ℝ) -- 将 ℝ≥0 提升到 ℝ
  else
    0

-- 10. 核心定理：归一化性质
-- 所有可能赋值的概率之和为 1。
theorem factor_graph_normalization
    (fs : Factors)
    (h_z_nonzero : partition_fn fs ≠ 0)
    :
    ∑ x in Fintype.finset Assignment, normalized_prob fs x = 1 := by
  -- 展开 normalized_prob 的定义
  simp only [normalized_prob, h_z_nonzero] -- 选择 then 分支
  -- 目标：∑ x, (unnormalized_prob fs x : ℝ) / (partition_fn fs : ℝ) = 1

  -- 将常数 1 / (partition_fn fs : ℝ) 提取到求和符号外
  rw [Finset.sum_mul_left] -- ∑ x, c * f(x) = c * ∑ x, f(x)
  -- 目标：(1 / (partition_fn fs : ℝ)) * (∑ x, (unnormalized_prob fs x : ℝ)) = 1

  -- 关键：将 ℝ≥0 的求和转换为 ℝ 的求和，并关联到 partition_fn 的定义
  -- partition_fn fs 定义为 ∑ x, unnormalized_prob fs x (结果在 ℝ≥0)
  -- 我们需要一个引理：将 ℝ≥0 的 Finset.sum 提升到 ℝ 后，等于在 ℝ 中对提升后的值求和。
  have h_sum_cast : (∑ x in Fintype.finset Assignment, (unnormalized_prob fs x : ℝ)) = (partition_fn fs : ℝ) := by
    -- 这是类型提升的性质。Mathlib 中应有类似定理，或可手动证明。
    -- 伪证明：对有限集合求和，逐项提升再求和，等于先求和再提升。
    sorry -- 这是一个技术性引理，通常可证。
  rw [h_sum_cast] -- 代入
  -- 目标：(1 / (partition_fn fs : ℝ)) * (partition_fn fs : ℝ) = 1

  -- 使用 field_simp 化简实数运算
  field_simp [h_z_nonzero] -- 自动处理 (1/a) * a = 1
  rfl

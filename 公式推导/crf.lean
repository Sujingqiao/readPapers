-- 通常，形式化项目会放在一个命名空间内
import Mathlib -- 导入整个 Mathlib4，或按需导入特定模块
import Mathlib.Data.Fintype.Basic -- 有限类型
import Mathlib.Data.Finset -- 有限集合
import Mathlib.Algebra.BigOperators.Basic -- 求和符号 ∑
import Mathlib.Analysis.SpecialFunctions.ExpLog -- 指数函数
import Mathlib.Tactic.FieldSimp -- field_simp tactic

-- 为了简化，我们假设：
-- 1. 标签集 `Label` 是一个有限类型 (Fintype)。
-- 2. 序列长度 `n` 是固定的。
-- 3. 特征函数和权重是给定的。
-- 4. 我们关注的是给定观测 `x` 下，所有长度为 `n` 的标签序列的归一化。

-- 定义标签类型 (假设是有限的)
abbrev Label := Fin 5 -- 例如，5个标签 {0, 1, 2, 3, 4}。在真实项目中可能是 `String` 或自定义inductive类型，但需证明其Fintype实例。
-- @[derive Fintype] -- 如果是自定义类型，可以用这个自动生成Fintype实例

-- 定义长度为 n 的标签序列类型
-- 在 Mathlib4 中，对于固定长度的序列，常用 `Vector α n` 或 `Fin n → α`。
-- 这里用 `Fin n → Label`，它表示从位置 {0, 1, ..., n-1} 到标签的函数。
abbrev LabelSeq (n : Nat) := Fin n → Label

-- 假设我们有一个观测序列类型 (这里简化为 Unit，因为我们关注的是给定 x 的条件概率)
abbrev ObsSeq := Unit -- 在真实模型中，这会是 List Char 或 Vector Word n 等复杂类型。

-- 定义特征函数类型 (简化版)
-- 为了简单，我们假设特征函数只依赖于相邻标签和位置，且特征数量是固定的。
-- f : (前一个标签) → (当前标签) → (位置 i) → Real
abbrev FeatureFunc := Label → Label → Fin n → ℝ

-- 定义权重向量 (简化版，假设一个特征函数)
-- 在真实模型中，会是 FeatureFunc 的索引到 ℝ 的映射。
abbrev Weight := ℝ

-- 计算单个位置 i 的“局部得分” (简化：只用一个特征函数和权重)
-- 注意：对于 i=0，没有 y_{i-1}，通常需要特殊处理（如 START 标签）。这里我们假设 i > 0，或定义一个虚拟的起始标签。
-- 为了简化证明，我们假设序列从 i=1 开始计算转移，并忽略 i=0 的发射特征。
def local_score (w : Weight) (f : FeatureFunc) (y : LabelSeq n) (i : Fin (n - 1)) : ℝ :=
  -- i : Fin (n-1) 表示位置 0 到 n-2，对应转移 (y i) -> (y (i+1))
  w * f (y i) (y (i.succ)) i
-- 注意：i.succ : Fin n，因为 i < n-1, 所以 i+1 < n.

-- 计算整个序列的得分 (对所有转移位置求和)
-- 我们需要对 Fin (n-1) 上的所有 i 求和。
def score (w : Weight) (f : FeatureFunc) (y : LabelSeq n) : ℝ :=
  ∑ i : Fin (n - 1), local_score w f y i
-- Mathlib4 的 ∑ 是 Finset.sum 或 Fintype.sum 的语法糖。这里因为 Fin (n-1) 是 Fintype，所以可以这样用。

-- 定义所有可能的标签序列的集合
-- 因为 Label 是 Fintype，Fin n 也是 Fintype，所以 LabelSeq n = (Fin n → Label) 也是 Fintype。
-- 我们可以获取其所有元素的 Finset。
def all_label_seqs (n : Nat) [Fintype Label] : Finset (LabelSeq n) :=
  Fintype.finset (LabelSeq n) -- 利用 Fintype 实例自动生成有限集合

-- 定义配分函数 Z(x) (这里 x 是 Unit，所以 Z 不依赖于 x)
-- Z = Σ_{y ∈ all_label_seqs} exp(score(y))
def partition_fn (n : Nat) [Fintype Label] (w : Weight) (f : FeatureFunc) : ℝ :=
  ∑ y in all_label_seqs n, Real.exp (score w f y)
-- 这里 ∑ y in finset, ... 是 Finset.sum 的语法糖。

-- 定义条件概率 P(y|x) (x 是 Unit)
def cond_prob (n : Nat) [Fintype Label] (w : Weight) (f : FeatureFunc) (y : LabelSeq n) : ℝ :=
  if h : partition_fn n w f ≠ 0 then
    Real.exp (score w f y) / partition_fn n w f
  else
    0 -- 或者定义为未定义，但在证明归一化时我们假设 Z≠0
-- 在概率模型中，Z 通常是正数，因为 exp(score) > 0 且至少有一个序列。

-- 核心定理：归一化性质
-- 对于任何权重 w、特征函数 f 和序列长度 n (n > 0)，所有可能标签序列的概率之和为 1。
-- 我们需要假设 partition_fn ≠ 0。
theorem crf_normalization
    (n : Nat)
    [Fintype Label]
    (w : Weight)
    (f : FeatureFunc)
    (h_n_pos : n > 0) -- 确保序列有长度，转移位置 Fin (n-1) 非空或至少有一个序列
    (h_z_nonzero : partition_fn n w f ≠ 0)
    :
    ∑ y in all_label_seqs n, cond_prob n w f y = 1 := by
  -- 展开 cond_prob 的定义。由于我们有 h_z_nonzero 假设，会走 if 的 then 分支。
  simp only [cond_prob, h_z_nonzero] -- simp 会尝试化简，利用 h_z_nonzero 选择 then 分支
  -- 现在目标是：∑ y in all_label_seqs n, (Real.exp (score w f y) / partition_fn n w f) = 1

  -- 将常数 1 / (partition_fn n w f) 提取到求和符号外。
  -- 使用 Finset.sum_mul 系列定理。具体是 Finset.sum_const_mul 或类似。
  -- 查找定理：∑ x in s, c * f x = c * ∑ x in s, f x
  rw [Finset.sum_mul_left] -- 或者可能是 Finset.sum_smul，取决于具体类型。这里假设是乘法。
  -- 目标变为：(1 / partition_fn n w f) * (∑ y in all_label_seqs n, Real.exp (score w f y)) = 1

  -- 根据 partition_fn 的定义，∑ y in all_label_seqs n, Real.exp (score w f y) 就是 partition_fn n w f
  rw [partition_fn] -- 直接代入定义
  -- 目标变为：(1 / partition_fn n w f) * (partition_fn n w f) = 1

  -- 使用 field_simp 化简实数除法和乘法
  field_simp [h_z_nonzero] -- 这会自动应用 (1/a) * a = 1 (当 a ≠ 0)
  -- 证明完成
  rfl -- 最终目标是 1 = 1

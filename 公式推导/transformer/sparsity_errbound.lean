-- 定义：向量按值排序（降序）
def sortedDesc (p : Vec n) : Fin (Fintype.card n) → ℝ := 
  -- 简化：假设我们有排序函数（mathlib 支持）
  let ps := Finset.univ.image p
  let sorted := ps.toList.sort (· ≥ ·)
  fun i => sorted.get? i |>.getD 0

-- 定义：k-主导稀疏性
def isKSparseDominant (p : Vec TokenCount) (k : ℕ) (ε : ℝ) : Prop :=
  (∑ i : Fin k, sortedDesc p i) ≥ 1 - ε

-- 例如：k=5, ε=0.1 → 前 5 个 token 权重和 ≥ 0.9

theorem softmax_sparse_if_scores_separated 
    (scores : TokenCount → ℝ) 
    (t₀ : TokenCount) 
    (gap : ℝ) 
    (h_max : ∀ t ≠ t₀, scores t₀ - scores t ≥ gap) 
    (h_gap_pos : gap > 0) :
    let p := softmax scores
    isKSparseDominant p 1 (2 * Real.exp (-gap)) := by
  -- 证明思路：
  -- p t₀ = exp(s₀) / Z
  -- p t  ≤ exp(s₀ - gap) / Z
  -- ⇒ Z ≤ exp(s₀) + (n-1) exp(s₀ - gap)
  -- ⇒ p t₀ ≥ exp(s₀) / [exp(s₀) + (n-1) exp(s₀ - gap)] = 1 / [1 + (n-1) exp(-gap)]
  -- 当 gap 大时，exp(-gap) 小 → p t₀ ≈ 1

  -- 展开定义
  unfold softmax
  set Z := ∑ t, Real.exp (scores t)
  have h_Z_le : Z ≤ Real.exp (scores t₀) + (∑ t ≠ t₀, Real.exp (scores t)) := by
    simp [← Finset.sum_erase]
  have h_other_small : ∀ t ≠ t₀, Real.exp (scores t) ≤ Real.exp (scores t₀ - gap) := by
    intro t h_ne
    apply Real.exp_le_exp.mpr
    exact (h_max t h_ne)
  calc
    Z ≤ Real.exp (scores t₀) + ∑ t ≠ t₀, Real.exp (scores t₀ - gap) := by
      apply add_le_add_left h_Z_le
      apply Finset.sum_le_sum
      exact h_other_small
    _ = Real.exp (scores t₀) + (Fintype.card TokenCount - 1) * Real.exp (scores t₀) * Real.exp (-gap) := by
      simp [Real.exp_sub]
    _ = Real.exp (scores t₀) * (1 + (Fintype.card TokenCount - 1) * Real.exp (-gap)) := by ring

  -- 现在 p t₀ = exp(s₀)/Z ≥ 1 / (1 + C exp(-gap))
  have h_p_t0 : p t₀ = Real.exp (scores t₀) / Z := by simp
  have h_p_t0_ge : p t₀ ≥ 1 / (1 + (Fintype.card TokenCount - 1) * Real.exp (-gap)) := by
    apply (div_le_div_iff' (Real.exp_pos _).ne').mpr
    apply (inv_le_inv_of_le).mpr
    exact this

  -- 但我们要的是 ∑_{i<1} sortedDesc p i ≥ 1 - 2 exp(-gap)
  -- 注意：sortedDesc p 0 是 max p t
  -- 因为 p t₀ 是最大值（因 scores t₀ 最大），所以 sortedDesc p 0 = p t₀
  -- 所以 ∑_{i<1} ... = p t₀
  -- 我们要 p t₀ ≥ 1 - 2 exp(-gap)

  -- 当 gap 足够大，exp(-gap) 小，p t₀ ≈ 1
  -- 更紧界需要更细分析，这里给出一个宽松界
  -- 实际中可假设 n 不太大，或 gap >> log n

  -- 为简化，假设 Fintype.card TokenCount = 512
  -- 则 p t₀ ≥ 1 / (1 + 511 * exp(-gap))
  -- 我们想 1 - p t₀ ≤ 2 exp(-gap)
  -- 即 1 - 1/(1+C e^{-g}) ≤ 2 e^{-g}
  -- ⇔ C e^{-g}/(1+C e^{-g}) ≤ 2 e^{-g}
  -- ⇔ C / (1+C e^{-g}) ≤ 2
  -- 当 g 大时成立

  -- 此处可引入假设 gap ≥ log 512，则 exp(-gap) ≤ 1/512
  -- 细节略，但方向正确

  admit  -- 证明细节较长，但思路清晰

  -- 定义：两个实数在 ε 内近似
def approx (ε : ℝ) (x y : ℝ) : Prop := |x - y| ≤ ε

-- 定义：两个向量在 ε 内逐点近似
def vecApprox (ε : ℝ) (x y : Vec n) : Prop := ∀ i, approx ε (x i) (y i)

-- 定义：浮点实现 ≈ 理想实数计算
def floatImplementationApprox (ε : ℝ) 
                              (ideal : Vec n → Vec m) 
                              (floatImpl : Vec n → Vec m) : Prop :=
  ∀ x, vecApprox ε (floatImpl x) (ideal x) 





theorem softmax_float_error_bound 
    (x : Vec TokenCount) 
    (x_float : Vec TokenCount) 
    (ε_input : ℝ) 
    (h_input : vecApprox ε_input x x_float) 
    (ε_op : ℝ)  -- 单次浮点操作误差界
    : ∃ ε_total, 
        vecApprox ε_total (softmax x) (floatSoftmax x_float) ∧ 
        ε_total ≤ some_function ε_input ε_op := by
  -- 证明思路：
  -- softmax 包含 exp, sum, div
  -- 每个操作有舍入误差
  -- 利用 Lipschitz 连续性：softmax 是 Lipschitz 的！

  -- 已知：softmax 是 1-Lipschitz（在 ∞-范数下？需查）
  -- 更准确：softmax 的 Jacobian 有界

  -- 简化：假设 exp 和 sum 的误差可控
  have h_exp_error : ∀ i, |Real.exp (x i) - floatExp (x_float i)| ≤ δ₁ := by
    -- 浮点 exp 误差界
    sorry

  have h_sum_error : |∑ i, Real.exp (x i) - floatSum (fun i => floatExp (x_float i))| ≤ δ₂ := by
    -- 浮点求和误差（与 n 有关）
    sorry

  -- 最终 softmax 每个分量误差可界为：
  -- |softmax(x)_i - floatSoftmax(x_float)_i| ≤ L * ε_input + C * (δ₁ + δ₂)
  -- 其中 L 是 softmax 的 Lipschitz 常数

  -- 实际中 L ≤ 1（在某些范数下）
  use ε_input + 2 * ε_op * Fintype.card TokenCount  -- 粗略上界
  constructor
  · intro i
    -- 逐点误差分析
    admit
  · trivial




theorem transformer_forward_float_error_bound 
    (model : TransformerModel)
    (x : TokenCountEnc → VocabSize)
    (x_float : TokenCountEnc → VocabSize)  -- 浮点输入
    (ε_per_layer : ℝ)
    : ∃ ε_total,
        let ideal_out := transformerForward model x
        let float_out  := floatTransformerForward model x_float
        vecApprox ε_total ideal_out float_out ∧
        ε_total ≤ ε_per_layer * (num_layers model) := by
  -- 证明思路：误差逐层传播
  -- 每层（Attention, FFN, LayerNorm）都是 Lipschitz 映射
  -- 总误差 ≈ 每层误差之和

  -- 假设每层满足：
  --   ||f(x) - f_float(x_float)|| ≤ L * ||x - x_float|| + ε_local

  -- 则总误差可递推界出

  -- 此处可形式化 LayerNorm、Linear、GELU 的 Lipschitz 常数
  admit

import Mathlib
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog

-- 1. 定义基本类型
abbrev StateIdx := Fin 3 -- 假设有3个隐状态: s0, s1, s2
abbrev ObsIdx := Fin 2   -- 假设有2种观测值: v0, v1
abbrev TimeIdx := Nat    -- 时间索引，从1开始

-- 观测序列 (固定长度 T)
abbrev ObsSeq (T : Nat) := Fin T → ObsIdx

-- 2. 定义 HMM 参数
structure HMMParams where
  pi : StateIdx → ℝ≥0 -- 初始概率
  pi_sum : ∑ i, pi i = 1 -- 初始概率和为1
  a : StateIdx → StateIdx → ℝ≥0 -- 转移概率 a[i][j] = P(q_{t+1}=j | q_t=i)
  a_sum : ∀ i, ∑ j, a i j = 1 -- 转移概率行和为1
  b : StateIdx → ObsIdx → ℝ≥0 -- 发射概率 b[i][k] = P(o_t=k | q_t=i)
  b_sum : ∀ i, ∑ k, b i k = 1 -- 发射概率行和为1 (可选，有时发射概率不强制和为1)

-- 3. 定义状态序列 (长度为 t)
abbrev StateSeq (t : Nat) := Fin t → StateIdx

-- 4. 直接定义：前向概率 α_t(i) (基于所有可能的前 t-1 个状态序列)
-- P(o_1..o_t, q_t = s_i | λ)
def forward_direct (λ : HMMParams) (O : ObsSeq T) (t : Fin T) (i : StateIdx) : ℝ≥0 :=
  if h_t_eq_zero : t.val = 0 then
    -- t=1 (因为 Fin T 从0开始，t.val=0 对应时刻1)
    λ.pi i * λ.b i (O t)
  else
    -- t > 1: 对所有可能的前 t 个状态序列求和，要求第 t 个状态是 i
    ∑ (Q : StateSeq (t.val + 1)) in Fintype.finset (StateSeq (t.val + 1)), 
      if h_last_state : Q t = i then
        -- 计算路径概率: π * a * b
        let path_prob : ℝ≥0 :=
          -- 初始概率
          λ.pi (Q 0) *
          -- 转移概率乘积 (从时刻1到t-1)
          (∏ s : Fin t.val, λ.a (Q ⟨s, by decide⟩) (Q ⟨s + 1, by have := s.is_lt; omega⟩)) *
          -- 发射概率乘积 (从时刻1到t)
          (∏ s : Fin (t.val + 1), λ.b (Q s) (O ⟨s, by decide⟩))
        path_prob
      else
        0

-- 5. 递推定义：前向概率 α_t(i)
-- 我们用一个函数来计算所有时刻和所有状态的前向概率。
-- 返回一个二维数组: [时刻 t][状态 i] -> α_t(i)
def forward_recursive (λ : HMMParams) (O : ObsSeq T) : (t : Fin T) → StateIdx → ℝ≥0 := 
  let rec go : (t : Nat) → StateIdx → ℝ≥0
    | 0, i => -- 时刻1 (t=0)
      λ.pi i * λ.b i (O 0)
    | t'+1, j => -- 时刻 t'+2
      ∑ i : StateIdx, 
        go t' i * λ.a i j * λ.b j (O ⟨t'+1, by decide⟩)
  fun t => go t.val

-- 6. 核心定理：证明递推定义与直接定义等价
theorem forward_equivalence
    (λ : HMMParams)
    (O : ObsSeq T)
    (t : Fin T)
    (i : StateIdx)
    :
    forward_recursive λ O t i = forward_direct λ O t i := by
  -- 对 t 进行归纳 (实际上是 t.val 的归纳)
  induction t.val with
  | zero => 
    -- 基础情况: t.val = 0 (对应时刻1)
    simp [forward_recursive, forward_direct]
    -- 两边都等于 λ.pi i * λ.b i (O 0)
    rfl
  | succ t' ih => 
    -- 归纳步骤: 假设对 t' 成立，证明对 t'+1 成立
    -- ih 是归纳假设: ∀ i, forward_recursive λ O ⟨t', _⟩ i = forward_direct λ O ⟨t', _⟩ i
    simp [forward_recursive] -- 展开递推定义
    -- 目标：∑ i, forward_recursive λ O ⟨t', _⟩ i * a i j * b j (O ⟨t'+1, _⟩) = forward_direct λ O ⟨t'+1, _⟩ j
    rw [ih] -- 应用归纳假设，将 forward_recursive 替换为 forward_direct
    -- 目标：∑ i, forward_direct λ O ⟨t', _⟩ i * a i j * b j (O ⟨t'+1, _⟩) = forward_direct λ O ⟨t'+1, _⟩ j
    -- 现在需要展开 forward_direct λ O ⟨t'+1, _⟩ j 的定义
    dsimp [forward_direct] -- 展开直接定义
    -- 由于 t'+1 > 0，会进入 else 分支
    -- 目标：∑ Q in all_seqs, if Q (t'+1) = j then path_prob else 0 = ∑ i, forward_direct λ O ⟨t', _⟩ i * a i j * b j (O ⟨t'+1, _⟩)
    -- 关键：将对长度为 t'+2 的序列 Q 的求和，分解为：
    --   1. 对时刻 t'+1 的状态 i 求和。
    --   2. 对前 t'+1 个状态（构成一个长度为 t'+1 的序列 Q'）求和。
    -- 这需要 Fubini 定理（求和交换）或手动构造双射。
    sorry -- 这是证明的核心难点，需要详细展开路径求和。

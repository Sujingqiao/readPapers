import Mathlib

namespace SwitchTransformer

-- 基础类型
abbrev BatchSize := Nat
abbrev TokenSize := Nat
abbrev ExpertSize := Nat
abbrev HiddenDim := Nat
abbrev NumExperts := Nat

-- 输入：一批 token 的 embedding
abbrev InputEmbeddings := Tensor ℝ BatchSize TokenSize HiddenDim

-- 专家索引
abbrev ExpertIndex := Fin NumExperts  -- 用 Fin 确保不越界

-- 路由概率（每个 token 对每个专家的概率）
abbrev RouterProbs := Tensor ℝ BatchSize TokenSize NumExperts

-- 门控值（每个 token 对选中专家的置信度）
abbrev GatingWeights := Tensor ℝ BatchSize TokenSize

-- 专家分配（每个 token 选哪个专家）
abbrev ExpertAssignment := Tensor ExpertIndex BatchSize TokenSize

-- 专家输出（每个专家处理后的结果）
abbrev ExpertOutput := Tensor ℝ BatchSize TokenSize HiddenDim

-- 损失值
abbrev Loss := ℝ

def SwitchRouter(x):
    logits = x @ W  # 投影到专家维度
    probs = softmax(logits, axis=-1)
    expert_index = argmax(probs, axis=-1)
    gate = probs.max(axis=-1)
    return expert_index, gate


-- 辅助：对数组应用 softmax（简化版，仅用于演示）
def softmax1d (xs : Array ℝ) : Array ℝ := 
  let exps := xs.map (·.exp)
  let sum := exps.foldl (·+·) 0
  exps.map (· / sum)

-- 辅助：找最大值及其索引
def argmaxWithVal (xs : Array ℝ) : ExpertIndex × ℝ := 
  let rec go (i : Nat) (maxIdx : Nat) (maxVal : ℝ) : Nat × ℝ :=
    if i >= xs.size then (maxIdx, maxVal)
    else if xs[i] > maxVal then go (i+1) i xs[i]
    else go (i+1) maxIdx maxVal
  let (idx, val) := go 1 0 xs[0]
  (⟨idx, by decide⟩, val)  -- 包装为 Fin 类型

-- 核心：路由器
def SwitchRouter (x : InputEmbeddings) (W : Tensor ℝ HiddenDim NumExperts) : 
    ExpertAssignment × GatingWeights := by
  -- 简化：假设 x 是 [B, T, H], W 是 [H, E]
  let logits : RouterProbs := 
    x.map (fun token => 
      token.map (fun embed => 
        embed.zipWith (·*·) W.transpose |>.sum  -- 简化矩阵乘
      )
    )
  let probs : RouterProbs := logits.map (·.map softmax1d)
  let assignments : ExpertAssignment := probs.map (·.map (fun ps => (argmaxWithVal ps).1))
  let gates : GatingWeights := probs.map (·.map (fun ps => (argmaxWithVal ps).2))
  (assignments, gates)




def SwitchFeedForward(x):
    expert_idx, gate = SwitchRouter(x)
    y = zeros_like(x)
    for e in experts:
        mask = (expert_idx == e)
        tokens = x[mask] * gate[mask].unsqueeze(-1)
        y[mask] = Expert(e)(tokens)
    return y



-- 专家函数类型：输入一批 token，输出同形状
abbrev ExpertFunction := InputEmbeddings → InputEmbeddings

-- 专家集合
abbrev Experts := Array ExpertFunction

-- 核心：Switch 前向
def SwitchFeedForward (x : InputEmbeddings) 
    (W_router : Tensor ℝ HiddenDim NumExperts)
    (experts : Experts) : 
    ExpertOutput := 
  let (assignments, gates) := SwitchRouter x W_router
  -- 初始化输出
  let y : ExpertOutput := x.map (·.map (·.map (· * 0)))  -- 全零
  -- 对每个专家 e
  let rec dispatch (e : ExpertIndex) (out : ExpertOutput) : ExpertOutput :=
    if e.val >= experts.size then out
    else
      -- 构造 mask：哪些 token 分配给专家 e
      let mask : Array (Array Bool) := assignments.map (·.map (· = e))
      -- 提取 tokens：x[mask] * gate[mask]
      let selectedTokens : InputEmbeddings := 
        x.zipWith (·.zipWith (·.zipWith (fun x_val gate_val mask_val =>
          if mask_val then x_val * gate_val else 0
        ))) gates mask
      -- 专家计算
      let expertOut : InputEmbeddings := experts[e.val] selectedTokens
      -- 合并回输出（覆盖 mask 位置）
      let newOut : ExpertOutput := 
        out.zipWith (·.zipWith (·.zipWith (fun out_val expert_val mask_val =>
          if mask_val then expert_val else out_val
        ))) expertOut mask
      dispatch (⟨e.val + 1, by decide⟩) newOut
  dispatch ⟨0, by decide⟩ y



def LoadBalancingLoss(router_probs, expert_mask):
    # fraction of tokens routed to each expert
    f = mean(expert_mask, axis=[0,1])
    # fraction of router probability allocated to each expert
    p = mean(router_probs, axis=[0,1])
    return NumExperts * dot(f, p)



-- 辅助：对二维数组求均值
def mean2d (xs : Array (Array ℝ)) : ℝ :=
  let total := xs.foldl (·+·) 0 |>.foldl (·+·) 0
  let count := xs.size * (if xs.isEmpty then 1 else xs[0].size)
  total / count

-- 辅助：点积
def dot (a b : Array ℝ) : ℝ :=
  a.zipWith (·*·) b |>.foldl (·+·) 0

-- 核心：负载均衡损失
def LoadBalancingLoss (routerProbs : RouterProbs) (assignments : ExpertAssignment) : Loss := 
  -- 构造 expert_mask：one-hot 形式
  let expertMask : RouterProbs := 
    assignments.map (·.map (fun e => 
      (Array.mk (fun i => if i = e.val then 1 else 0) NumExperts)
    ))
  -- f: 每个专家被分配的 token 比例
  let f : Array ℝ := mean2d expertMask.transpose  -- [E]
  -- p: 每个专家的平均路由概率
  let p : Array ℝ := mean2d routerProbs.transpose  -- [E]
  -- 损失 = NumExperts * f · p
  NumExperts * dot f p



  def TrainingStep (x : InputEmbeddings)
    (W_router : Tensor ℝ HiddenDim NumExperts)
    (experts : Experts) : 
    ExpertOutput × Loss := 
  let (assignments, gates) := SwitchRouter x W_router
  let y := SwitchFeedForward x W_router experts
  let routerProbs := ... -- 需从 SwitchRouter 内部暴露（略）
  let loss := LoadBalancingLoss routerProbs assignments
  (y, loss)



-- 修改：返回三元组
def SwitchRouter (x : InputEmbeddings) (W : Tensor ℝ HiddenDim NumExperts) : 
    ExpertAssignment × GatingWeights × RouterProbs := by
  let logits : RouterProbs := 
    x.map (fun token => 
      token.map (fun embed => 
        embed.zipWith (·*·) W.transpose |>.sum
      )
    )
  let probs : RouterProbs := logits.map (·.map softmax1d)
  let assignments : ExpertAssignment := probs.map (·.map (fun ps => (argmaxWithVal ps).1))
  let gates : GatingWeights := probs.map (·.map (fun ps => (argmaxWithVal ps).2))
  (assignments, gates, probs)  -- ← 返回完整概率



logits = x @ W
noisy_logits = logits + noise
probs = softmax(noisy_logits)
expert_idx = argmax(probs)



-- 噪声函数（简化：加固定扰动，实际可用随机，但 Lean 中我们用确定性扰动做形式化）
def deterministicNoise (seed : Nat) (shape : (Nat × Nat × Nat)) : RouterProbs := 
  let (b, t, e) := shape
  Array.mk (fun _ => 
    Array.mk (fun _ => 
      Array.mk (fun i => (seed + i) % 1000 / 1000.0 - 0.5) e  -- [-0.5, 0.5) 扰动
    ) t
  ) b

-- Noisy Switch Router
def NoisySwitchRouter (x : InputEmbeddings) (W : Tensor ℝ HiddenDim NumExperts) (noiseSeed : Nat) : 
    ExpertAssignment × GatingWeights × RouterProbs := 
  let logits : RouterProbs := 
    x.map (fun token => 
      token.map (fun embed => 
        embed.zipWith (·*·) W.transpose |>.sum
      )
    )
  let noise := deterministicNoise noiseSeed (x.size, x[0].size, NumExperts)
  let noisyLogits := logits.zipWith (·.zipWith (·.zipWith (·+·))) noise
  let probs : RouterProbs := noisyLogits.map (·.map softmax1d)
  let assignments : ExpertAssignment := probs.map (·.map (fun ps => (argmaxWithVal ps).1))
  let gates : GatingWeights := probs.map (·.map (fun ps => (argmaxWithVal ps).2))
  (assignments, gates, probs)



  -- 假设 NumExperts > 0
theorem LoadBalancingLoss_uniform_min 
    (hE : NumExperts > 0)
    (routerProbs : RouterProbs)
    (assignments : ExpertAssignment) :
    -- 假设：f 和 p 都是均匀分布
    (∀ e : Fin NumExperts, 
      mean2d (assignments.map (·.map (· = e))) = 1 / NumExperts) →
    (∀ e : Fin NumExperts, 
      mean2d (routerProbs.map (·.map (·[e.val]))) = 1 / NumExperts) →
    LoadBalancingLoss routerProbs assignments = 1 := by
  intro hf hp
  let f := Array.mk (fun i => 1 / NumExperts) NumExperts
  let p := Array.mk (fun i => 1 / NumExperts) NumExperts
  let dot_fp := dot f p
  have h_dot : dot_fp = NumExperts * (1/NumExperts) * (1/NumExperts) := by
    simp [dot, f, p, Fin.sum_univ, Nat.cast_inv, hE]
    ring
  simp [LoadBalancingLoss, h_dot, hf, hp]  -- 需要更多引理，此处示意
  sorry




  -- 可微函数类型（占位符）
abbrev DifferentiableFunction := {f : ℝ → ℝ} × {df : ℝ → ℝ}

-- 可微 softmax（简化：标量版，向量版类似）
def diffSoftmax1d (xs : Array ℝ) : Array ℝ × (Array ℝ → Array ℝ) := 
  let exps := xs.map (·.exp)
  let sum := exps.foldl (·+·) 0
  let probs := exps.map (· / sum)
  -- 梯度函数（略，实际需计算 Jacobian）
  let grad (dout : Array ℝ) : Array ℝ := 
    dout  -- 占位，实际需实现
  (probs, grad)

-- 可微 SwitchRouter（示意）
def DifferentiableSwitchRouter (x : InputEmbeddings) (W : Tensor ℝ HiddenDim NumExperts) :
    (ExpertAssignment × GatingWeights × RouterProbs) × (/* 梯度函数 */) := 
  -- 同前，但用 diffSoftmax1d
  sorry




  def TrainingStep (x : InputEmbeddings)
    (W_router : Tensor ℝ HiddenDim NumExperts)
    (experts : Experts) 
    (noiseSeed : Nat) : 
    ExpertOutput × Loss := 
  let (assignments, gates, routerProbs) := NoisySwitchRouter x W_router noiseSeed
  let y := SwitchFeedForward x W_router experts
  let loss := LoadBalancingLoss routerProbs assignments
  (y, loss)

-- 定理：如果路由完全均衡，则损失为1（需完整证明）
theorem TrainingStep_balanced_implies_loss_one 
    (h_balanced_f : ∀ e, mean2d (assignments.map (·.map (· = e))) = 1/NumExperts)
    (h_balanced_p : ∀ e, mean2d (routerProbs.map (·.map (·[e.val]))) = 1/NumExperts) :
    (TrainingStep x W_router experts noiseSeed).2 = 1 := by
  sorry  -- 依赖前面的定理



  

    

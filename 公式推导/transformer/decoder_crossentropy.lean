import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Analysis.SpecialFunctions.LogExp

-- 保持之前的定义
abbrev TokenCountEnc := Fin 512
abbrev TokenCountDec := Fin 256  -- 解码序列长度
abbrev EmbedDim := Fin 512
abbrev HeadCount := Fin 8
abbrev HiddenDim := Fin 2048
abbrev VocabSize := Fin 30000   -- 词表大小

-- 向量/矩阵
abbrev Vec (n : Type) [Fintype n] := n → ℝ
abbrev Mat (m n : Type) [Fintype m] [Fintype n] := m → n → ℝ

-- 嵌入类型
def Embedding := EmbedDim → ℝ
def EncoderOutput := TokenCountEnc → Embedding
def DecoderInput  := TokenCountDec → Embedding

  -- 定义掩码：t1 不能 attend 到 t2 如果 t2 > t1
def causalMask (t1 t2 : TokenCountDec) : ℝ :=
  if t2.val ≤ t1.val then 1 else 0  -- 1 表示允许，0 表示屏蔽

-- 注意：实际中在 softmax 前加 -∞，这里用乘法近似（简化）


  -- 交叉注意力头
def CrossAttentionHead :=
  { Wq : Linear EmbedDim EmbedDim  -- Q 来自解码器
  , Wk : Linear EmbedDim EmbedDim  -- K 来自编码器
  , Wv : Linear EmbedDim EmbedDim  -- V 来自编码器
  , Wo : Linear EmbedDim EmbedDim
  }

-- 计算交叉注意力
def computeCrossAttention 
    (head : CrossAttentionHead) 
    (dec_x : DecoderInput) 
    (enc_output : EncoderOutput) 
    : TokenCountDec → Embedding := 
  let Q := fun t_dec => applyLinear head.Wq (dec_x t_dec)
  let K := fun t_enc => applyLinear head.Wk (enc_output t_enc)
  let V := fun t_enc => applyLinear head.Wv (enc_output t_enc)
  
  let scores : TokenCountDec → TokenCountEnc → ℝ := 
    fun t_dec t_enc => (∑ d, Q t_dec d * K t_enc d) / Real.sqrt (512.0)
  
  -- 注意：这里没有因果掩码，因为是 enc-dec 注意力
  let attn_weights : TokenCountDec → TokenCountEnc → ℝ :=
    fun t_dec => softmax (fun t_enc => scores t_dec t_enc)
  
  fun t_out => 
    fun d_out => 
      ∑ t_enc : TokenCountEnc, attn_weights t_out t_enc * V t_enc d_out



-- 多头交叉注意力
def MultiHeadCrossAttention :=
  { heads : HeadCount → CrossAttentionHead
  , W_o : Linear EmbedDim EmbedDim
  }

def computeMultiHeadCross 
    (mha : MultiHeadCrossAttention) 
    (dec_x : DecoderInput) 
    (enc_output : EncoderOutput) 
    : DecoderInput := 
  let head_outputs : HeadCount → DecoderInput :=
    fun h => computeCrossAttention (mha.heads h) dec_x enc_output
  
  let summed : DecoderInput :=
    fun t => fun d => ∑ h, head_outputs h t d
  
  fun t => applyLinear mha.W_o (summed t)

-- 完整解码器层
def DecoderLayer :=
  { self_attn : MultiHeadAttention        -- Masked Self-Attention
  , cross_attn : MultiHeadCrossAttention  -- Encoder-Decoder Attention
  , ffn : FFN
  , ln1 : LayerNorm EmbedDim
  , ln2 : LayerNorm EmbedDim
  , ln3 : LayerNorm EmbedDim
  }

-- 前向传播（简化：忽略掩码实现细节）
def forwardDecoderLayer 
    (layer : DecoderLayer) 
    (dec_x : DecoderInput) 
    (enc_output : EncoderOutput) 
    : DecoderInput := 
  -- 1. Masked Self-Attention + Residual + LN
  let x1 := fun t => applyLayerNorm layer.ln1 
    (dec_x t + computeMultiHead layer.self_attn dec_x t)  -- 假设 computeMultiHead 支持掩码
  
  -- 2. Cross-Attention + Residual + LN
  let x2 := fun t => applyLayerNorm layer.ln2 
    (x1 t + computeMultiHeadCross layer.cross_attn x1 enc_output t)
  
  -- 3. FFN + Residual + LN
  let x3 := fun t => applyLayerNorm layer.ln3 
    (x2 t + applyFFN layer.ffn (x2 t))
  
  x3




-- 线性投影 + Softmax 得到词表概率
def OutputProjection :=
  { W_out : Linear EmbedDim VocabSize
  , b_out : Vec VocabSize
  }

def logits (proj : OutputProjection) (x : Embedding) : Vec VocabSize :=
  applyLinear proj.W_out x + proj.b_out

def outputProbs (proj : OutputProjection) (x : Embedding) : Vec VocabSize :=
  softmax (logits proj x)


-- 单个 token 的交叉熵损失
def crossEntropyLossAtToken 
    (proj : OutputProjection) 
    (x : Embedding)         -- 解码器输出
    (y_true : VocabSize)    -- 真实 token ID
    : ℝ := 
  let p := outputProbs proj x
  -Real.log (p y_true + 1e-8)  -- 加小数防止 log(0)

-- 整个序列的平均损失
def sequenceCrossEntropyLoss 
    (proj : OutputProjection)
    (decoder_outputs : DecoderInput)
    (target_tokens : TokenCountDec → VocabSize)  -- 真实目标序列
    : ℝ := 
  (∑ t : TokenCountDec, crossEntropyLossAtToken proj (decoder_outputs t) (target_tokens t)) 
    / Fintype.card TokenCountDec


-- 单个 token 的交叉熵损失
def crossEntropyLossAtToken 
    (proj : OutputProjection) 
    (x : Embedding)         -- 解码器输出
    (y_true : VocabSize)    -- 真实 token ID
    : ℝ := 
  let p := outputProbs proj x
  -Real.log (p y_true + 1e-8)  -- 加小数防止 log(0)

-- 整个序列的平均损失
def sequenceCrossEntropyLoss 
    (proj : OutputProjection)
    (decoder_outputs : DecoderInput)
    (target_tokens : TokenCountDec → VocabSize)  -- 真实目标序列
    : ℝ := 
  (∑ t : TokenCountDec, crossEntropyLossAtToken proj (decoder_outputs t) (target_tokens t)) 
    / Fintype.card TokenCountDec

theorem cross_attention_weights_sum_to_one 
    (head : CrossAttentionHead) 
    (dec_x : DecoderInput) 
    (enc_output : EncoderOutput) 
    (t_dec : TokenCountDec) :
    ∑ t_enc : TokenCountEnc, 
      (let scores := fun t1 t2 => ...; 
       softmax (fun t_enc => scores t_dec t_enc)) t_enc = 1 := by
  simp [softmax]
  ring
  -- 需要 Real.exp 的求和性质

theorem softmax_is_probability_distribution (z : Vec VocabSize) :
    0 ≤ ∑ i, softmax z i ∧ ∑ i, softmax z i = 1 := by
  -- 证明每一项 ≥0，且总和为 1
  constructor
  · apply Finset.sum_nonneg; intro; apply Real.exp_nonneg
  · simp [softmax]; ring


theorem cross_entropy_nonnegative 
    (proj : OutputProjection) (x : Embedding) (y : VocabSize) :
    crossEntropyLossAtToken proj x y ≥ 0 := by
  -- 因为 p_y ≤ 1 → -log(p_y) ≥ 0
  unfold crossEntropyLossAtToken
  have h1 : outputProbs proj x y ≤ 1 := by
    -- softmax 输出 ≤1
    sorry
  have h2 : Real.log (outputProbs proj x y + 1e-8) ≤ 0 := by
    apply Real.log_le_one; linarith
  linarith [neg_nonpos_of_nonneg (Real.log_nonneg ?_)]  -- 需补全


  -- 完整 Transformer 模型
def TransformerModel :=
  { encoder : TransformerEncoder
  , decoder_layers : Fin 6 → DecoderLayer
  , output_proj : OutputProjection
  }

-- 形式化前向传播 + 损失计算
def transformerLoss 
    (model : TransformerModel)
    (input_tokens : TokenCountEnc → VocabSize)   -- 编码输入
    (target_tokens : TokenCountDec → VocabSize)  -- 解码目标
    : ℝ := 
  -- 1. 编码
  let enc_embed := fun t => tokenEmbedding (input_tokens t)  -- 查嵌入表
  let enc_x := fun t => enc_embed t + posEnc t
  let enc_output := applyEncoder model.encoder enc_x

  -- 2. 解码（自回归，简化为单步）
  let dec_embed := fun t => tokenEmbedding (target_tokens t)  -- 假设 teacher forcing
  let dec_x := fun t => dec_embed t + posEnc t
  let dec_out := forwardDecoderLayer (model.decoder_layers 0) dec_x enc_output
  -- 可堆叠多层

  -- 3. 计算损失
  sequenceCrossEntropyLoss model.output_proj dec_out target_tokens

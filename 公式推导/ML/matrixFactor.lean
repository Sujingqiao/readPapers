import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Tactic

-- 设置有限类型作为索引
def UserIdx := Fin m
def ItemIdx := Fin n
def LatentIdx := Fin k

-- 用户隐因子矩阵 U: m × k
def UserFactors (m k : ℕ) := Matrix (Fin m) (Fin k) ℝ

-- 物品隐因子矩阵 V: n × k
def ItemFactors (n k : ℕ) := Matrix (Fin n) (Fin k) ℝ

-- 评分矩阵（稀疏，用部分函数或集合表示观测值）
structure RatingMatrix (m n : ℕ) where
  ratings : Set (Fin m × Fin n × ℝ) -- (i, j, r_ij)

-- 示例：构造一个 3×4 评分矩阵，k=2
def exampleRatings : RatingMatrix 3 4 := {
  ratings := { (⟨0, by decide⟩, ⟨0, by decide⟩, 5.0),
               (⟨0, by decide⟩, ⟨1, by decide⟩, 3.0),
               (⟨1, by decide⟩, ⟨2, by decide⟩, 4.0),
               (⟨2, by decide⟩, ⟨3, by decide⟩, 2.0) }
}


-- 向量点积（两个 Fin k → ℝ 的向量）
def dotProduct {k : ℕ} (u v : Fin k → ℝ) : ℝ :=
  (Finset.univ : Finset (Fin k)).sum (fun i => u i * v i)

-- 预测评分：⟨U_i, V_j⟩
def predict {m n k : ℕ} (U : UserFactors m k) (V : ItemFactors n k) (i : Fin m) (j : Fin n) : ℝ :=
  dotProduct (U i) (V j)

-- 示例：验证点积
#eval dotProduct (fun i => if i = ⟨0, by decide⟩ then 1.0 else 2.0)
                 (fun i => if i = ⟨0, by decide⟩ then 3.0 else 4.0)
-- 应为 1*3 + 2*4 = 11 （当 k=2）



-- Frobenius 范数平方（用于正则化）
def frobNormSq {m k : ℕ} (M : Matrix (Fin m) (Fin k) ℝ) : ℝ :=
  (Finset.univ : Finset (Fin m)).sum fun i =>
    (Finset.univ : Finset (Fin k)).sum fun j =>
      M i j * M i j

-- 损失函数
def lossFunction {m n k : ℕ} (λ : ℝ) (R : RatingMatrix m n) (U : UserFactors m k) (V : ItemFactors n k) : ℝ :=
  let observed := R.ratings
  let errorTerm := (observed.map (fun ⟨i, j, r⟩ => (predict U V i j - r)^2)).sum
  let regTerm := λ * (frobNormSq U + frobNormSq V)
  errorTerm / 2 + regTerm / 2

-- 示例调用（需先构造 U, V）
-- 后面我们将构造示例矩阵



-- 对 U_i 的梯度分量（向量 ∈ ℝ^k）
def gradU_i {m n k : ℕ} (λ : ℝ) (R : RatingMatrix m n) (U : UserFactors m k) (V : ItemFactors n k) (i : Fin m) : Fin k → ℝ :=
  fun l : Fin k =>
    let sumPart := (R.ratings.filter (fun ⟨i', j, r⟩ => i' = i)).sum fun ⟨_, j, r⟩ =>
      (predict U V i j - r) * V j l
    sumPart + λ * U i l

-- 对 V_j 的梯度分量
def gradV_j {m n k : ℕ} (λ : ℝ) (R : RatingMatrix m n) (U : UserFactors m k) (V : ItemFactors n k) (j : Fin j) : Fin k → ℝ :=
  fun l : Fin k =>
    let sumPart := (R.ratings.filter (fun ⟨i, j', r⟩ => j' = j)).sum fun ⟨i, _, r⟩ =>
      (predict U V i j - r) * U i l
    sumPart + λ * V j l

-- 对 U_i 的梯度分量（向量 ∈ ℝ^k）
def gradU_i {m n k : ℕ} (λ : ℝ) (R : RatingMatrix m n) (U : UserFactors m k) (V : ItemFactors n k) (i : Fin m) : Fin k → ℝ :=
  fun l : Fin k =>
    let sumPart := (R.ratings.filter (fun ⟨i', j, r⟩ => i' = i)).sum fun ⟨_, j, r⟩ =>
      (predict U V i j - r) * V j l
    sumPart + λ * U i l


-- 上面笔误
-- 对 V_j 的梯度分量
def gradV_j {m n k : ℕ} (λ : ℝ) (R : RatingMatrix m n) (U : UserFactors m k) (V : ItemFactors n k) (j : Fin j) : Fin k → ℝ :=
  fun l : Fin k =>
    let sumPart := (R.ratings.filter (fun ⟨i, j', r⟩ => j' = j)).sum fun ⟨i, _, r⟩ =>
      (predict U V i j - r) * U i l
    sumPart + λ * V j l


-- 学习率 η
def updateU {m n k : ℕ} (η λ : ℝ) (R : RatingMatrix m n) (U : UserFactors m k) (V : ItemFactors n k) : UserFactors m k :=
  fun i => fun l =>
    U i l - η * (gradU_i λ R U V i l)

def updateV {m n k : ℕ} (η λ : ℝ) (R : RatingMatrix m n) (U : UserFactors m k) (V : ItemFactors n k) : ItemFactors n k :=
  fun j => fun l =>
    V j l - η * (gradV_j λ R U V j l)

-- 一次迭代
def gdStep {m n k : ℕ} (η λ : ℝ) (R : RatingMatrix m n) (U : UserFactors m k) (V : ItemFactors n k) :
    UserFactors m k × ItemFactors n k :=
  let U' := updateU η λ R U V
  let V' := updateV η λ R U' V -- 顺序更新（也可并行）
  (U', V')

-- 初始化：随机或零初始化（这里用常量示例）
def initU (m k : ℕ) (initVal : ℝ) : UserFactors m k :=
  fun i => fun j => initVal

def initV (n k : ℕ) (initVal : ℝ) : ItemFactors n k :=
  fun i => fun j => initVal

-- 训练 N 步
def train {m n k : ℕ} (η λ : ℝ) (R : RatingMatrix m n) (steps : ℕ) : UserFactors m k × ItemFactors n k :=
  let rec loop (step : ℕ) (U : UserFactors m k) (V : ItemFactors n k) : UserFactors m k × ItemFactors n k :=
    if step = 0 then (U, V)
    else
      let (U', V') := gdStep η λ R U V
      loop (step - 1) U' V'
  loop steps (initU m k 0.1) (initV n k 0.1)


  -- 定义具体尺寸
def m := 3
def n := 4
def k := 2

-- 构造示例评分
def R_example : RatingMatrix m n := {
  ratings := {
    (⟨0, by decide⟩, ⟨0, by decide⟩, 5.0),
    (⟨0, by decide⟩, ⟨1, by decide⟩, 3.0),
    (⟨1, by decide⟩, ⟨2, by decide⟩, 4.0),
    (⟨2, by decide⟩, ⟨3, by decide⟩, 2.0)
  }
}

-- 运行训练
#eval let (U_final, V_final) := train 0.01 0.1 R_example 100
       predict U_final V_final ⟨0, by decide⟩ ⟨0, by decide⟩
-- 输出应接近 5.0（取决于训练步数和学习率）


-- 需要导入微积分库
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.FDeriv

-- 为简化，考虑固定 V 和 R，将 ℒ 视为 U 的函数
-- 验证 ∂ℒ/∂U_i[l] = gradU_i λ R U V i l

-- 这需要将矩阵视为向量空间中的点，使用 FDeriv 等工具
-- 此处仅示意，完整证明较复杂

theorem gradU_i_correct {m n k : ℕ} (λ : ℝ) (R : RatingMatrix m n) (V : ItemFactors n k) (i : Fin m) (l : Fin k) :
    ∀ (U : UserFactors m k),
    HasFDerivAt (fun U => lossFunction λ R U V) (gradU_i λ R U V i) U := by
  sorry -- 此处可展开为对每个分量的偏导数验证，使用 Summable 和线性性质
  

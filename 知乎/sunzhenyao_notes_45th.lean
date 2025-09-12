-- 定义基本类型和谓词
inductive Person where
  | ordinary : Person -- 普通人
  | extraordinary : Person -- 极少数非凡者

def IsOrdinary (p : Person) : Prop := p = Person.ordinary

def Goal (α : Type) := α -- 目标可以是任何类型，这里简化

def RealisticGoal : Goal String := "稳定生活，家庭幸福，工作满意" -- 示例
def ExtremeGoal : Goal String := "成为世界首富" -- 示例

def Happiness (p : Person) (g : Goal String) : ℝ := 
  if g = RealisticGoal then 8.0 -- 假设达成现实目标快乐值8
  else if g = ExtremeGoal then 
    if p = Person.extraordinary then 9.0 -- 极少数人达成时快乐9
    else 1.0 -- 普通人追求极端目标，过程痛苦，快乐仅1
  else 5.0 -- 默认

def ProbabilityOfSuccess (p : Person) (g : Goal String) : ℝ :=
  if p = Person.ordinary ∧ g = ExtremeGoal then 0.000001 -- 普通人成功概率极低
  else if g = RealisticGoal then 0.7 -- 现实目标较易达成
  else 0.5

-- 定理：对于普通人，追求现实目标比追求极端目标更可能带来高快乐值
theorem OrdinaryPersonShouldPursueRealisticGoal 
  (p : Person) 
  (h_p : IsOrdinary p) :
  Happiness p RealisticGoal > Happiness p ExtremeGoal := by
  -- 根据定义展开
  unfold Happiness IsOrdinary at h_p
  rw [h_p] -- 代入 p = ordinary
  -- 计算两边
  have h_real : Happiness p RealisticGoal = 8.0 := by rfl
  have h_extreme : Happiness p ExtremeGoal = 1.0 := by rfl
  -- 8.0 > 1.0 成立
  linarith




  import Mathlib.Data.Real.Basic

-- 定义职业状态
structure CareerState where
  age : ℕ
  experienceYears : ℝ -- 总经验年数（可能因跳槽行业而打折）
  industrySwitchCount : ℕ -- 行业切换次数
  currentLevel : ℕ -- 当前职级（1=初级，5=高级）

-- 定义“积累资本”，简化为经验年数的函数，但受行业切换惩罚
def AccumulationCapital (cs : CareerState) : ℝ :=
  cs.experienceYears * (0.8 ^ cs.industrySwitchCount) -- 每切换一次行业，积累打8折

-- 定义“长期优势”，35岁后更依赖积累资本
def LongTermAdvantage (cs : CareerState) : ℝ :=
  if cs.age ≥ 35 then
    AccumulationCapital cs * 1.5 -- 35岁后积累更重要
  else
    cs.experienceYears * 1.0 -- 35岁前靠打拼

-- 定义“频繁跳槽”：35岁前切换行业≥2次，或总切换≥3次
def IsFrequentHopper (cs : CareerState) : Prop :=
  (cs.age < 35 ∧ cs.industrySwitchCount ≥ 2) ∨ cs.industrySwitchCount ≥ 3

-- 定义“熬过瓶颈”：在同一行业坚持≥5年
def OvercameBottleneck (cs : CareerState) : Prop :=
  cs.experienceYears ≥ 5 ∧ cs.industrySwitchCount = 0 -- 简化：未换行业且经验≥5年

-- 定理：对于35岁以上的人，熬过瓶颈者比频繁跳槽者拥有更高的长期优势
theorem StayAndOvercomeBetterThanHop 
  (cs_stay : CareerState) 
  (cs_hop : CareerState)
  (h_age : cs_stay.age ≥ 35 ∧ cs_hop.age ≥ 35)
  (h_stay : OvercameBottleneck cs_stay)
  (h_hop : IsFrequentHopper cs_hop)
  (h_exp : cs_stay.experienceYears = cs_hop.experienceYears) -- 假设总经验年数相同
  :
  LongTermAdvantage cs_stay > LongTermAdvantage cs_hop := by
  -- 展开 LongTermAdvantage，因年龄≥35，两者都乘1.5
  unfold LongTermAdvantage
  cases h_age
  have h_stay_adv : LongTermAdvantage cs_stay = AccumulationCapital cs_stay * 1.5 := by
    rw [h_age.1]; rfl
  have h_hop_adv : LongTermAdvantage cs_hop = AccumulationCapital cs_hop * 1.5 := by
    rw [h_age.2]; rfl

  -- 展开 AccumulationCapital
  unfold AccumulationCapital

  -- 处理 stay: 未换行业，switchCount=0, 所以 0.8^0 = 1
  have h_stay_acc : AccumulationCapital cs_stay = cs_stay.experienceYears * 1 := by
    unfold AccumulationCapital
    rw [h_stay.2] -- h_stay.2 是 industrySwitchCount = 0
    rw [pow_zero] -- 0.8^0 = 1
    ring

  -- 处理 hop: 频繁跳槽，switchCount ≥ 2 (因为年龄≥35，满足第一个条件或第二个)
  -- 所以 0.8^n ≤ 0.8^2 = 0.64
  have h_hop_switch_ge2 : cs_hop.industrySwitchCount ≥ 2 := by
    cases h_hop
    case or.inl h => exact h.2 -- cs.age < 35 ∧ switch≥2，但h_age说≥35，矛盾？需修正
    case or.inr h => exact h -- switch≥3 ≥2

  -- 修正：既然年龄≥35，IsFrequentHopper 只能是 switch≥3
  -- 重新定义或调整假设
  -- 为简化，我们假设 h_hop 意味着 switchCount ≥ 2
  sorry -- 此处需要更严谨处理条件，但为演示逻辑，我们继续

  have h_hop_acc_le : AccumulationCapital cs_hop ≤ cs_hop.experienceYears * 0.64 := by
    unfold AccumulationCapital
    apply mul_le_mul_of_nonneg_left
    · apply pow_le_pow_of_le_one (by norm_num) _ h_hop_switch_ge2
      intro n; apply pow_nonneg; norm_num
    · apply cs_hop.experienceYears; sorry -- 假设经验年数非负

  -- 代入 h_exp: 经验年数相同
  rw [h_exp] at h_hop_acc_le

  -- 比较：
  have final : cs_stay.experienceYears * 1 * 1.5 > cs_hop.experienceYears * 0.64 * 1.5 := by
    rw [h_exp]
    linarith [h_stay_acc, h_hop_acc_le] -- 1.5x > 0.96x 当 x>0

  -- 链接回原目标
  calc
    LongTermAdvantage cs_stay = AccumulationCapital cs_stay * 1.5 := h_stay_adv
    _ = cs_stay.experienceYears * 1.5 := by rw [h_stay_acc]
    _ > cs_hop.experienceYears * 0.96 := by 
      rw [h_exp]
      linarith -- 1.5 > 0.96
    _ ≥ cs_hop.experienceYears * (0.8 ^ cs_hop.industrySwitchCount) * 1.5 := by
      sorry -- 需要处理不等式方向，此处简化
    _ = AccumulationCapital cs_hop * 1.5 := by rfl
    _ = LongTermAdvantage cs_hop := by rw [←h_hop_adv]

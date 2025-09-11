import Mathlib
import Lean

-- 时间点
abbrev Time := Nat

-- 系统在时间 t 的状态
abbrev SystemState := PaxosSystem  -- 使用之前定义的 PaxosSystem

-- 执行轨迹：从时间 0 开始的系统状态序列
def Execution := Time → SystemState

-- 初始状态
def initExecution (nodes : List NodeId) (quorumSize : Nat) : Execution :=
  fun t => if t = 0 then initSystem nodes quorumSize else default  -- 简化：后续状态需由 step 定义



-- 在时间 T 之后，所有“应投递”的消息都会在有限步内被投递
structure EventuallySynchronous where
  T : Time
  -- 在 T 之后，若消息在时间 t 发出，则必在 t + Δ 内被投递（Δ 固定）
  Δ : Nat
  deliveryGuarantee : ∀ (exec : Execution) (t : Time),
    t ≥ T →
    ∀ (msg : PaxosMsg) (from to : NodeId),
    -- 若 msg 在时间 t 被加入网络，则必在 t + Δ 前被投递
    (msg ∈ (exec t).network ∧ (from, to, msg) ∈ (exec t).network) →
    ∃ (t' : Time), t ≤ t' ≤ t + Δ ∧
      (exec t').step? (Step.Deliver from to msg) ≠ none



def isEventuallyChosen (exec : Execution) (quorum : Set NodeId) (v : Value) : Prop :=
  ∃ (T_final : Time), ∀ (t : Time), t ≥ T_final →
    ∃ (b : Ballot), isChosen (exec t) quorum b v

-- 或更强：存在某个 ballot b 和值 v 被 chosen
def liveness (exec : Execution) (quorum : Set NodeId) : Prop :=
  ∃ (b : Ballot) (v : Value), isEventuallyChosen exec quorum v



-- Proposer 带超时重试
structure LiveProposer where
  id : NodeId
  currentValue : Value
  currentBallot : Ballot
  timeout : Nat := 3  -- 超时步数
  lastActionTime : Time
  quorumSize : Nat

def LiveProposer.nextBallot (self : LiveProposer) : Ballot :=
  { round := self.currentBallot.round + 1, proposer := self.id }

def LiveProposer.onTimeout (self : LiveProposer) (currentTime : Time) : LiveProposer × Option PaxosMsg :=
  if currentTime - self.lastActionTime ≥ self.timeout then
    let newBallot := self.nextBallot
    ({ self with currentBallot := newBallot, lastActionTime := currentTime },
     some (PaxosMsg.Prepare newBallot))
  else
    (self, none)


theorem paxos_liveness
    (exec : Execution)
    (es : EventuallySynchronous)
    (quorum : Set NodeId)
    (propId : NodeId)
    -- 假设存在一个永不崩溃的 Proposer，持续重试
    (h_proposer_alive : ∀ (t : Time), (exec t).proposers propId ≠ default)
    (h_proposer_retries : ∀ (t : Time),
        let p := (exec t).proposers propId
        let (p', msg?) := p.onTimeout t
        -- 若超时，则在下一步发送 Prepare
        (msg? = none ∨ ∃ t' > t, (exec t').network.contains (propId, ·, msg?.get!))) :
    liveness exec quorum := by
  -- 证明思路：
  -- 1. 由 ♢Synchrony，存在 T_sync 之后消息在 Δ 步内送达
  -- 2. Proposer 持续重试，其 ballot 无限递增
  -- 3. 最终存在一个 ballot b，使得：
  --    a. b > 所有 acceptor 的 promisedBallot（因 ballot 递增）
  --    b. Prepare(b) 被多数派接收并回复 Promise（因 ♢Synchrony）
  --    c. Proposer 收到多数 Promise，发送 Accept(b, v)
  --    d. Accept(b, v) 被多数派接收并 Accepted（因 ♢Synchrony）
  --    e. v 被 chosen
  let T_sync := es.T
  let Δ := es.Δ

  -- 关键引理：Proposer 的 ballot 趋于无穷
  have h_ballot_unbounded : ∀ (B : Ballot), ∃ (t : Time), (exec t).proposers propId.currentBallot > B := by
    intro B
    -- 由于提议者持续超时重试，ballot 每 timeout 步至少 +1
    let t0 := T_sync + (B.round + 1) * (LiveProposer.timeout default)
    use t0
    -- 详细证明需归纳，此处省略
    sorry

  -- 选取足够大的 ballot b
  let b := { round := (exec T_sync).proposers propId.currentBallot.round + quorum.cardinal + 100, proposer := propId }
  have h_b_large : ∃ t1, (exec t1).proposers propId.currentBallot ≥ b := by
    apply h_ballot_unbounded b
    done

  -- 在 t1 发送 Prepare(b)
  -- 由 ♢Synchrony，所有消息在 Δ 步内送达
  -- 故在 t1 + Δ 前，多数派 acceptor 收到 Prepare(b) 并回复 Promise(b, ·)
  -- 在 t1 + 2Δ 前，proposer 收到多数 Promise，发送 Accept(b, v)
  -- 在 t1 + 3Δ 前，多数派 acceptor 收到 Accept(b, v) 并 Accepted
  -- 故在 t_final = t1 + 3Δ 时，v 被 chosen

  use b
  use (exec (T_sync + 1)).proposers propId.currentValue -- 任意值，实际由协议决定
  use T_sync + 3 * Δ + 100

  intro t h_t_large
  -- 此处需详细展开每一步的消息传递和状态变更
  -- 依赖 ♢Synchrony 的 deliveryGuarantee 和 proposer 重试行为
  sorry


  -- 假设 Leader 在 T_leader 后稳定
def leaderStable (exec : Execution) (leaderId : NodeId) (T_leader : Time) : Prop :=
  ∀ (t : Time), t ≥ T_leader → (exec t).multiProposer.leader = leaderId

-- 在 Leader 稳定 + ♢Synchrony 下，Multi-Paxos 活性更强（1 RTT 提案）
theorem multi_paxos_liveness
    (exec : Execution)
    (es : EventuallySynchronous)
    (quorum : Set NodeId)
    (leaderId : NodeId)
    (T_leader : Time)
    (h_leader_stable : leaderStable exec leaderId T_leader)
    (h_leader_alive : ∀ t, (exec t).proposers leaderId ≠ default) :
    liveness exec quorum := by
  -- 证明类似，但更简单：
  -- 1. Leader 稳定 → 无需 Prepare，直接 Accept
  -- 2. ♢Synchrony → Accept 在 Δ 步内送达多数派
  -- 3. 值被 chosen
  let T_sync := max es.T T_leader
  let Δ := es.Δ

  -- 选取任意实例
  let inst := 1
  let b := { round := inst, proposer := leaderId }
  let v := "some_value"

  -- 在 T_sync 发送 Accept(b, v)
  -- 在 T_sync + Δ 前，多数派接收并 Accepted
  use b
  use v
  use T_sync + Δ + 1

  intro t h_t_large
  -- 详细证明略
  sorry

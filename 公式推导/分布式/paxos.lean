import Mathlib
import Lean

-- 节点标识符（Proposer/Acceptor/Learner）
abbrev NodeId := Nat

-- 提议值（泛型，但为简化用 String）
abbrev Value := String

-- 提案编号（Ballot Number）—— (round, nodeId) 避免冲突
structure Ballot where
  round : Nat
  proposer : NodeId
deriving BEq, Repr, Inhabited

-- 比较 Ballot：先比 round，再比 nodeId（全序）
instance : Ord Ballot where
  compare b1 b2 :=
    match compare b1.round b2.round with
    | .eq => compare b1.proposer b2.proposer
    | o => o

-- 消息类型
inductive PaxosMsg where
  | Prepare (b : Ballot)                          -- 阶段1a
  | Promise (b : Ballot) (lastVoted : Option (Ballot × Value)) -- 阶段1b
  | Accept (b : Ballot) (v : Value)               -- 阶段2a
  | Accepted (b : Ballot) (v : Value)             -- 阶段2b
deriving Repr


structure AcceptorState where
  promisedBallot : Ballot := { round := 0, proposer := 0 } -- 初始为最小 ballot
  acceptedBallot : Option Ballot := none
  acceptedValue : Option Value := none
deriving Repr, Inhabited

-- Acceptor 处理 Prepare 消息
def AcceptorState.onPrepare (self : AcceptorState) (b : Ballot) :
    Option (PaxosMsg × AcceptorState) := -- 返回 Promise + 新状态
  if b > self.promisedBallot then
    some (PaxosMsg.Promise b (self.acceptedBallot.zip self.acceptedValue), {
      self with
        promisedBallot := b
    })
  else
    none -- 不回应或回应拒绝（Basic Paxos 中可忽略）

-- Acceptor 处理 Accept 消息
def AcceptorState.onAccept (self : AcceptorState) (b : Ballot) (v : Value) :
    Option (PaxosMsg × AcceptorState) :=
  if b ≥ self.promisedBallot then
    some (PaxosMsg.Accepted b v, {
      self with
        promisedBallot := b,
        acceptedBallot := some b,
        acceptedValue := some v
    })
  else
    none

structure ProposerState where
  currentBallot : Ballot
  proposedValue : Value
  promisesReceived : List (Ballot × Option (Ballot × Value)) -- (fromBallot, lastVote)
  acceptsReceived : Nat -- 计数即可，为简化
  quorumSize : Nat      -- 多数派大小
deriving Repr

def ProposerState.init (b : Ballot) (v : Value) (q : Nat) : ProposerState :=
  { currentBallot := b, proposedValue := v, promisesReceived := [], acceptsReceived := 0, quorumSize := q }

-- 处理 Promise 消息
def ProposerState.onPromise (self : ProposerState) (from : Ballot) (lastVote : Option (Ballot × Value)) :
    ProposerState × Option PaxosMsg := -- 可能触发 Accept
  let self' := { self with promisesReceived := (from, lastVote) :: self.promisesReceived }
  let hasQuorum := self'.promisesReceived.length ≥ self.quorumSize
  if hasQuorum then
    -- 选择 value：若有 lastVote，则选最大 ballot 对应的值；否则用 proposedValue
    let selectedValue := match self'.promisesReceived.foldl (fun acc => fun (_, mv) =>
      match mv, acc with
      | some (b', v'), some (b_max, _) => if b' > b_max then some (b', v') else acc
      | some bv, none => some bv
      | none, _ => acc
    ) none with
    | some (_, v_max) => v_max
    | none => self.proposedValue
    (self', some (PaxosMsg.Accept self.currentBallot selectedValue))
  else
    (self', none)

-- 处理 Accepted 消息
def ProposerState.onAccepted (self : ProposerState) : ProposerState × Bool := -- 是否达成 chosen
  let self' := { self with acceptsReceived := self.acceptsReceived + 1 }
  let isChosen := self'.acceptsReceived ≥ self.quorumSize
  (self', isChosen)


-- 全局状态：所有节点状态 + 消息队列（简化为集合）
structure PaxosSystem where
  acceptors : NodeId → AcceptorState
  proposers : NodeId → ProposerState
  network : List (NodeId × NodeId × PaxosMsg) -- (from, to, msg)
deriving Repr

-- 单步执行：任选一个消息处理
inductive Step where
  | Deliver (from to : NodeId) (msg : PaxosMsg)
deriving Repr

-- 执行一步
def PaxosSystem.step (self : PaxosSystem) (s : Step) : Option PaxosSystem :=
  match s with
  | Step.Deliver from to msg =>
    match msg with
    | PaxosMsg.Prepare b =>
      match (self.acceptors to).onPrepare b with
      | some (reply, newAcc) =>
        some { self with
          acceptors := Function.update self.acceptors to newAcc,
          network := (to, from, reply) :: self.network
        }
      | none => none -- 拒绝或忽略
    | PaxosMsg.Promise b lastVote =>
      match (self.proposers from).onPromise b lastVote with
      | (newProp, some acceptMsg) =>
        some { self with
          proposers := Function.update self.proposers from newProp,
          network := (from, to, acceptMsg) :: self.network -- 广播给 acceptors，简化：只发一个
        }
      | (newProp, none) =>
        some { self with proposers := Function.update self.proposers from newProp }
    | PaxosMsg.Accept b v =>
      match (self.acceptors to).onAccept b v with
      | some (reply, newAcc) =>
        some { self with
          acceptors := Function.update self.acceptors to newAcc,
          network := (to, from, reply) :: self.network
        }
      | none => none
    | PaxosMsg.Accepted b v =>
      match (self.proposers from).onAccepted with
      | (newProp, isChosen) =>
        some { self with proposers := Function.update self.proposers from newProp }
        -- isChosen 可触发 Learner 通知，此处省略


-- 若存在一个 ballot b 和值 v，使得至少 quorum 个 acceptor 接受了 (b, v)
def isChosen (sys : PaxosSystem) (quorum : Set NodeId) (b : Ballot) (v : Value) : Prop :=
  (quorum.filter (fun a => (sys.acceptors a).acceptedBallot = some b ∧
                            (sys.acceptors a).acceptedValue = some v)).cardinal ≥ quorum.cardinal / 2 + 1

-- 辅助：最大 chosen ballot（若有）
def maxChosenBallot (sys : PaxosSystem) (quorum : Set NodeId) : Option Ballot :=
  quorum.foldl (fun maxB a =>
    match (sys.acceptors a).acceptedBallot with
    | some b => if maxB.isNone ∨ b > maxB.get! then some b else maxB
    | none => maxB
  ) none

-- 不变式：若 b 是 chosen 且值为 v，则任何 >b 的 chosen 值必须也是 v
theorem paxos_safety (sys : PaxosSystem) (quorum : Set NodeId) (b : Ballot) (v : Value)
    (h_chosen : isChosen sys quorum b v)
    (b' : Ballot) (v' : Value) (h_b'_gt : b' > b)
    (h_chosen' : isChosen sys quorum b' v') :
    v' = v := by
  -- 关键引理：任何被 chosen 的 ballot b，其值 v 必须等于某个 acceptor 的 lastVote 中 ≥b 的最大值
  -- 由于 acceptor 只接受 ≥ promised 的值，且 proposer 从多数派中选取最大 lastVote，故 v' 必等于 v
  sorry -- 此处需详细展开 acceptor 和 proposer 的行为约束


-- 实例编号
abbrev InstanceId := Nat

-- Multi-Paxos Proposer：跟踪每个 instance 的状态
structure MultiProposer where
  leader : NodeId
  currentInstance : InstanceId
  instanceStates : InstanceId → ProposerState
  stableLeader : Bool := true -- 假设 leader 稳定，可跳过 Prepare


def MultiProposer.propose (self : MultiProposer) (inst : InstanceId) (v : Value) :
    List PaxosMsg :=
  if self.stableLeader ∧ inst > 0 then
    -- Fast path: assume highest ballot is still valid
    let b := { round := inst, proposer := self.leader } -- 简化：用 inst 作为 round
    [PaxosMsg.Accept b v]
  else
    -- Slow path: run full Prepare/Promise
    let b := { round := inst, proposer := self.leader }
    [PaxosMsg.Prepare b]



  inductive FastPaxosMsg where
  | FastAccept (b : Ballot) (v : Value) -- client 直接发送
  | Accept! (b : Ballot) (v : Value)    -- proposer 发送（冲突时）
  | ... -- 其他同 Paxos

-- Acceptor 需记录多个候选值（若来自不同 client）
structure FastAcceptorState where
  promisedBallot : Ballot
  accepted : List (Ballot × Value) -- 可能多个值，若 ballot 相同则冲突


def FastAcceptorState.onFastAccept (self : FastAcceptorState) (b : Ballot) (v : Value) :
    Option (FastPaxosMsg × FastAcceptorState) :=
  if b > self.promisedBallot then
    let newAccepted := (b, v) :: self.accepted.filter (fun (b', _) => b' ≠ b) -- 清除旧 ballot
    some (FastPaxosMsg.Accepted b v, { self with promisedBallot := b, accepted := newAccepted })
  else if b = self.promisedBallot then
    -- 检查是否冲突
    match self.accepted.find? (fun (b', v') => b' = b ∧ v' ≠ v) with
    | some _ => -- 冲突！触发 recovery
      some (FastPaxosMsg.ConflictDetected b, self) -- 简化：实际需通知 leader
    | none =>
      let newAccepted := (b, v) :: self.accepted.filter (fun (b', _) => b' ≠ b)
      some (FastPaxosMsg.Accepted b v, { self with accepted := newAccepted })
  else
    none


  -- 初始化系统
def initSystem (nodes : List NodeId) (quorumSize : Nat) : PaxosSystem :=
  let initAcc := fun (id : NodeId) => ⟨⟩ -- 默认初始状态
  let initProp := fun (id : NodeId) => ProposerState.init { round := 1, proposer := id } "value1" quorumSize
  { acceptors := initAcc, proposers := initProp, network := [] }

-- 模拟步骤
#eval let sys0 := initSystem [0, 1, 2] 2
       let sys1 := sys0.step (Step.Deliver 0 1 (PaxosMsg.Prepare { round := 1, proposer := 0 }))
       let sys2 := sys1.get!.step (Step.Deliver 1 0 (PaxosMsg.Promise { round := 1, proposer := 0 } none))
       let sys3 := sys2.get!.step (Step.Deliver 0 1 (PaxosMsg.Accept { round := 1, proposer := 0 } "value1"))
       sys3

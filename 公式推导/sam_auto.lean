import Std.Data.HashMap

structure SAMNode where
  len : Nat          -- 该状态对应的最长子串长度
  link : Nat         -- 后缀链接（指向“次长后缀”所在状态）
  next : Std.HashMap Char Nat  -- 转移边：字符 → 状态编号
  deriving Inhabited, BEq, Repr

structure SAM where
  nodes : Array SAMNode  -- 所有状态节点，按编号索引
  last : Nat             -- 当前最后一个状态编号
  sz : Nat               -- 下一个可用状态编号（= nodes.size）
  deriving Inhabited, Repr

def mkNode (len : Nat) (link : Nat) : SAMNode :=
  { len := len, link := link, next := Std.HashMap.empty }

def SAM.extend (s : SAM) (c : Char) : SAM := do
  let cur := s.sz
  let newNodes := s.nodes.push (mkNode (s.nodes[s.last].len + 1) 0)
  let mut p := s.last

  -- Step 1: 沿后缀链接向上，填充缺失的转移边
  let (newNodes, p) := Id.run do
    let mut nodes := newNodes
    let mut p := p
    while p != 0 && !nodes[p].next.contains c do
      nodes := nodes.set! p { (nodes[p]) with next := nodes[p].next.insert c cur }
      p := nodes[p].link
    return (nodes, p)

  -- Step 2: 处理后缀链接
  let newNodes := if p == 0 then
    newNodes.set! cur { (newNodes[cur]) with link := 0 }
  else
    let q := newNodes[p].next[c]!
    if newNodes[p].len + 1 == newNodes[q].len then
      newNodes.set! cur { (newNodes[cur]) with link := q }
    else
      let clone := newNodes.size
      let newNodes := newNodes.push {
        len := newNodes[p].len + 1,
        link := newNodes[q].link,
        next := newNodes[q].next
      }
      -- Step 2.1: 修复 p 到 q 的转移
      let (newNodes, _) := Id.run do
        let mut nodes := newNodes
        let mut p := p
        while p != 0 && nodes[p].next.contains c && nodes[p].next[c]! == q do
          nodes := nodes.set! p { (nodes[p]) with next := nodes[p].next.insert c clone }
          p := nodes[p].link
        return (nodes, ())
      -- Step 2.2: 设置 q 和 cur 的后缀链接
      let newNodes := newNodes.set! q { (newNodes[q]) with link := clone }
      let newNodes := newNodes.set! cur { (newNodes[cur]) with link := clone }
      newNodes

  { s with nodes := newNodes, last := cur, sz := s.sz + 1 }

def SAM.init : SAM :=
  { nodes := #[mkNode 0 0], last := 0, sz := 1 }

#eval let s0 := SAM.init
      let s1 := s0.extend 'a'
      let s2 := s1.extend 'b'
      let s3 := s2.extend 'a'
      let s4 := s3.extend 'b'
      s4.nodes.size  -- 应输出状态总数（通常 7~8 个）


structure State where
  len : Nat
  link : Nat
  next : List (Char × Nat)  -- 使用列表表示转移

inductive SAM where
  | empty : SAM
  | cons : State → SAM → SAM

def SAM.size : SAM → Nat
  | empty => 0
  | cons _ rest => 1 + rest.size

def SAM.lastState : SAM → Option State
  | empty => none
  | cons s _ => some s

def SAM.findState (sam : SAM) (idx : Nat) : Option State :=
  match sam, idx with
  | empty, _ => none
  | cons s rest, 0 => some s
  | cons _ rest, n+1 => rest.findState n

partial def SAM.extend (sam : SAM) (c : Char) : SAM :=
  match sam.lastState with
  | none => 
    -- 初始状态
    let newState : State := {len := 1, link := 0, next := []}
    SAM.cons newState sam
  | some last =>
    let curIdx := sam.size
    let newState : State := {len := last.len + 1, link := 0, next := []}
    let samWithNew := SAM.cons newState sam
    
    -- 沿着后缀链接回溯并添加转移
    let rec addTransitions (p : Option Nat) (sam' : SAM) : SAM × Option Nat =
      match p with
      | none => (sam', none)  -- 到达根节点
      | some pIdx =>
        match sam'.findState pIdx with
        | none => (sam', none)
        | some pState =>
          -- 检查是否已经有字符c的转移
          if pState.next.any (λ (char, _) => char = c) then
            (sam', some pIdx)
          else
            -- 添加转移
            let updatedPState : State := 
              {pState with next := (c, curIdx) :: pState.next}
            let updatedSam := sam'.updateState pIdx updatedPState
            addTransitions (some pState.link) updatedSam
    
    let (samWithTrans, pOption) := addTransitions (some last.link) samWithNew
    
    match pOption with
    | none => samWithTrans  -- 回溯到了根节点
    | some pIdx =>
      match samWithTrans.findState pIdx with
      | none => samWithTrans
      | some pState =>
        -- 找到字符c的转移
        match pState.next.find (λ (char, _) => char = c) with
        | none => samWithTrans  -- 不应该发生
        | some (_, qIdx) =>
          match samWithTrans.findState qIdx with
          | none => samWithTrans
          | some qState =>
            if pState.len + 1 = qState.len then
              -- 更新新状态的后缀链接
              let updatedNewState : State := {newState with link := qIdx}
              samWithTrans.updateState curIdx updatedNewState
            else
              -- 需要克隆状态
              let cloneIdx := samWithTrans.size
              let cloneState : State := 
                {len := pState.len + 1, link := qState.link, next := qState.next}
              
              let samWithClone := SAM.cons cloneState samWithTrans
              
              -- 更新转移
              let rec updateTransitions (p' : Option Nat) (sam'' : SAM) : SAM =
                match p' with
                | none => sam''
                | some pIdx' =>
                  match sam''.findState pIdx' with
                  | none => sam''
                  | some pState' =>
                    -- 检查转移是否指向q
                    match pState'.next.find (λ (char, idx) => char = c && idx = qIdx) with
                    | none => updateTransitions (some pState'.link) sam''
                    | some _ =>
                      -- 更新转移指向克隆状态
                      let updatedPState' : State := 
                        {pState' with next := pState'.next.map (λ (char, idx) => 
                          if char = c && idx = qIdx then (c, cloneIdx) else (char, idx))}
                      let updatedSam := sam''.updateState pIdx' updatedPState'
                      updateTransitions (some pState'.link) updatedSam
              
              let samUpdated := updateTransitions (some pIdx) samWithClone
              
              -- 更新q和新状态的后缀链接
              let updatedQState : State := {qState with link := cloneIdx}
              let updatedNewState : State := {newState with link := cloneIdx}
              
              samUpdated
                .updateState qIdx updatedQState
                .updateState curIdx updatedNewState

-- 辅助函数：更新SAM中的状态
def SAM.updateState (sam : SAM) (idx : Nat) (newState : State) : SAM :=
  match sam, idx with
  | empty, _ => empty
  | cons s rest, 0 => cons newState rest
  | cons s rest, n+1 => cons s (rest.updateState n newState)






  

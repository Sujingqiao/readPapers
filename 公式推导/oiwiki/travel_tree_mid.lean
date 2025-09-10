inductive State where
  | Fetch      -- 取栈顶帧，准备处理
  | VisitNode  -- 处理非空节点：展开左、中、右
  | VisitNil   -- 处理空节点：忽略
  | Emit       -- 输出值
  | Halt       -- 停止


inductive Frame (α : Type) where
  | visit : Tree α → Frame α
  | print : α → Frame α


structure Context (α : Type) where
  stack  : List (Frame α)
  result : List α
  state  : State

def step {α : Type} (ctx : Context α) : Option (Context α)


| state := Fetch, stack := [] =>
  some { ctx with state := Halt }

| state := Fetch, stack := .visit nil :: rest =>
  some { ctx with state := VisitNil, stack := rest }

| state := Fetch, stack := .visit (node l v r) :: rest =>
  some {
    ctx with
    state := VisitNode,
    stack := rest,  -- 临时弹出，后面压入新帧
    -- 注意：我们在这里不压栈，而是让 VisitNode 状态负责压栈
  }

| state := Fetch, stack := .print x :: rest =>
  some { ctx with state := Emit, stack := rest }



| state := VisitNode =>
  let node := match ctx.stack.reverse.head? with
    | some (.visit (node l v r)) => node
    | _ => panic! "VisitNode without node frame"
  let newFrames := [
    .visit node.left,
    .print node.value,
    .visit node.right
  ]
  some {
    ctx with
    state := Fetch,
    stack := newFrames ++ ctx.stack  -- 压入新帧
  }


  | state := VisitNil =>
  some { ctx with state := Fetch }


  | state := Emit =>
  let x := match ctx.stack.reverse.head? with
    | some (.print x) => x
    | _ => panic! "Emit without print frame"
  some {
    ctx with
    state := Fetch,
    stack := ctx.stack.dropLast 1,  -- 弹出 print 帧
    result := x :: ctx.result       -- 输出
  }


  | state := Halt => none  -- 执行结束


  def runStateMachine {α : Type} (initialStack : List (Frame α)) : List α :=
  let rec loop (ctx : Context α) : List α :=
    match step ctx with
    | none => ctx.result.reverse
    | some nextCtx => loop nextCtx

  let initialCtx := {
    stack := initialStack,
    result := [],
    state := Fetch
  }

  loop initialCtx


  def inorderStateMachine {α : Type} (t : Tree α) : List α :=
  runStateMachine [.visit t]

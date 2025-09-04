# 代码区                                          | 说明区（公式/语义）
# ----------------------------------------------|----------------------------------------
D = [[], [], []]                                 #| 𝒟 ← {𝒟₀, 𝒟₁, 𝒟₂}  
                                                 #| 全局任务池：ded, abd, ind
                                                  
for b in range(B):                               # | 循环 B 次提案
    p_seed = random.choice(D[0] + D[1])          # | p ∼ 𝒟_ded ∪ 𝒟_abd
    inputs, method = policy_propose("ind", p_seed)#| (iₙ, oₙ) ← π_propose(p)  
    if validate(inputs):                          #| 验证执行合法性
        D[2].append((p_seed[0], inputs, method))  #| 𝒟_ind ← 𝒟_ind ∪ {(p, trace, m)}

                                                  #|
x = task[0]                                       #| 输入：问题 p
y = policy_solve(x)                               #| 输出：y ← π_solve(p)
r = compute_reward(y, task)                       #| 奖励：r ← R(y, y*)

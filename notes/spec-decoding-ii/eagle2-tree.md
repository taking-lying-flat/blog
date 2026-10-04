<figure class="lake-figure eagle2-tree-combined" id="eagle2-dynamic-tree">
<img src="assets/images/1791111210610-dc39f27b-138b-4a95-9d3a-264b01d4c45c.png" alt="EAGLE-2 草稿树的扩展与重排序" width="617" height="593" decoding="async">
<img src="assets/images/1791111229758-277f9cf8-a23c-4215-8013-33b898a1c3bd.png" alt="EAGLE-2 重排序后 token 的展平与树形注意力掩码" width="573" height="595" decoding="async">
</figure>

**EAGLE-2 逐层选择高分节点继续扩展，再从整棵候选树中重排得到最终待验证草稿**

以下使用示意概率，每次扩展从各父节点的分布中取 top-2 候选，每轮选择当前层的两个节点继续扩展，最终保留八个节点（含根节点 `I`）。`c` 表示草稿模型的条件概率，`V` 为路径上各候选概率的乘积，用于近似衡量路径的接受概率；根节点已经确定，令 `V(I) = 1`。`S` 记录特征序列，`E` 记录向后错开一个 token 的嵌入序列，各分支仅使用自身及祖先节点的历史

<div class="eagle2-walkthrough-grid" id="eagle2-walkthrough">
<div>

```text 初始化与第 1–2 轮
目标模型：How can → f_can → LMHead
          → softmax → 采样 I
S = [f_How, f_can]，E = [e_can, e_I]
草稿计算：配对输入 → Concat → FC
          → DraftDecoder → 末位输出

第 1 轮：生成第一层候选
  配对：[(f_How, e_can), (f_can, e_I)]
  → f_hat_I → LMHead → softmax
    do：   c = 0.60，V = 1 × 0.60 = 0.60
    make： c = 0.20，V = 1 × 0.20 = 0.20
  选择 do、make 继续扩展
  S_do = S_make = S + [f_hat_I]
  E_do = E + [e_do]
  E_make = E + [e_make]

第 2 轮：并行扩展 do、make
  do 新增配对 (f_hat_I, e_do)
  → f_hat_do → LMHead → softmax
    it：   c = 0.80，V = 0.60 × 0.80 = 0.48
    this： c = 0.10，V = 0.60 × 0.10 = 0.06
  make 新增配对 (f_hat_I, e_make)
  → f_hat_make → LMHead → softmax
    a：    c = 0.70，V = 0.20 × 0.70 = 0.14
    some： c = 0.10，V = 0.20 × 0.10 = 0.02

  当前层排序：it(0.48) > a(0.14)
              > this(0.06) > some(0.02)
  选择 it、a 继续扩展
  this、some 留在候选树中，不继续扩展
  S_it = S_do + [f_hat_do]
  E_it = E_do + [e_it]
  S_a = S_make + [f_hat_make]
  E_a = E_make + [e_a]
```

</div>
<div>

```text 第 3 轮、全树重排与目标模型验证
第 3 轮：并行扩展 it、a
  it 新增配对 (f_hat_do, e_it)
  → f_hat_it → LMHead → softmax
    now：   c = 0.70，V = 0.48 × 0.70 = 0.336
    later： c = 0.10，V = 0.48 × 0.10 = 0.048
  a 新增配对 (f_hat_make, e_a)
  → f_hat_a → LMHead → softmax
    plan： c = 0.60，V = 0.14 × 0.60 = 0.084
    list： c = 0.20，V = 0.14 × 0.20 = 0.028
  本例到此停止扩展

全树重排：比较所有已生成节点的 V
  I(1.00) > do(0.60) > it(0.48) > now(0.336)
  > make(0.20) > a(0.14) > plan(0.084)
  > this(0.06) > later(0.048)
  > list(0.028) > some(0.02)

  保留前八个节点（含根节点 I）：
  I、do、it、now、make、a、plan、this
  this 未继续扩展，但分数高于 later、list
  因此仍进入最终待验证草稿

目标模型验证：按树结构展平
  [I, do, make, it, this, a, now, plan]
  相对位置：[0, 1, 1, 2, 2, 2, 3, 3]
  tree attention 限制各节点仅关注自身与祖先
  所有节点均可关注已验证前缀 How can
  一次前向计算以下分支的目标模型分布：
    How can I do it now
    How can I do this
    How can I make a plan

  随后按投机采样规则接受或纠正候选
  展平顺序仅用于计算，不作为一句话直接输出
```

</div>
</div>

<section class="rotation-identities" id="rotation-identities">

以下讨论量化前的等价关系，$`H`$ 为正交矩阵，$`HH^\top=I`$

<div class="rotation-case" id="rotation-rmsnorm">

**① RMSNorm：先融合逐通道缩放，再旋转权重**

去掉逐通道缩放后，$`\operatorname{RMSNorm}_0`$ 与正交旋转可交换：

```math
\operatorname{RMSNorm}_0(ZH)=\operatorname{RMSNorm}_0(Z)H.
```

未融合缩放时，缩放矩阵隔开两次旋转，一般不能抵消：

```math
(W_1H)\operatorname{diag}(\alpha)(H^\top W_2)=W_1\bigl[H\operatorname{diag}(\alpha)H^\top\bigr]W_2\neq W_1\operatorname{diag}(\alpha)W_2.
```

先将缩放融合进 $`W_2`$，再旋转权重，两次旋转相邻即可抵消：

```math
(W_1H)\Bigl[H^\top\underbrace{\operatorname{diag}(\alpha)W_2}_{\text{融合后的权重}}\Bigr]=W_1\underbrace{HH^\top}_{I}\operatorname{diag}(\alpha)W_2=W_1\operatorname{diag}(\alpha)W_2.
```

</div>

<div class="rotation-case" id="rotation-nonlinearity">

**② 两层权重之间有非线性激活函数**

一般有 $`\phi(ZH)\neq\phi(Z)H`$，两次旋转不能跨过非线性抵消：

```math
\phi\!\left(X(W_1H)\right)(H^\top W_2)\neq\phi(XW_1)W_2.
```

激活后在线旋转，逆变换融合进 $`W_2`$；门控 FFN 中，旋转放在门控相乘后、下投影前：

```math
\underbrace{\bigl[\phi(XW_1)H\bigr]}_{\text{在线旋转}}\underbrace{(H^\top W_2)}_{\text{融合进权重}}=\phi(XW_1)\underbrace{HH^\top}_{I}W_2=\phi(XW_1)W_2.
```

</div>

<div class="rotation-case" id="rotation-flatquant">

**③ FlatQuant：一般可逆矩阵**

$`P`$ 一般会改变范数，不能直接穿过 RMSNorm；抵消时用 $`P^{-1}`$，而非 $`P^\top`$：

```math
\operatorname{RMSNorm}_0\!\left(X(W_1P)\right)(P^{-1}W_2)\neq\operatorname{RMSNorm}_0(XW_1)W_2.
```

线性缩放 $`D=\operatorname{diag}(\alpha)`$ 可先融合进 $`W_2`$：

```math
\bigl[X(W_1P)\bigr]\Bigl[P^{-1}\underbrace{(DW_2)}_{\text{先融合缩放}}\Bigr]=XW_1\underbrace{PP^{-1}}_{I}DW_2=XW_1DW_2.
```

非线性之后在线应用 $`P`$，将 $`P^{-1}`$ 融合进后层权重：

```math
\underbrace{\bigl[\phi(XW_1)P\bigr]}_{\text{在线变换}}\underbrace{(P^{-1}W_2)}_{\text{融合进权重}}=\phi(XW_1)\underbrace{PP^{-1}}_{I}W_2=\phi(XW_1)W_2.
```

</div>

<div class="rotation-case" id="rotation-rope">

**④ Q/K 的 RoPE：在位置旋转之后配对**

$`q_i`$、$`k_j`$ 为行向量，$`R_i`$、$`R_j`$ 为 RoPE 旋转。提前将 $`H`$ 融合进 Q/K 权重，一般会改变点积：

```math
(q_iHR_i)(k_jHR_j)^\top=q_iHR_iR_j^\top H^\top k_j^\top\neq q_iR_iR_j^\top k_j^\top.
```

RoPE 后对 Q、K 在线应用同一个 $`H`$，即可在点积中抵消：

```math
(q_iR_iH)(k_jR_jH)^\top=q_iR_i\underbrace{HH^\top}_{I}R_j^\top k_j^\top=q_iR_iR_j^\top k_j^\top.
```

</div>

</section>

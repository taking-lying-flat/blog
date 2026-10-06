<section class="rotation-identities" id="rotation-identities">

以下讨论量化前的数学等价性，$`H`$ 为正交矩阵，$`HH^\top=I`$；不等式表示一般情形下不能保持原计算

<div class="rotation-case" id="rotation-rmsnorm">

**① RMSNorm：先融合逐通道缩放，再旋转权重**

记不含逐通道缩放的归一化为 $`\operatorname{RMSNorm}_0`$。正交旋转不改变向量范数，因此归一化本身可以与旋转交换

```math
\operatorname{RMSNorm}_0(ZH)=\operatorname{RMSNorm}_0(Z)H.
```

未融合缩放时，缩放矩阵夹在两次旋转之间，一般不能抵消

```math
\begin{aligned}
(W_1H)\operatorname{diag}(\alpha)(H^\top W_2)
&=W_1\bigl[H\operatorname{diag}(\alpha)H^\top\bigr]W_2\\
&\neq W_1\operatorname{diag}(\alpha)W_2.
\end{aligned}
```

先将缩放融合进 $`W_2`$，再旋转权重，就能保持原计算

```math
\begin{aligned}
(W_1H)\Bigl[H^\top\underbrace{\operatorname{diag}(\alpha)W_2}_{\text{融合后的权重}}\Bigr]
&=W_1\underbrace{HH^\top}_{I}\operatorname{diag}(\alpha)W_2\\
&=W_1\operatorname{diag}(\alpha)W_2.
\end{aligned}
```

区别在于：第一式中缩放隔开两次旋转；第二式中缩放已吸收到 $`W_2`$，两次旋转相邻，可以直接抵消

</div>

<div class="rotation-case" id="rotation-nonlinearity">

**② 两层权重之间有非线性激活函数**

原计算为 $`Y=\phi(XW_1)W_2`$。由于一般有 $`\phi(ZH)\neq\phi(Z)H`$，不能把两次旋转分别融合进前后权重，再跨过激活函数抵消

```math
\phi\!\left(X\underbrace{W_1H}_{\text{修改后的前层权重}}\right)
\underbrace{(H^\top W_2)}_{\text{修改后的后层权重}}
\neq\phi(XW_1)W_2.
```

正确做法是在激活函数之后在线执行变换，将逆变换离线融合进 $`W_2`$。对于门控 FFN，变换放在门控相乘之后、Down Projection 之前

```math
\begin{aligned}
\underbrace{\bigl[\phi(XW_1)H\bigr]}_{\text{激活之后在线变换}}
\underbrace{(H^\top W_2)}_{\text{离线融合进权重}}
&=\phi(XW_1)\underbrace{HH^\top}_{I}W_2\\
&=\phi(XW_1)W_2.
\end{aligned}
```

</div>

<div class="rotation-case" id="rotation-flatquant">

**③ FlatQuant：一般可逆矩阵不能直接穿过 RMSNorm 或非线性**

FlatQuant 使用一般可逆矩阵 $`P`$，抵消时需要 $`P^{-1}`$，不能将其直接替换为 $`P^\top`$。$`P`$ 一般会改变向量范数，即使去掉逐通道缩放，也不能像正交旋转一样穿过 RMSNorm

```math
\operatorname{RMSNorm}_0\!\left(X(W_1P)\right)(P^{-1}W_2)
\neq\operatorname{RMSNorm}_0(XW_1)W_2.
```

若中间只有线性缩放，记 $`D=\operatorname{diag}(\alpha)`$，先融合缩放仍可保持等价

```math
\begin{aligned}
\bigl[X(W_1P)\bigr]\Bigl[P^{-1}\underbrace{(DW_2)}_{\text{先融合缩放}}\Bigr]
&=XW_1\underbrace{PP^{-1}}_{I}DW_2\\
&=XW_1DW_2.
\end{aligned}
```

遇到非线性时，仍需将 $`P`$ 放在非线性之后在线执行，$`P^{-1}`$ 则离线融合进后层权重；位于 RMSNorm 之后的变换也采用同样的配对方式

```math
\begin{aligned}
\underbrace{\bigl[\phi(XW_1)P\bigr]}_{\text{在线变换}}
\underbrace{(P^{-1}W_2)}_{\text{离线融合进权重}}
&=\phi(XW_1)\underbrace{PP^{-1}}_{I}W_2\\
&=\phi(XW_1)W_2.
\end{aligned}
```

</div>

<div class="rotation-case" id="rotation-rope">

**④ Q/K 的 RoPE：在位置旋转之后配对**

记 $`q_i=x_iW_q`$、$`k_j=x_jW_k`$ 为行向量，RoPE 后分别为 $`q_iR_i`$、$`k_jR_j`$。$`H`$ 一般不与位置旋转交换；若将 $`H`$ 提前融合进 $`W_q`$、$`W_k`$，注意力点积一般会改变

```math
\begin{aligned}
(q_iHR_i)(k_jHR_j)^\top
&=q_iHR_iR_j^\top H^\top k_j^\top\\
&\neq q_iR_iR_j^\top k_j^\top.
\end{aligned}
```

在 RoPE 之后对 Q、K 在线应用同一个正交变换，两次旋转便能在点积中相邻抵消

```math
\begin{aligned}
\underbrace{(q_iR_iH)}_{\text{Q：RoPE 后在线旋转}}
\underbrace{(k_jR_jH)^\top}_{\text{K：RoPE 后在线旋转}}
&=q_iR_i\underbrace{HH^\top}_{I}R_j^\top k_j^\top\\
&=q_iR_iR_j^\top k_j^\top.
\end{aligned}
```

</div>

</section>

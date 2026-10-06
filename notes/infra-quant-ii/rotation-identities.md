<section class="rotation-identities" id="rotation-identities">

<div class="rotation-case" id="rotation-nonlinearity">

**两层权重之间有非线性激活函数：** $`\phi\!\left(X(W_1H)\right)(H^\top W_2)\neq\phi(XW_1)W_2`$

激活后在线旋转，逆变换融合进 $`W_2`$；门控 FFN 中，旋转放在门控相乘后、下投影前：

```math
\underbrace{\bigl[\phi(XW_1)H\bigr]}_{\text{在线旋转}}\underbrace{(H^\top W_2)}_{\text{融合进权重}}=\phi(XW_1)\underbrace{HH^\top}_{I}W_2=\phi(XW_1)W_2.
```

</div>

<div class="rotation-case" id="rotation-flatquant">

**FlatQuant：一般可逆矩阵**

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

</section>

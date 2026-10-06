<aside class="rotation-identities" id="rotation-identities">

```math
D=\operatorname{diag}(\alpha),\qquad HH^\top=I,\qquad P=P_1\otimes P_2,\qquad Z=XW_1.
```

```math
\begin{aligned}
(W_1H)D(H^\top W_2)
&=W_1\bigl[HDH^\top\bigr]W_2
\neq W_1DW_2,\\[5pt]
(W_1H)\Bigl[H^\top\underbrace{DW_2}_{\text{融合后的权重}}\Bigr]
&=W_1\underbrace{HH^\top}_{I}DW_2
=W_1DW_2.
\end{aligned}
```

```math
\begin{aligned}
\phi\!\left(X\underbrace{W_1H}_{\text{修改后的前层权重}}\right)
\underbrace{(H^\top W_2)}_{\text{修改后的后层权重}}
&\neq\phi(XW_1)W_2,\\[5pt]
\underbrace{\bigl[\phi(XW_1)H\bigr]}_{\text{激活之后在线变换}}
\underbrace{(H^\top W_2)}_{\text{离线融合进权重}}
&=\phi(XW_1)\underbrace{HH^\top}_{I}W_2
=\phi(XW_1)W_2.
\end{aligned}
```

```math
\begin{aligned}
\bigl[X(W_1P)\bigr]\Bigl[P^{-1}\underbrace{(DW_2)}_{\text{先融合缩放}}\Bigr]
&=XW_1\underbrace{PP^{-1}}_{I}DW_2=XW_1DW_2,\\[5pt]
\underbrace{\bigl[\phi(XW_1)P\bigr]}_{\text{在线变换}}
\underbrace{(P^{-1}W_2)}_{\text{离线融合进权重}}
&=\phi(XW_1)\underbrace{PP^{-1}}_{I}W_2
=\phi(XW_1)W_2,\\[5pt]
\operatorname{RMSNorm}_0\!\left(X(W_1P)\right)(P^{-1}W_2)
&\neq\operatorname{RMSNorm}_0(XW_1)W_2.
\end{aligned}
```

```math
\begin{aligned}
(q_iHR_i)(k_jHR_j)^\top
&=q_iHR_iR_j^\top H^\top k_j^\top
\neq q_iR_iR_j^\top k_j^\top,\\[5pt]
\underbrace{(q_iR_iH)}_{\text{Q：RoPE 后在线旋转}}
\underbrace{(k_jR_jH)^\top}_{\text{K：RoPE 后在线旋转}}
&=q_iR_i\underbrace{HH^\top}_{I}R_j^\top k_j^\top
=q_iR_iR_j^\top k_j^\top.
\end{aligned}
```

</aside>

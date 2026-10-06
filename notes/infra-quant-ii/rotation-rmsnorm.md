<div class="rotation-case" id="rotation-rmsnorm">

**RMSNorm**

未融合缩放时，缩放矩阵隔开两次旋转，一般不能抵消：

```math
(W_1H)\operatorname{diag}(\alpha)(H^\top W_2)=W_1\bigl[H\operatorname{diag}(\alpha)H^\top\bigr]W_2\neq W_1\operatorname{diag}(\alpha)W_2.
```

先将缩放融合进 $`W_2`$，再旋转权重，两次旋转相邻即可抵消：

```math
(W_1H)\Bigl[H^\top\underbrace{\operatorname{diag}(\alpha)W_2}_{\text{融合后的权重}}\Bigr]=W_1\underbrace{HH^\top}_{I}\operatorname{diag}(\alpha)W_2=W_1\operatorname{diag}(\alpha)W_2.
```

</div>

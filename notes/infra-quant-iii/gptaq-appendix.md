## 🧮 Appendix

<!-- Faithful translation of Appendix A.1–A.3, li25dn.pdf, pp. 13–15. Original equation groups (19)–(39) correspond to (A.1)–(A.21). No derivation steps are omitted. -->

### A.1 Optimal Framework

**本节给出最优框架的详细推导。为求解拉格朗日函数的局部极小值，令偏导数为零：**

```math
\begin{cases}
\dfrac{\partial L}{\partial\Delta\mathbf{w}}=2\Delta\mathbf{w}\mathbf{H}-2\mathbf{r}\mathbf{X}^{\top}+\lambda\mathbf{e}_q=0,\\[0.4em]
\dfrac{\partial L}{\partial\lambda}=\Delta\mathbf{w}\mathbf{e}_q^{\top}+w_q-\hat{w}_q=0
\end{cases}\tag{A.1}
```

- 求解第一个方程，得到

```math
\Delta\mathbf{w}\mathbf{H}=-\frac{\lambda}{2}\mathbf{e}_q+\mathbf{r}\mathbf{X}^{\top}\tag{A.2}
```

- 在两侧右乘 Hessian 逆矩阵，得到

```math
\begin{aligned}
\Delta\mathbf{w}&=-\frac{\lambda}{2}\mathbf{e}_q\mathbf{H}^{-1}+\mathbf{r}\mathbf{X}^{\top}\mathbf{H}^{-1}\\
&=-\frac{\lambda}{2}\mathbf{H}_{q,:}^{-1}+\mathbf{r}\mathbf{X}^{\top}\mathbf{H}^{-1}
\end{aligned}\tag{A.3}
```

- 将上式代入式（A.1）的第二个方程，得到

```math
-\frac{\lambda}{2}\mathbf{H}_{q,:}^{-1}\mathbf{e}_q^{\top}+\mathbf{r}\mathbf{X}^{\top}\mathbf{H}^{-1}\mathbf{e}_q^{\top}+w_q-\hat{w}_q=-\frac{\lambda}{2}\mathbf{H}_{qq}^{-1}+\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}+w_q-\hat{w}_q=0\tag{A.4}
```

- 由此求得 $`\lambda`$

```math
\lambda=2\left(\frac{w_q-\hat{w}_q}{\mathbf{H}_{qq}^{-1}}+\frac{\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}}{\mathbf{H}_{qq}^{-1}}\right)\tag{A.5}
```

- 将上式代入式（A.3）的第二行，可得 $`\Delta\mathbf{w}`$

```math
\begin{aligned}
\Delta\mathbf{w}&=-\frac{\lambda}{2}\mathbf{H}_{q,:}^{-1}+\mathbf{r}\mathbf{X}^{\top}\mathbf{H}^{-1}\\
&=-\frac{w_q-\hat{w}_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{H}_{q,:}^{-1}\right)-\frac{\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}\mathbf{H}_{q,:}^{-1}}{\mathbf{H}_{qq}^{-1}}+\mathbf{r}\mathbf{X}^{\top}\mathbf{H}^{-1}\\
&=-\frac{w_q-\hat{w}_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{H}_{q,:}^{-1}\right)+\mathbf{r}\mathbf{X}^{\top}\left(\mathbf{H}^{-1}-\frac{\mathbf{H}_{:,q}^{-1}\mathbf{H}_{q,:}^{-1}}{\mathbf{H}_{qq}^{-1}}\right)\\
&=-\frac{w_q-\hat{w}_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{H}_{q,:}^{-1}\right)+\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}
\end{aligned}\tag{A.6}
```

- 将最优 $`\Delta\mathbf{w}`$ 代入损失函数 $`\|\Delta\mathbf{w}\mathbf{X}-\mathbf{r}\|_F^2`$，得到 $`L_q`$

```math
L_q=\left\|\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}-\mathbf{r}+\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{H}_{q,:}^{-1}\right)\mathbf{X}\right\|_F^2\tag{A.7}
```

- 展开该函数，得到

```math
\begin{aligned}
L_q&=\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}+\mathbf{r}\mathbf{r}^{\top}-2\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}+\frac{(\hat{w}_q-w_q)^2}{(\mathbf{H}_{qq}^{-1})^2}\cdot\mathbf{H}_{q,:}^{-1}\mathbf{X}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}\\
&\quad+2\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}\right)-2\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}\right)\\
&=\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{H}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}+\mathbf{r}\mathbf{r}^{\top}-2\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}+\frac{(\hat{w}_q-w_q)^2}{(\mathbf{H}_{qq}^{-1})^2}\cdot\mathbf{H}_{q,:}^{-1}\mathbf{H}\mathbf{H}_{:,q}^{-1}\\
&\quad+2\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{H}\mathbf{H}_{:,q}^{-1}\right)-2\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}\right)
\end{aligned}\tag{A.8}
```

- 首先化简 $`\mathbf{H}_{-q}^{-1}\mathbf{H}`$

```math
\begin{aligned}
\mathbf{H}_{-q}^{-1}\mathbf{H}&=\left(\mathbf{H}^{-1}-\frac{\mathbf{H}_{:,q}^{-1}\mathbf{H}_{q,:}^{-1}}{\mathbf{H}_{qq}^{-1}}\right)\mathbf{H}\\
&=\mathbf{I}-\left(\frac{\mathbf{H}_{:,q}^{-1}\mathbf{H}_{q,:}^{-1}}{\mathbf{H}_{qq}^{-1}}\right)\mathbf{H}\\
&=\mathbf{I}-\left(\frac{\mathbf{H}^{-1}\mathbf{e}_q^{\top}\mathbf{e}_q\mathbf{H}^{-1}}{\mathbf{H}_{qq}^{-1}}\right)\mathbf{H}\\
&=\mathbf{I}-\frac{1}{\mathbf{H}_{qq}^{-1}}\mathbf{H}^{-1}\mathbf{e}_q^{\top}\mathbf{e}_q
\end{aligned}\tag{A.9}
```

- 将该结果代回式（A.8）的最后一个等式，得到

```math
\begin{aligned}
L_q&=\mathbf{r}\mathbf{X}^{\top}\left(\mathbf{I}-\frac{1}{\mathbf{H}_{qq}^{-1}}\mathbf{H}^{-1}\mathbf{e}_q^{\top}\mathbf{e}_q\right)\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}+\mathbf{r}\mathbf{r}^{\top}-2\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}+\frac{(\hat{w}_q-w_q)^2}{(\mathbf{H}_{qq}^{-1})^2}\cdot\mathbf{H}_{q,:}^{-1}\mathbf{H}\mathbf{H}_{:,q}^{-1}\\
&\quad+2\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{r}\mathbf{X}^{\top}\left(\mathbf{I}-\frac{1}{\mathbf{H}_{qq}^{-1}}\mathbf{H}^{-1}\mathbf{e}_q^{\top}\mathbf{e}_q\right)\mathbf{H}_{:,q}^{-1}\right)-2\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\cdot\left(\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}\right)\\
&=-\frac{1}{\mathbf{H}_{qq}^{-1}}\mathbf{r}\mathbf{X}^{\top}\mathbf{H}^{-1}\mathbf{e}_q^{\top}\mathbf{e}_q\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}+\mathbf{r}\mathbf{r}^{\top}-\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}+\frac{(\hat{w}_q-w_q)^2}{(\mathbf{H}_{qq}^{-1})^2}\cdot\mathbf{H}_{q,:}^{-1}\mathbf{H}\mathbf{H}_{:,q}^{-1}\\
&\quad-2\frac{\hat{w}_q-w_q}{(\mathbf{H}_{qq}^{-1})^2}\mathbf{r}\mathbf{X}^{\top}\mathbf{H}^{-1}\mathbf{e}_q^{\top}\mathbf{e}_q\mathbf{H}_{:,q}^{-1}
\end{aligned}\tag{A.10}
```

- 由于 $`\mathbf{H}_{-q}^{-1}`$ 的第 $`q`$ 行已被消去，$`\mathbf{e}_q\mathbf{H}_{-q}^{-1}=\mathbf{0}`$ 为全零向量，因此式（A.10）最后一个等式中的第一项为零。又因 $`\mathbf{e}_q\mathbf{H}_{:,q}^{-1}=\mathbf{H}_{qq}^{-1}`$，最后一项可改写为 $`-2\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}`$

- 进一步化简 $`\mathbf{H}_{q,:}^{-1}\mathbf{H}\mathbf{H}_{:,q}^{-1}`$

```math
\begin{aligned}
\mathbf{H}_{q,:}^{-1}\mathbf{H}\mathbf{H}_{:,q}^{-1}&=\mathbf{e}_q\mathbf{H}^{-1}\mathbf{H}\mathbf{H}^{-1}\mathbf{e}_q^{\top}\\
&=\mathbf{e}_q\mathbf{H}^{-1}\mathbf{e}_q^{\top}\\
&=\mathbf{H}_{qq}^{-1}
\end{aligned}\tag{A.11}
```

- 最终，$`L_q`$ 可化简为

```math
L_q=\frac{(\hat{w}_q-w_q)^2}{\mathbf{H}_{qq}^{-1}}+\mathbf{r}\mathbf{r}^{\top}-\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{-q}^{-1}\mathbf{X}\mathbf{r}^{\top}-2\frac{\hat{w}_q-w_q}{\mathbf{H}_{qq}^{-1}}\mathbf{r}\mathbf{X}^{\top}\mathbf{H}_{:,q}^{-1}\tag{A.12}
```

### A.2 Proof Of Lemma 4.1

**不失一般性，首先证明 $`\mathbf{H}_{-1}^{-1}=(\mathbf{X}_{-1}\mathbf{X}_{-1}^{\top})^{-1}=\mathbf{L}_{2:,2:}\mathbf{L}_{2:,2:}^{\top}`$ 的情形。对 $`\mathbf{H}^{-1}`$ 进行 Cholesky 分解，有 $`\mathbf{H}^{-1}=\mathbf{L}\mathbf{L}^{\top}`$，其中 $`\mathbf{L}`$ 为下三角矩阵**

- 因此，Cholesky 因子可改写为

```math
\mathbf{L}=\begin{bmatrix}L_{11}&0\\\mathbf{L}_{2:,1}&\mathbf{L}_{2:,2:}\end{bmatrix}\tag{A.13}
```

- 将上式代入 $`\mathbf{H}^{-1}=\mathbf{L}\mathbf{L}^{\top}`$，得到

```math
\mathbf{H}^{-1}=\begin{bmatrix}\mathbf{H}_{11}^{-1}&\mathbf{H}_{1,2:}^{-1}\\\mathbf{H}_{2:,1}^{-1}&\mathbf{H}_{2:,2:}^{-1}\end{bmatrix}=\begin{bmatrix}L_{11}&0\\\mathbf{L}_{2:,1}&\mathbf{L}_{2:,2:}\end{bmatrix}\begin{bmatrix}L_{11}&\mathbf{L}_{2:,1}^{\top}\\0&\mathbf{L}_{2:,2:}^{\top}\end{bmatrix}=\begin{bmatrix}L_{11}^{2}&L_{11}\mathbf{L}_{2:,1}^{\top}\\L_{11}\mathbf{L}_{2:,1}&\mathbf{L}_{2:,1}\mathbf{L}_{2:,1}^{\top}+\mathbf{L}_{2:,2:}\mathbf{L}_{2:,2:}^{\top}\end{bmatrix}\tag{A.14}
```

- 由此可构造如下线性方程组

```math
\begin{cases}
\mathbf{H}_{11}^{-1}=L_{11}^{2},\\
\mathbf{H}_{2:,1}^{-1}=L_{11}\mathbf{L}_{2:,1},\\
\mathbf{H}_{2:,2:}^{-1}=\mathbf{L}_{2:,1}\mathbf{L}_{2:,1}^{\top}+\mathbf{L}_{2:,2:}\mathbf{L}_{2:,2:}^{\top}
\end{cases}\tag{A.15}
```

- 求解该方程组，得到

```math
\begin{cases}
L_{11}=\sqrt{\mathbf{H}_{11}^{-1}},\\[0.4em]
\mathbf{L}_{2:,1}=\dfrac{1}{\sqrt{\mathbf{H}_{11}^{-1}}}\mathbf{H}_{2:,1}^{-1},\\[0.6em]
\mathbf{L}_{2:,2:}\mathbf{L}_{2:,2:}^{\top}=\mathbf{H}_{2:,2:}^{-1}-\dfrac{1}{\sqrt{\mathbf{H}_{11}^{-1}}}\dfrac{1}{\sqrt{\mathbf{H}_{11}^{-1}}}\mathbf{H}_{2:,1}^{-1}\mathbf{H}_{1,2:}^{-1}
\end{cases}\tag{A.16}
```

- 回顾 $`\mathbf{H}_{1:,1:}^{-1}-\frac{1}{\mathbf{H}_{11}^{-1}}\mathbf{H}_{1:,1}^{-1}\mathbf{H}_{1,1:}^{-1}`$，该式对第一行和第一列执行高斯消元，结果对应 $`\mathbf{H}_{-1}^{-1}`$。式（A.16）的第三个方程实质上对应移除第一行和第一列后的 $`\mathbf{H}_{-1}^{-1}`$，因此有 $`\mathbf{H}_{-1}^{-1}=\mathbf{L}_{2:,2:}\mathbf{L}_{2:,2:}^{\top}`$

- 递归移除当前 Hessian 逆矩阵的第一行和第一列，即得 $`\mathbf{H}_{-q:}^{-1}=\mathbf{L}_{q+1:,q+1:}\mathbf{L}_{q+1:,q+1:}^{\top}`$，引理得证

### A.3 Proof Of Theorem 4.2

**首先计算矩阵 $`\mathbf{P}`$ 的各行，其表达式为：**

```math
\mathbf{P}_{i,:}=\Delta\mathbf{X}_{i,:}\mathbf{X}^{\top}\mathbf{L}_{i+1:,i+1:}\mathbf{L}_{i+1:,i+1:}^{\top}\tag{A.17}
```

- 对于矩阵 $`\mathbf{L}_{i+1:,i+1:}^{\top}`$，仅当 $`j>i`$ 时，第 $`j`$ 列才含有非零元素，因此 $`\mathbf{P}`$ 的元素可写为

```math
\mathbf{P}_{i,j}=\begin{cases}
\displaystyle\sum_{a=i+1}^{j}\mathbf{O}_{i,a}\mathbf{L}_{a,j}^{\top},&i<j,\\[0.5em]
0,&i\geq j
\end{cases}\tag{A.18}
```

<!-- The PDF writes Delta X_{i:,} in the following definition; its original index order is retained. -->
- 其中，$`\mathbf{O}_{i,a}=(\Delta\mathbf{X}_{i:,}\mathbf{X}^{\top}\mathbf{L}_{i+1:,i+1:})_a`$ 是前三项乘积中的元素。矩阵 $`\mathbf{L}_{i+1:,i+1:}`$ 仅在 $`a>i`$ 时于第 $`a`$ 列含有非零元素，因此 $`\mathbf{O}`$ 可计算为

```math
\mathbf{O}_{i,a}=\begin{cases}
\displaystyle\sum_{b=a}^{n}(\Delta\mathbf{X}_{i,:}\mathbf{X}^{\top})_b\mathbf{L}_{b,a}=\sum_{b=a}^{n}(\Delta\mathbf{X}\mathbf{X}^{\top})_{i,b}\mathbf{L}_{b,a},&i<a,\\[0.5em]
0,&i\geq a
\end{cases}\tag{A.19}
```

- 因此，将 $`\Delta\mathbf{X}\mathbf{X}^{\top}\mathbf{L}`$ 的下三角区域掩去，即可得到 $`\mathbf{O}`$

```math
\mathbf{O}=(\Delta\mathbf{X}\mathbf{X}^{\top}\mathbf{L})\odot\mathbf{M}_U\tag{A.20}
```

- 由于 $`\mathbf{O}`$ 的下三角区域为零，可直接将 $`\mathbf{O}`$ 与 $`\mathbf{L}^{\top}`$ 相乘得到 $`\mathbf{P}`$

```math
(\mathbf{O}\mathbf{L}^{\top})_{i,j}=\sum_{a=1}^{j}\mathbf{O}_{i,a}\mathbf{L}_{a,j}^{\top}=\sum_{a=1}^{i}0\times\mathbf{L}_{a,j}^{\top}+\sum_{a=i+1}^{j}\mathbf{O}_{i,a}\mathbf{L}_{a,j}^{\top}+\sum_{a=j+1}^{n}\mathbf{O}_{i,a}\times0=\sum_{a=i+1}^{j}\mathbf{O}_{i,a}\mathbf{L}_{a,j}^{\top}=\mathbf{P}_{i,j}\tag{A.21}
```

- 因此，$`\mathbf{P}=\left((\Delta\mathbf{X}\mathbf{X}^{\top}\mathbf{L})\odot\mathbf{M}_U\right)\mathbf{L}^{\top}`$，定理得证

<figure class="paper-algorithm" aria-labelledby="gptaq-algorithm-1">
<figcaption id="gptaq-algorithm-1"><strong>Algorithm 1</strong> GPTAQ Quantization For One Layer</figcaption>

```math
\begin{array}{r l}
&\textbf{Input:}\ \text{FP weight }\mathbf{W},\ \text{calibration input }\mathbf{X},\ \text{FP input }\tilde{\mathbf{X}},\ \text{block size }B\\[-0.25em]
1:&\mathbf{H}\gets\mathbf{X}\mathbf{X}^{\top},\quad\Delta\mathbf{X}\mathbf{X}^{\top}\gets(\tilde{\mathbf{X}}-\mathbf{X})\mathbf{X}^{\top}\\[-0.25em]
2:&\mathbf{L}=\operatorname{Inverse\_Cholesky}(\mathbf{H}+\lambda_1\mathbf{I})\\[-0.25em]
3:&\mathbf{P}\gets\left((\Delta\mathbf{X}\mathbf{X}^{\top}\mathbf{L})\odot\mathbf{M}_U\right)\mathbf{L}^{\top}\\[-0.25em]
4:&\mathbf{Q}\gets\mathbf{0}_{m\times n},\ \mathbf{E}\gets\mathbf{0}_{m\times B}\\[-0.25em]
5:&\textbf{for }i=0,B,2B,\ldots\textbf{ do}\\[-0.25em]
6:&\quad\textbf{for }j=i,i+1,\ldots,i+B-1\textbf{ do}\\[-0.25em]
7:&\qquad\mathbf{Q}_{:,j}\gets\operatorname{quant}(\mathbf{W}_{:,j})\\[-0.25em]
8:&\qquad\mathbf{E}_{:,j-i}\gets(\mathbf{W}_{:,j}-\mathbf{Q}_{:,j})/L_{jj}\\[-0.25em]
9:&\qquad\mathbf{W}_{:,j:(i+B)}\gets\mathbf{W}_{:,j:(i+B)}-\mathbf{E}_{:,j-i}\mathbf{L}_{j,j:(i+B)}^{\top}+\mathbf{W}_{:,j}\mathbf{P}_{j,j:(i+B)}\\[-0.25em]
10:&\quad\textbf{end for}\\[-0.25em]
11:&\quad\mathbf{W}_{:,(i+B):}\gets\mathbf{W}_{:,(i+B):}-\mathbf{E}\cdot\mathbf{L}_{i:(i+B),(i+B):}^{\top}+\mathbf{W}_{:,i:(i+B)}\mathbf{P}_{i:(i+B),(i+B):}\\[-0.25em]
12:&\textbf{end for}
\end{array}
```

</figure>

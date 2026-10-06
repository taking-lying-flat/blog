<!-- Source: https://arxiv.org/html/2210.17323v2#alg1 . Restore the two residual subtractions omitted in the archived screenshot; use the upper Cholesky factor as in Algorithm 1. -->
<figure class="paper-algorithm" aria-labelledby="gptq-algorithm">
<figcaption id="gptq-algorithm"><strong>Algorithm 1</strong> GPTQ</figcaption>

```math
\begin{array}{l l}
\textbf{Require: }\mathbf W,\quad\mathbf H^{-1}=(2\mathbf X\mathbf X^\top+\lambda\mathbf I)^{-1},\quad\text{block size }B.\\[4pt]
\mathbf Q\leftarrow\mathbf0_{d_{\mathrm{row}}\times d_{\mathrm{col}}}
 &\triangleright\text{ Quantized output}\\[3pt]
\mathbf E\leftarrow\mathbf0_{d_{\mathrm{row}}\times B}
 &\triangleright\text{ Block quantization errors}\\[3pt]
\mathbf H^{-1}\leftarrow\operatorname{Cholesky}(\mathbf H^{-1})^\top
 &\triangleright\text{ Hessian inverse information}\\[3pt]
\textbf{for }i=0,B,2B,\ldots\ \textbf{do}\\[3pt]
\quad\textbf{for }j=i,\ldots,i+B-1\ \textbf{do}\\[3pt]
\qquad\mathbf Q_{:,j}\leftarrow\operatorname{quant}(\mathbf W_{:,j})
 &\triangleright\text{ Quantize column}\\[4pt]
\qquad\mathbf E_{:,j-i}\leftarrow(\mathbf W_{:,j}-\mathbf Q_{:,j})/[\mathbf H^{-1}]_{jj}
 &\triangleright\text{ Quantization error}\\[4pt]
\qquad\mathbf W_{:,j:(i+B)}\leftarrow\mathbf W_{:,j:(i+B)}-\mathbf E_{:,j-i}\mathbf H^{-1}_{j,j:(i+B)}
 &\triangleright\text{ Update weights in block}\\[3pt]
\quad\textbf{end for}\\[3pt]
\quad\mathbf W_{:,(i+B):}\leftarrow\mathbf W_{:,(i+B):}-\mathbf E\mathbf H^{-1}_{i:(i+B),(i+B):}
 &\triangleright\text{ Update remaining weights}\\[3pt]
\textbf{end for}
\end{array}
```

</figure>

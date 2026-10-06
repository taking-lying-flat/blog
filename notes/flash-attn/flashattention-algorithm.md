<figure class="paper-algorithm" aria-labelledby="flashattention-algorithm">
<figcaption id="flashattention-algorithm"><strong>Algorithm 1</strong> FlashAttention</figcaption>

```math
\begin{array}{r l}
&\textbf{Require: }\mathbf Q,\mathbf K,\mathbf V\in\mathbb R^{N\times d}\text{ in HBM; on-chip SRAM of size }M.\\[4pt]
1:&\text{Set }B_c=\left\lceil\frac{M}{4d}\right\rceil,\quad B_r=\min\!\left(\left\lceil\frac{M}{4d}\right\rceil,d\right).\\[5pt]
2:&\mathbf O\leftarrow\mathbf0_{N\times d},\quad\ell\leftarrow\mathbf0_N,\quad m\leftarrow(-\infty)_N\quad\text{in HBM}.\\[3pt]
3:&T_r\leftarrow\left\lceil N/B_r\right\rceil,\quad T_c\leftarrow\left\lceil N/B_c\right\rceil.\\[3pt]
&\text{Split }\mathbf Q\text{ into }\mathbf Q_1,\ldots,\mathbf Q_{T_r}\in\mathbb R^{B_r\times d};\quad
  \mathbf K,\mathbf V\text{ into }\mathbf K_j,\mathbf V_j\in\mathbb R^{B_c\times d}.\\[3pt]
4:&\text{Split }\mathbf O\text{ into }\mathbf O_i\in\mathbb R^{B_r\times d},\text{ and }\ell,m\text{ into }\ell_i,m_i\in\mathbb R^{B_r}.\\[3pt]
5:&\textbf{for }j=1,\ldots,T_c\ \textbf{do}\\[3pt]
6:&\quad\text{Load }\mathbf K_j,\mathbf V_j\text{ from HBM to on-chip SRAM}.\\[3pt]
7:&\quad\textbf{for }i=1,\ldots,T_r\ \textbf{do}\\[3pt]
8:&\qquad\text{Load }\mathbf Q_i,\mathbf O_i,\ell_i,m_i\text{ from HBM to on-chip SRAM}.\\[3pt]
9:&\qquad\mathbf S_{ij}\leftarrow\mathbf Q_i\mathbf K_j^\top\in\mathbb R^{B_r\times B_c}\quad\text{on chip}.\\[3pt]
10:&\qquad\widetilde m_{ij}\leftarrow\operatorname{rowmax}(\mathbf S_{ij}),\quad
 \widetilde{\mathbf P}_{ij}\leftarrow\exp(\mathbf S_{ij}-\widetilde m_{ij}),\quad
 \widetilde\ell_{ij}\leftarrow\operatorname{rowsum}(\widetilde{\mathbf P}_{ij}).\\[4pt]
11:&\qquad m_i^{\mathrm{new}}\leftarrow\max(m_i,\widetilde m_{ij}),\quad
 \ell_i^{\mathrm{new}}\leftarrow e^{m_i-m_i^{\mathrm{new}}}\ell_i+
 e^{\widetilde m_{ij}-m_i^{\mathrm{new}}}\widetilde\ell_{ij}.\\[4pt]
12:&\qquad\mathbf O_i\leftarrow\operatorname{diag}(\ell_i^{\mathrm{new}})^{-1}
 \left[\operatorname{diag}(\ell_i)e^{m_i-m_i^{\mathrm{new}}}\mathbf O_i+
 e^{\widetilde m_{ij}-m_i^{\mathrm{new}}}\widetilde{\mathbf P}_{ij}\mathbf V_j\right];\quad\text{write to HBM}.\\[4pt]
13:&\qquad\ell_i\leftarrow\ell_i^{\mathrm{new}},\quad m_i\leftarrow m_i^{\mathrm{new}};\quad\text{write to HBM}.\\[3pt]
14:&\quad\textbf{end for}\\[3pt]
15:&\textbf{end for}\\[3pt]
16:&\textbf{return }\mathbf O.
\end{array}
```

</figure>

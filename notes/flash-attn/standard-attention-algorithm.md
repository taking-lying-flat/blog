<figure class="paper-algorithm" aria-labelledby="standard-attention-algorithm">
<figcaption id="standard-attention-algorithm"><strong>Algorithm 0</strong> Standard Attention Implementation</figcaption>

```math
\begin{array}{r l}
&\textbf{Require: }\mathbf Q,\mathbf K,\mathbf V\in\mathbb R^{N\times d}\text{ in HBM}.\\[4pt]
1:&\text{Load }\mathbf Q,\mathbf K\text{ by blocks from HBM; compute }\mathbf S=\mathbf Q\mathbf K^\top;\text{ write }\mathbf S\text{ to HBM}.\\[3pt]
2:&\text{Read }\mathbf S\text{ from HBM; compute }\mathbf P=\operatorname{softmax}(\mathbf S);\text{ write }\mathbf P\text{ to HBM}.\\[3pt]
3:&\text{Load }\mathbf P,\mathbf V\text{ by blocks from HBM; compute }\mathbf O=\mathbf P\mathbf V;\text{ write }\mathbf O\text{ to HBM}.\\[3pt]
4:&\textbf{return }\mathbf O.
\end{array}
```

</figure>

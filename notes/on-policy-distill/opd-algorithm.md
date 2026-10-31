<!-- Transcribed from the supplied GKD algorithm image. The notation and update
order follow the source; line numbers are presentation only. -->
<figure class="paper-algorithm" aria-labelledby="gkd-algorithm-1">
<figcaption id="gkd-algorithm-1"><strong>Algorithm 1</strong> Generalized Knowledge Distillation (GKD)</figcaption>

```math
\begin{array}{r l}
1:&\textbf{Given:}\ \text{teacher model }p_T,\ \text{student model }p_S^\theta,\ \text{dataset }(X,Y)\text{ containing input-output pairs}\\[-0.25em]
2:&\textbf{Hyperparameters:}\ \text{student data fraction }\lambda\in[0,1],\ \text{divergence }D,\ \text{learning rate }\eta\\[-0.25em]
3:&\textbf{for each step }k=1,\ldots,K\textbf{ do}\\[-0.25em]
4:&\quad\text{Generate }u\sim\operatorname{Uniform}(0,1)\\[-0.25em]
5:&\quad\textbf{if }u\leq\lambda\textbf{ then}\\[-0.25em]
6:&\qquad\text{Sample }x\text{ from }X\text{ and generate }y\sim p_S^\theta(\cdot\mid x)\text{ to obtain }B=\{(x_b,y_b)\}_{b=1}^{B}\\[-0.25em]
7:&\quad\textbf{else}\\[-0.25em]
8:&\qquad\text{Sample }B=\{(x_b,y_b)\}_{b=1}^{B}\text{ from }(X,Y)\\[-0.25em]
9:&\quad\textbf{end if}\\[-0.25em]
10:&\quad\theta\gets\theta-\eta\frac{1}{B}\sum_{(x,y)\in B}\nabla_\theta D\!\left(p_T\middle\|p_S^\theta\right)(y\mid x)\\[-0.25em]
11:&\textbf{end for}
\end{array}
```

</figure>

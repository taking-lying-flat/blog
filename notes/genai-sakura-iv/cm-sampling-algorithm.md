<figure class="cm-algorithm" aria-labelledby="cm-algorithm-1">
<figcaption id="cm-algorithm-1"><strong>Algorithm 1</strong> Multistep Consistency Sampling</figcaption>

```math
\begin{array}{l}
\textbf{Input: }\text{Consistency model }f_\theta(\cdot,\cdot),\text{ sequence of time points }\tau_1>\tau_2>\cdots>\tau_{N-1},\text{ initial noise }\hat{\mathbf x}_T\\[-0.2em]
\mathbf x\leftarrow f_\theta(\hat{\mathbf x}_T,T)\\[-0.2em]
\textbf{for }n=1\textbf{ to }N-1\textbf{ do}\\[-0.2em]
\quad\text{Sample }\mathbf z\sim\mathcal N(\mathbf0,\mathbf I)\\[-0.2em]
\quad\hat{\mathbf x}_{\tau_n}\leftarrow\mathbf x+\sqrt{\tau_n^2-\epsilon^2}\,\mathbf z\\[-0.2em]
\quad\mathbf x\leftarrow f_\theta(\hat{\mathbf x}_{\tau_n},\tau_n)\\[-0.2em]
\textbf{end for}\\[-0.2em]
\textbf{Output: }\mathbf x
\end{array}
```

</figure>

<figure class="paper-algorithm" aria-labelledby="pc-ve-algorithm">
<figcaption id="pc-ve-algorithm"><strong>Algorithm 2</strong> PC Sampling (VE SDE)</figcaption>

```math
\begin{array}{r l l}
1:&\mathbf x_N\sim\mathcal N(\mathbf0,\sigma_{\max}^2\mathbf I)\\[3pt]
2:&\textbf{for }i=N-1,\ldots,0\ \textbf{do}\\[3pt]
3:&\quad\mathbf x'_i\leftarrow\mathbf x_{i+1}+(\sigma_{i+1}^2-\sigma_i^2)s_{\theta^*}(\mathbf x_{i+1},\sigma_{i+1})
 &\triangleright\text{ Predictor}\\[3pt]
4:&\quad\mathbf z\sim\mathcal N(\mathbf0,\mathbf I)\\[3pt]
5:&\quad\mathbf x_i\leftarrow\mathbf x'_i+\sqrt{\sigma_{i+1}^2-\sigma_i^2}\,\mathbf z\\[3pt]
6:&\quad\textbf{for }j=1,\ldots,M\ \textbf{do}
 &\triangleright\text{ Corrector}\\[3pt]
7:&\qquad\mathbf z\sim\mathcal N(\mathbf0,\mathbf I)\\[3pt]
8:&\qquad\mathbf x_i\leftarrow\mathbf x_i+\epsilon_i s_{\theta^*}(\mathbf x_i,\sigma_i)+\sqrt{2\epsilon_i}\,\mathbf z\\[3pt]
&\quad\textbf{end for}\\[3pt]
&\textbf{end for}\\[3pt]
9:&\textbf{return }\mathbf x_0.
\end{array}
```

</figure>

<figure class="paper-algorithm" aria-labelledby="pc-vp-algorithm">
<figcaption id="pc-vp-algorithm"><strong>Algorithm 3</strong> PC Sampling (VP SDE)</figcaption>

```math
\begin{array}{r l l}
1:&\mathbf x_N\sim\mathcal N(\mathbf0,\mathbf I)\\[3pt]
2:&\textbf{for }i=N-1,\ldots,0\ \textbf{do}\\[3pt]
3:&\quad\mathbf x'_i\leftarrow\left(2-\sqrt{1-\beta_{i+1}}\right)\mathbf x_{i+1}+\beta_{i+1}s_{\theta^*}(\mathbf x_{i+1},i+1)
 &\triangleright\text{ Predictor}\\[3pt]
4:&\quad\mathbf z\sim\mathcal N(\mathbf0,\mathbf I)\\[3pt]
5:&\quad\mathbf x_i\leftarrow\mathbf x'_i+\sqrt{\beta_{i+1}}\,\mathbf z\\[3pt]
6:&\quad\textbf{for }j=1,\ldots,M\ \textbf{do}
 &\triangleright\text{ Corrector}\\[3pt]
7:&\qquad\mathbf z\sim\mathcal N(\mathbf0,\mathbf I)\\[3pt]
8:&\qquad\mathbf x_i\leftarrow\mathbf x_i+\epsilon_i s_{\theta^*}(\mathbf x_i,i)+\sqrt{2\epsilon_i}\,\mathbf z\\[3pt]
&\quad\textbf{end for}\\[3pt]
&\textbf{end for}\\[3pt]
9:&\textbf{return }\mathbf x_0.
\end{array}
```

</figure>

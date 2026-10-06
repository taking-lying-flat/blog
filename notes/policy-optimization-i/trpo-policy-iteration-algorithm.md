<figure class="policy-algorithm" aria-labelledby="trpo-policy-iteration-algorithm">
<figcaption id="trpo-policy-iteration-algorithm"><strong>Algorithm 1</strong> Approximate Policy Iteration Algorithm Guaranteeing Non-Increasing Expected Cost η</figcaption>

```math
\begin{array}{l}
\text{Initialize }\pi_0.\\[2pt]
\textbf{for }i=0,1,2,\ldots\ \text{until convergence }\textbf{do}\\[2pt]
\quad\text{Compute all advantage values }A_{\pi_i}(s,a).\\[2pt]
\quad\text{Solve the following optimization problem:}\\[6pt]
\displaystyle\qquad\pi_{i+1}=\underset{\pi}{\arg\min}\left[L_{\pi_i}(\pi)+\frac{2\epsilon\gamma}{(1-\gamma)^2}D_{\mathrm{KL}}^{\max}(\pi_i,\pi)\right]\\[6pt]
\displaystyle\qquad\text{where }\epsilon=\max_s\max_a\left|A_{\pi_i}(s,a)\right|\\[6pt]
\displaystyle\qquad\text{and }L_{\pi_i}(\pi)=\eta(\pi_i)+\sum_s\rho_{\pi_i}(s)\sum_a\pi(a\mid s)A_{\pi_i}(s,a)\\[6pt]
\textbf{end for}
\end{array}
```

</figure>

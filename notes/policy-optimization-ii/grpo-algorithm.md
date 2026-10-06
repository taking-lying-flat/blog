<figure class="policy-algorithm" aria-labelledby="grpo-algorithm">
<figcaption id="grpo-algorithm"><strong>Algorithm 1</strong> Iterative Group Relative Policy Optimization</figcaption>

**Input:** initial policy $`\pi_{\theta_{\mathrm{init}}}`$, reward model $`r_\varphi`$, task prompts $`\mathcal D`$, hyperparameters $`\varepsilon,\beta,\mu`$

```math
\begin{array}{r l}
1:&\pi_\theta\gets\pi_{\theta_{\mathrm{init}}}\\[2pt]
2:&\textbf{for }\text{iteration}=1,\ldots,I\ \textbf{do}\\[2pt]
3:&\quad\pi_{\mathrm{ref}}\gets\pi_\theta\\[2pt]
4:&\quad\textbf{for }\text{step}=1,\ldots,M\ \textbf{do}\\[2pt]
5:&\qquad\text{Sample a batch }\mathcal D_b\text{ from }\mathcal D\\[2pt]
6:&\qquad\pi_{\theta_{\mathrm{old}}}\gets\pi_\theta\\[2pt]
7:&\qquad\text{Sample }G\text{ outputs }\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)\text{ for each }q\in\mathcal D_b\\[2pt]
8:&\qquad\text{Compute rewards }\{r_i\}_{i=1}^{G}\text{ for the sampled outputs using }r_\varphi\\[2pt]
9:&\qquad\text{Compute }\hat A_{i,t}\text{ for each token of }o_i\text{ through group relative advantage estimation}\\[2pt]
10:&\qquad\textbf{for }\text{GRPO iteration}=1,\ldots,\mu\ \textbf{do}\\[2pt]
11:&\qquad\quad\text{Update }\pi_\theta\text{ by maximizing the GRPO objective (3)}\\[2pt]
12:&\quad\text{Update }r_\varphi\text{ through continuous training using a replay mechanism}\\[2pt]
&\textbf{Output: }\pi_\theta
\end{array}
```

</figure>

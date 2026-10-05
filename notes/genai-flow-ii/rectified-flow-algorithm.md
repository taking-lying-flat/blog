<figure class="rf-algorithm" aria-labelledby="rf-algorithm-1">
<figcaption id="rf-algorithm-1"><strong>Algorithm 1</strong> Rectified Flow: Main Algorithm</figcaption>

```math
\begin{array}{l}
\textbf{Procedure: }\boldsymbol Z=\operatorname{RectFlow}((X_0,X_1)):\\[-0.2em]
\quad\textit{Inputs: }\text{Velocity model }v_\theta:\mathbb R^d\to\mathbb R^d\text{ with parameter }\theta.\\[-0.2em]
\quad\textit{Training: }\displaystyle\hat\theta=\arg\min_\theta\mathbb E\!\left[\left\|X_1-X_0-v(tX_1+(1-t)X_0,t)\right\|^2\right],\text{ with }t\sim\operatorname{Uniform}([0,1]).\\[-0.2em]
\quad\textit{Sampling: }\text{Draw }(Z_0,Z_1)\text{ following }\mathrm dZ_t=v_{\hat\theta}(Z_t,t)\,\mathrm dt\text{ starting from }Z_0\sim\pi_0\text{ (or }Z_1\sim\pi_1\text{)}.\\[-0.2em]
\quad\textit{Return: }\boldsymbol Z=\{Z_t:t\in[0,1]\}.
\end{array}
```

**Reflow** (optional): $`\boldsymbol Z^{k+1}=\operatorname{RectFlow}((Z_0^k,Z_1^k))`$, starting from $`(Z_0^0,Z_1^0)=(X_0,X_1)`$, where $`(X_0,X_1)`$ is drawn from $`\pi_0`$ and $`\pi_1`$

**Distill** (optional): Learn a neural network $`\hat T`$ to distill the $`k`$-rectified flow, such that $`Z_1^k\approx\hat T(Z_0^k)`$

</figure>

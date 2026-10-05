<figure class="flow-algorithm" aria-labelledby="ffjord-algorithm-1">
<figcaption id="ffjord-algorithm-1"><strong>Algorithm 1</strong> Unbiased Stochastic Log-Density Estimation Using The FFJORD Model</figcaption>

```math
\begin{array}{l}
\textbf{Require: }\text{dynamics }f_\theta,\text{ start time }t_0,\text{ stop time }t_1,\text{ data samples }\mathbf{x},\text{ data dimension }D\\[-0.2em]
\begin{array}{l l}
\quad\boldsymbol{\epsilon}\leftarrow\operatorname{sample\_unit\_variance}(\mathbf{x}.\mathrm{shape})
&\triangleright\text{ Sample }\boldsymbol{\epsilon}\text{ outside the integral}\\[-0.2em]
\quad\textbf{function }f_{\mathrm{aug}}([\mathbf{z}_t,\log p_t],t):
&\triangleright\text{ Augment }f\text{ with log-density dynamics}\\[-0.2em]
\qquad f_t\leftarrow f_\theta(\mathbf{z}(t),t)
&\triangleright\text{ Evaluate neural network}\\[-0.2em]
\qquad g\leftarrow\left.\boldsymbol{\epsilon}^\top\frac{\partial f}{\partial\mathbf{z}}\right|_{\mathbf{z}(t)}
&\triangleright\text{ Compute vector-Jacobian product with automatic differentiation}\\[-0.2em]
\qquad\widetilde{\mathrm{Tr}}=g\boldsymbol{\epsilon}
&\triangleright\text{ Unbiased estimate of }\mathrm{Tr}\!\left(\frac{\partial f}{\partial\mathbf{z}}\right)\text{ with }\boldsymbol{\epsilon}^\top\frac{\partial f}{\partial\mathbf{z}}\boldsymbol{\epsilon}\\[-0.2em]
\qquad\textbf{return }[f_t,-\widetilde{\mathrm{Tr}}]
&\triangleright\text{ Concatenate dynamics of state and log-density}\\[-0.2em]
\quad\textbf{end function}\\[-0.2em]
\quad[\mathbf{z}_0,\Delta_{\log p}]\leftarrow\operatorname{odeint}(f_{\mathrm{aug}},[\mathbf{x},\vec{0}],t_0,t_1)
&\triangleright\text{ Solve the ODE }\int_{t_0}^{t_1} f_{\mathrm{aug}}([\mathbf{z}(t),\log p(\mathbf{z}(t))],t)\,dt\\[-0.2em]
\quad\log\hat{p}(\mathbf{x})\leftarrow\log p_{\mathbf{z}_0}(\mathbf{z}_0)-\Delta_{\log p}
&\triangleright\text{ Add change in log-density}\\[-0.2em]
\textbf{return }\log\hat{p}(\mathbf{x})
\end{array}
\end{array}
```

</figure>

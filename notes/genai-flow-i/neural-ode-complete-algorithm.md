<figure class="flow-algorithm" aria-labelledby="neural-ode-algorithm-2">
<figcaption id="neural-ode-algorithm-2"><strong>Algorithm 2</strong> Complete Reverse-Mode Derivative Of An ODE Initial Value Problem</figcaption>

```math
\begin{array}{l}
\textbf{Input: }\text{dynamics parameters }\theta,\text{ start time }t_0,\text{ stop time }t_1,\text{ final state }\mathbf{z}(t_1),\text{ loss gradient }\frac{\partial L}{\partial\mathbf{z}(t_1)}\\[-0.2em]
\begin{array}{l l}
\quad\frac{\partial L}{\partial t_1}=\frac{\partial L}{\partial\mathbf{z}(t_1)}^\top f(\mathbf{z}(t_1),t_1,\theta)
&\triangleright\text{ Compute gradient w.r.t. }t_1\\[-0.2em]
\quad s_0=\left[\mathbf{z}(t_1),\frac{\partial L}{\partial\mathbf{z}(t_1)},\mathbf{0}_{|\theta|},-\frac{\partial L}{\partial t_1}\right]
&\triangleright\text{ Define initial augmented state}\\[-0.2em]
\quad\textbf{def }\operatorname{aug\_dynamics}([\mathbf{z}(t),\mathbf{a}(t),\cdot,\cdot],t,\theta):
&\triangleright\text{ Define dynamics on augmented state}\\[-0.2em]
\qquad\textbf{return }\left[f(\mathbf{z}(t),t,\theta),-\mathbf{a}(t)^\top\frac{\partial f}{\partial\mathbf{z}},-\mathbf{a}(t)^\top\frac{\partial f}{\partial\theta},-\mathbf{a}(t)^\top\frac{\partial f}{\partial t}\right]
&\triangleright\text{ Compute vector-Jacobian products}\\[-0.2em]
\quad\left[\mathbf{z}(t_0),\frac{\partial L}{\partial\mathbf{z}(t_0)},\frac{\partial L}{\partial\theta},\frac{\partial L}{\partial t_0}\right]=\operatorname{ODESolve}(s_0,\operatorname{aug\_dynamics},t_1,t_0,\theta)
&\triangleright\text{ Solve reverse-time ODE}\\[-0.2em]
\textbf{return }\frac{\partial L}{\partial\mathbf{z}(t_0)},\frac{\partial L}{\partial\theta},\frac{\partial L}{\partial t_0},\frac{\partial L}{\partial t_1}
&\triangleright\text{ Return all gradients}
\end{array}
\end{array}
```

</figure>

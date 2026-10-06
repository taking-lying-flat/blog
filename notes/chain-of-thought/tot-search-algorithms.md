<!-- Source: archived ToT algorithm screenshot and https://arxiv.org/html/2305.10601v2 . Normalize z/z_t and v_th/v_thres; evaluate the candidate s' and make terminal backtracking explicit, as described in Section 3. -->
<figure class="paper-algorithm" aria-labelledby="tot-bfs-algorithm">
<figcaption id="tot-bfs-algorithm"><strong>Algorithm 1</strong> ToT-BFS</figcaption>

```math
\begin{array}{l}
\textbf{Require: }\text{input }x,\text{ LM }p_\theta,\text{ thought generator }G,\text{ generation limit }k,\\[2pt]
\qquad\text{state evaluator }V,\text{ step limit }T,\text{ breadth limit }b.\\[4pt]
S_0\leftarrow\{x\}\\[3pt]
\textbf{for }t=1,\ldots,T\ \textbf{do}\\[3pt]
\quad S'_t\leftarrow\{[s,z]\mid s\in S_{t-1},\ z\in G(p_\theta,s,k)\}\\[3pt]
\quad V_t\leftarrow V(p_\theta,S'_t)\\[3pt]
\quad S_t\leftarrow\underset{S\subseteq S'_t,\,|S|=b}{\arg\max}\ \sum_{s\in S}V_t(s)\\[5pt]
\textbf{end for}\\[3pt]
\textbf{return }G\!\left(p_\theta,\underset{s\in S_T}{\arg\max}\ V_T(s),1\right).
\end{array}
```

</figure>

<figure class="paper-algorithm" aria-labelledby="tot-dfs-algorithm">
<figcaption id="tot-dfs-algorithm"><strong>Algorithm 2</strong> ToT-DFS</figcaption>

```math
\begin{array}{l}
\textbf{Require: }\text{current state }s,\text{ step }t,\text{ LM }p_\theta,\text{ thought generator }G,\text{ generation limit }k,\\[2pt]
\qquad\text{state evaluator }V,\text{ step limit }T,\text{ threshold }v_{\mathrm{th}}.\\[4pt]
\textbf{if }t>T\ \textbf{then}\\[3pt]
\quad\text{Record output }G(p_\theta,s,1);\quad\textbf{return}.\\[3pt]
\textbf{end if}\\[3pt]
\textbf{for }s'\in G(p_\theta,s,k)\ \textbf{do}\quad\triangleright\text{ Sorted candidates}\\[3pt]
\quad\textbf{if }V(p_\theta,\{s'\})(s')>v_{\mathrm{th}}\ \textbf{then}\quad\triangleright\text{ Prune other candidates}\\[3pt]
\qquad\operatorname{DFS}(s',t+1)\\[3pt]
\quad\textbf{end if}\\[3pt]
\textbf{end for}
\end{array}
```

</figure>

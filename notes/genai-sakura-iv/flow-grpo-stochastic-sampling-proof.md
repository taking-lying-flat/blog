## 🧮 Mathematical Derivations for Stochastic Sampling

**在漂移中加入与 score 相关的修正项，可以构造与 Flow-ODE 具有相同边缘分布的 SDE，并由其逆时间过程得到随机采样规则**

- 固定文本条件 $`c`$，以下省略该条件。记 $`p_t(x)`$ 为 ODE $`\mathrm dx_t=v_t(x_t)\,\mathrm dt`$ 的边缘密度，扩散系数 $`\sigma_t`$ 仅依赖时间；在密度为正、系数足够光滑且相应方程解唯一的条件下，比较具有相同初始分布的两个过程

**Fokker–Planck 方程将随机动力学转化为边缘密度的演化方程**

- 考虑从数据向噪声演化的正向 SDE

```math
\mathrm dx_t=f_{\mathrm{SDE}}(x_t,t)\,\mathrm dt+\sigma_t\,\mathrm dw_t\tag{14}
```

- 其 Fokker–Planck 方程与原 ODE 的连续性方程分别为

```math
\begin{aligned}
\partial_t p_t(x)&=-\nabla\!\cdot\!\left(f_{\mathrm{SDE}}(x,t)p_t(x)\right)+\frac12\nabla^2\!\left(\sigma_t^2p_t(x)\right)\\
\partial_t p_t(x)&=-\nabla\!\cdot\!\left(v_t(x)p_t(x)\right)
\end{aligned}\tag{15}
```

- 因为 $`\sigma_t`$ 与空间位置无关，且 $`\nabla p_t=p_t\nabla\log p_t`$，扩散项可以改写为

```math
\nabla^2\!\left(\sigma_t^2p_t(x)\right)=\sigma_t^2\nabla\!\cdot\!\left(p_t(x)\nabla\log p_t(x)\right)\tag{16}
```

- 取下述漂移，代入式（15）后，score 对应的概率通量与扩散项恰好抵消

```math
f_{\mathrm{SDE}}(x,t)=v_t(x)+\frac{\sigma_t^2}{2}\nabla\log p_t(x)\tag{17}
```

- 因此，在初始分布相同且密度演化方程解唯一时，下述正向 SDE 与原 ODE 在各时刻具有相同的边缘分布

```math
\mathrm dx_t=\left[v_t(x_t)+\frac{\sigma_t^2}{2}\nabla\log p_t(x_t)\right]\mathrm dt+\sigma_t\,\mathrm dw_t\tag{18}
```

**逆时间公式将上述正向 SDE 转换为从噪声到数据的生成过程**

- 对漂移为 $`f_{\mathrm{SDE}}`$、扩散系数为 $`\sigma_t`$ 的正向过程，逆时间漂移需减去 $`\sigma_t^2\nabla\log p_t`$。令时间从 1 向 0 递减，$`\mathrm d\bar w_t`$ 表示逆时间维纳增量，代入式（17）可得

```math
\begin{aligned}
\mathrm dx_t
&=\left[f_{\mathrm{SDE}}(x_t,t)-\sigma_t^2\nabla\log p_t(x_t)\right]\mathrm dt+\sigma_t\,\mathrm d\bar w_t\\
&=\left[v_t(x_t)-\frac{\sigma_t^2}{2}\nabla\log p_t(x_t)\right]\mathrm dt+\sigma_t\,\mathrm d\bar w_t
\end{aligned}\tag{19}
```

- 这就得到正文式（10）；相同边缘分布不要求两种过程具有相同的样本轨迹

**高斯插值路径将不可直接计算的 score 表示为流匹配速度场的函数**

- 设 $`x_0\sim\mathcal X_0`$ 与 $`x_1\sim\mathcal N(0,I)`$ 相互独立，$`x_t=\alpha_t x_0+\beta_t x_1`$，$`\dot\alpha_t`$、$`\dot\beta_t`$ 表示时间导数。这里 $`\beta_t`$ 为插值系数，与 GRPO 目标中的 KL 权重 $`\beta`$ 区分

- 给定 $`x_0`$ 后，条件密度及其 score 为

```math
\begin{aligned}
p_{t\mid0}(x_t\mid x_0)&=\mathcal N(x_t\mid\alpha_t x_0,\beta_t^2I)\\
\nabla_{x_t}\log p_{t\mid0}(x_t\mid x_0)&=-\frac{x_t-\alpha_t x_0}{\beta_t^2}=-\frac{x_1}{\beta_t}
\end{aligned}\tag{20}
```

- 对条件 score 按后验 $`p(x_0\mid x_t)`$ 求期望，即得边缘 score；这里不能直接用先验均值 $`\mathbb E[x_1]=0`$ 替代条件期望

```math
\nabla\log p_t(x_t)=\mathbb E\!\left[\nabla\log p_{t\mid0}(x_t\mid x_0)\mid x_t\right]=-\frac1{\beta_t}\mathbb E[x_1\mid x_t]\tag{21}
```

- 流匹配的理想速度场是路径导数的条件期望。利用 $`x_0=(x_t-\beta_t x_1)/\alpha_t`$，在 $`\alpha_t,\beta_t>0`$ 的内部时间区间有

```math
\begin{aligned}
v_t(x)
&=\mathbb E[\dot\alpha_t x_0+\dot\beta_t x_1\mid x_t=x]\\
&=\frac{\dot\alpha_t}{\alpha_t}x+\left(\dot\beta_t-\frac{\dot\alpha_t\beta_t}{\alpha_t}\right)\mathbb E[x_1\mid x_t=x]\\
&=\frac{\dot\alpha_t}{\alpha_t}x-\left(\dot\beta_t\beta_t-\frac{\dot\alpha_t\beta_t^2}{\alpha_t}\right)\nabla\log p_t(x)
\end{aligned}\tag{22}
```

- 代入 Rectified Flow 的 $`\alpha_t=1-t`$、$`\beta_t=t`$，整理得到

```math
v_t(x)=-\frac{x}{1-t}-\frac{t}{1-t}\nabla\log p_t(x)
\quad\Longrightarrow\quad
\nabla\log p_t(x)=-\frac{x}{t}-\frac{1-t}{t}v_t(x)\tag{23}
```

**将 score 代回逆时间 SDE，即可仅使用速度网络构造随机转移**

- 将式（23）代入式（19），得到正文式（11）所用的生成动力学

```math
\mathrm dx_t=\left[v_t(x_t)+\frac{\sigma_t^2}{2t}\left(x_t+(1-t)v_t(x_t)\right)\right]\mathrm dt+\sigma_t\,\mathrm d\bar w_t\tag{24}
```

- 采用速度网络 $`v_\theta`$ 和 Euler–Maruyama 离散化，令 $`\Delta t<0`$、$`\epsilon\sim\mathcal N(0,I)`$。逆时间维纳增量的协方差为 $`|\Delta t|I`$，因此漂移使用有符号步长，噪声使用 $`\sqrt{|\Delta t|}`$

```math
x_{t+\Delta t}=x_t+\left[v_\theta(x_t,t)+\frac{\sigma_t^2}{2t}\left(x_t+(1-t)v_\theta(x_t,t)\right)\right]\Delta t+\sigma_t\sqrt{|\Delta t|}\,\epsilon\tag{25}
```

- 给定当前状态后，式（25）为各向同性高斯转移，其均值由漂移更新确定，协方差为 $`\sigma_t^2|\Delta t|I`$，因而可直接计算 GRPO 所需的转移密度比以及正文式（13）的 KL 散度

- 上述边缘分布等价性针对精确速度场与连续时间过程；实际采样包含网络近似和时间离散化误差，$`t=0`$ 与 $`t=1`$ 的端点按正文所述调度处理

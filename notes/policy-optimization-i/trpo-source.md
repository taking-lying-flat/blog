### 源码：TRPO 的梯度与 Hessian 向量积

OpenAI Spinning Up 的官方实现使用 TensorFlow 1。下面沿着“代理损失 → 矩阵向量积 → 共轭梯度 → 回溯线搜索”读关键计算。代码中的 `adv_ph` 是**奖励优势**，所以先加负号构造需要最小化的 `pi_loss`；这与本节的代价最小化方向一致。

```python 代理损失与阻尼 HVP | openai/spinningup · 038665d
ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
pi_loss = -tf.reduce_mean(ratio * adv_ph)
pi_params = core.get_vars('pi')
gradient = core.flat_grad(pi_loss, pi_params)
v_ph, hvp = core.hessian_vector_product(d_kl, pi_params)
if damping_coeff > 0:
    hvp += damping_coeff * v_ph
```

[对应源码](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/trpo/trpo.py#L236-L248)。

`gradient` 是展平后的损失梯度。`hvp` 表示 $`Hv`$，加入 `damping_coeff * v_ph` 后变为 $`(H+\xi I)v`$，用于改善线性方程的数值条件。HVP 的内部只需要两次求导：

```python flat_grad / hessian_vector_product | openai/spinningup · 038665d
def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def flat_grad(f, params):
    return flat_concat(tf.gradients(xs=params, ys=f))

def hessian_vector_product(f, params):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = tf.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g*x), params)
```

[对应源码](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/trpo/core.py#L70-L80)。

先得到 $`g=\nabla_\theta f`$，再对标量 $`g^T v`$ 求导，就得到 $`Hv`$。这里 `v` 是固定向量，`f` 是平均 KL；整个过程不显式构造 Hessian 矩阵。

### 源码：共轭梯度只调用矩阵向量积

```python cg(Ax, b) · 省略函数说明 | openai/spinningup · 038665d
def cg(Ax, b):
    x = np.zeros_like(b)
    r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r,r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r,r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x
```

[对应源码](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/trpo/trpo.py#L263-L280)。

`x` 是线性方程的近似解，`r` 是残差，`p` 是搜索方向，`z = Ax(p)` 是唯一需要访问矩阵的操作。每轮沿 `p` 更新解与残差，再构造下一个共轭方向。这里按官方实现使用固定的 `cg_iters` 次迭代，`EPS = 1e-8`。

### 源码：步长缩放与回溯线搜索

下面保留 TRPO 分支，省略 NPG 分支和日志；两段连续计算的外层缩进已移除。

```python update · TRPO 分支 | openai/spinningup · 038665d
inputs = {k:v for k,v in zip(all_phs, buf.get())}
Hx = lambda x : mpi_avg(sess.run(hvp, feed_dict={**inputs, v_ph: x}))
g, pi_l_old, v_l_old = sess.run([gradient, pi_loss, v_loss], feed_dict=inputs)
g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)

# Core calculations for TRPO or NPG
x = cg(Hx, g)
alpha = np.sqrt(2*delta/(np.dot(x, Hx(x))+EPS))
old_params = sess.run(get_pi_params)

def set_and_eval(step):
    sess.run(set_pi_params, feed_dict={v_ph: old_params - alpha * x * step})
    return mpi_avg(sess.run([d_kl, pi_loss], feed_dict=inputs))
for j in range(backtrack_iters):
    kl, pi_l_new = set_and_eval(step=backtrack_coeff**j)
    if kl <= delta and pi_l_new <= pi_l_old:
        break

    if j==backtrack_iters-1:
        kl, pi_l_new = set_and_eval(step=0.)
```

[对应源码](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/trpo/trpo.py#L284-L314)。

这里 `g` 是 `pi_loss` 的梯度，`cg(Hx, g)` 返回 $`x\approx H^{-1}g`$，因此参数更新使用 **`old_params - alpha * x * step`**。它与前文直接把负梯度方向记作 `x` 的写法等价。`alpha` 用二次 KL 近似确定初始尺度，`backtrack_coeff**j` 再逐次缩短步长。只有实际平均 KL 不超过 `delta`，且代理损失没有上升，才接受新参数；全部尝试失败后，`step=0` 恢复旧参数。

源码：OpenAI Spinning Up，Copyright © 2018 OpenAI，按 [MIT License](assets/licenses/SpinningUp-LICENSE.txt) 摘录。

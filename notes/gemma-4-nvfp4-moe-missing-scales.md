# Gemma 4 NVFP4 MoE 空输出

**目录**

- [故障现象与根因](#gemma-4-nvfp4-moe-空输出)
- [NVFP4](#nvfp4)
- [矩阵融合](#矩阵融合)
- [Kernel Backend](#kernel-backend)
  - [后端选择原则](#后端选择原则)
  - [本案例：CUTLASS 与 Marlin](#本案例cutlass-与-marlin)

> [!CAUTION]
> **Issue：** [vLLM #51525：Gemma 4 NVFP4 MoE 在 CUTLASS 路径返回 PAD token 或空内容](https://github.com/vllm-project/vllm/issues/51525)
>
> **ROOT CAUSE — `VLLM_CUTLASS` W4A4 消费未完成完整性校验的 per-expert activation scale**
>
> ModelOpt calibration 按模型的真实 MoE routing 收集 activation statistics。低频 expert 若在校准语料中没有接收到 token，对应的 gate/up 或 down projection 就不会形成有效统计量；exporter 随后不会为该 expert 写出 `w13_input_scale` 或 `w2_input_scale`。Packed FP4 weight 及其 weight scale 仍然存在，因此 checkpoint 可以通过只关注权重数量和 shape 的检查
>
> vLLM 的 `ModelOptNvFp4FusedMoE` 使用 `torch.empty` 分配完整的 per-expert scale tensor，loader 只覆盖 checkpoint 中实际存在的条目。加载完成后，代码没有在 backend format conversion 前验证所有 expert slot 都已写入。选择 `VLLM_CUTLASS` 时，format conversion 继续对 `w13_input_scale` 和 `w2_input_scale` 取倒数，生成 `a1_gscale` 和 `a2_gscale`；缺失 slot 中的未初始化 bit pattern 因而被转换为 `inf`、`NaN` 或任意有限垃圾值，并被当作合法量化参数传给 kernel
>
> Router 命中缺项 expert 后，CUTLASS 路径首先将 `a1_gscale[expert_idx]` 传给 `scaled_fp4_experts_quant`。CUDA kernel 用该值计算每 16 个 activation 的 E4M3 block scale，并把归一化结果编码为 E2M1；错误 scale 会在进入第一个 `w13` expert GEMM 前直接破坏 activation。SiLU/mul 之后，第二次量化再由 `a2_gscale[expert_idx]` 驱动，并进入 `w2` expert GEMM。污染随后返回 transformer residual path，继续影响后续 layer、final norm、LM head logits 和 sampling。GPU kernel、worker 和 HTTP server 均可正常结束，因此请求仍返回 HTTP 200，但 token 可能全部为 PAD 或被解码为空内容


| 项目 | 已确认结论 |
| --- | --- |
| 复现模型 | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` |
| 已确认环境 | RTX 5090 / SM120，vLLM `0.24.0` |
| 失败路径 | 显式选择 `VLLM_CUTLASS` NVFP4 MoE，执行 `W4A4 expert computation` |
| checkpoint 缺陷 | 12 个 layer 中共有 25 个 `expert activation scale` 条目缺失 |
| loader 缺陷 | `per-expert scale` 由 `torch.empty` 分配，加载结束后没有验证 `checkpoint coverage` |
| 外部症状 | 请求耗尽 `max_tokens`，返回 PAD token、`content: null` 或空字符串 |

## NVFP4

ModelOpt 是 NVIDIA 的量化工具及量化 checkpoint 格式，不是一种数据类型。`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` 的量化配置包含 `producer=modelopt`、`quant_algo=NVFP4` 和 `group_size=16`：原始 BF16 模型先由 NVIDIA ModelOpt 执行 PTQ，再将量化权重及其 scale 按 ModelOpt 的字段约定保存到 Hugging Face checkpoint。vLLM 启动参数 `--quantization modelopt` 不会在服务启动时重新量化模型；该参数用于声明 checkpoint 的序列化格式，使 loader 按 ModelOpt 的字段名称、shape 和 layout 加载已有量化数据。ModelOpt、NVFP4 和执行 backend 分属不同层级

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">名称</th>
      <th align="center">所属层级</th>
      <th align="center">职责</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center"><strong>Gemma 4 MoE</strong></td><td align="center">模型结构</td><td>决定 expert、router、gate/up/down projection 和每个 token 的执行路径</td></tr>
    <tr><td align="center"><strong>ModelOpt</strong></td><td align="center">量化工具与 schema</td><td>执行 PTQ、calibration 和 export，同时规定量化 checkpoint 的序列化字段</td></tr>
    <tr><td align="center"><strong>NVFP4</strong></td><td align="center">数值格式</td><td>规定 E2M1 数据以及两级 scaling factor</td></tr>
    <tr><td align="center"><strong>CUTLASS / Marlin</strong></td><td align="center">运行时 backend</td><td>vLLM 加载 checkpoint 后实际执行 GEMM</td></tr>
  </tbody>
</table>
</div>

```text
google/gemma-4-26B-A4B-it
        │  BF16 权重
        ▼
NVIDIA ModelOpt 执行 PTQ
        │  导出 NVFP4 W4A4 checkpoint
        ▼
bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4
        │  vLLM 按 ModelOpt schema 加载
        ▼
ModelOptNvFp4FusedMoE
        │
        ├── VLLM_CUTLASS  → W4A4 执行  → 需要 activation input_scale
        │
        └── MARLIN        → W4A16 执行 → 不消费 activation input_scale
```

Gemma 4 的 expert weight 原本是 fused 3D tensor，而不是 128 组普通 `nn.Linear`。该模型的量化脚本先通过 ModelOpt plugin 将 expert 展开为可量化的 Linear，完成量化后再把导出键名转换为 vLLM FusedMoE 所需的形式。模型卡记录的 calibration 配置是 ModelOpt `0.43`、CNN/DailyMail `4096` 个样本、`seq_len=1024`，并保留由真实 router 决定的 natural expert routing；vision encoder 不参与量化

ModelOpt 负责生成 checkpoint，vLLM 不会在启动时重新 calibration。对 MoE expert，checkpoint 中与 NVFP4 计算直接相关的数据可以归纳为：

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">数据</th>
      <th align="center">checkpoint 内容</th>
      <th align="center">粒度</th>
      <th align="center">产生时间</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Weight FP4</td><td><code>w13_weight</code>、<code>w2_weight</code></td><td>两个 E2M1 值打包进一个 byte</td><td>ModelOpt export</td></tr>
    <tr><td>Weight block scale</td><td><code>w13_weight_scale</code>、<code>w2_weight_scale</code></td><td>K 维每 16 个 weight 一个 E4M3 scale</td><td>ModelOpt export</td></tr>
    <tr><td>Weight global scale</td><td><code>w13_weight_scale_2</code>、<code>w2_weight_scale_2</code></td><td>每个 expert/projection 一个 FP32 scale</td><td>ModelOpt calibration/export</td></tr>
    <tr><td>Activation global scale</td><td><code>w13_input_scale</code>、<code>w2_input_scale</code></td><td>每个 expert/projection 一个 FP32 scale</td><td>ModelOpt calibration/export</td></tr>
    <tr><td>Activation FP4 与 block scale</td><td>不在 checkpoint 中</td><td>当前请求路由到 expert 后动态生成</td><td>vLLM runtime</td></tr>
  </tbody>
</table>
</div>

NVFP4 是面向 NVIDIA Blackwell Tensor Core 的 4-bit 浮点量化格式。其数值主体采用 FP4 E2M1：最高 1 bit 为符号位，中间 2 bit 为指数位，最低 1 bit 为尾数位。`6` 是 E2M1 的最大有限幅值，全部编码如下：

```text
0000 ->  0.0    0001 ->  0.5    0010 ->  1.0    0011 ->  1.5
0100 ->  2.0    0101 ->  3.0    0110 ->  4.0    0111 ->  6.0
1000 -> -0.0    1001 -> -0.5    1010 -> -1.0    1011 -> -1.5
1100 -> -2.0    1101 -> -3.0    1110 -> -4.0    1111 -> -6.0
```

vLLM 的 NVFP4 activation kernel 不使用手写分段公式选择 E2M1 编码，而是在 CUDA C++ 中调用 Blackwell PTX 转换指令。下面是 `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh` 中实际执行转换和打包的代码：

```cpp
// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(
    float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y),
        "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
}

__device__ __forceinline__ uint32_t pack_fp4(float2 (&v)[4]) {
  return fp32_vec8_to_e2m1(v);
}
```

`cvt.rn.satfinite.e2m1x2.f32` 是真正完成数值选择的指令：`rn` 表示 round-to-nearest、ties-to-even，`satfinite` 将超出 E2M1 范围的输入饱和到最大有限值，`e2m1x2.f32` 表示从两个 FP32 输入生成两个 E2M1 4-bit 值。四条转换指令共生成 8 个 E2M1 值，每两个值装入一个 byte；`mov.b32` 再把 4 个 byte 合并为一个 `uint32_t`。后面的 `cvt_warp_fp16_to_fp4` 在完成 block scaling 后调用 `pack_fp4`，因此上面的编码表就是由硬件转换指令实施的，而不是由软件逐项查表。NVFP4 不直接用这组离散值拟合整个 tensor，而是采用两级 scale：

| 项目 | 第一层缩放 | 第二层缩放 |
| --- | --- | --- |
| 名称 | Micro-block scale | Tensor-level global scale |
| 格式 | `FP8 E4M3` | `FP32` |
| 范围 | 每连续 `16` 个元素共享一个 scale | 整个 tensor；在 MoE 中细化到 expert/projection |
| 权重 | `weight_scale` | `weight_scale_2` |
| 激活 | 运行时动态生成 | `input_scale` |

第一层 block scale 描述 16 元素 micro-block 的局部动态范围。FP4 E2M1 主要保留 block 内部的相对数值，E4M3 block scale 将该 block 恢复到相应量级。第二层 global scale 描述 tensor 级量化范围，用于把各 block scale 映射到 E4M3 可表示的范围，并在 GEMM 输出缩放时恢复整体数值尺度。E2M1 的最大有限值为 `6`，E4M3 的最大有限值为 `448`，二者对应的组合范围为 `2688`；ModelOpt 保存的 global scale 以这一范围为基础，并可结合 calibration headroom 调整

FP4 数据不能脱离 scale 独立解释。Weight 是静态参数，ModelOpt 在 PTQ/export 阶段把 packed FP4 weight、E4M3 block scale 和 FP32 global scale 一并写入 checkpoint。Activation 随请求内容和 MoE routing 改变，checkpoint 只保存 calibration 得到的 FP32 global scale；FP4 activation 和 E4M3 block scale 由 vLLM 在每次 GEMM 前动态生成

> [!IMPORTANT]
> **`input_scale` 对应第二层 FP32 activation global scale**
>
> 它不是 16 元素 micro-block 使用的 E4M3 block scale，也不是某次请求即时统计得到的 scale；它由 calibration 确定、写入 checkpoint，并在 MoE 中按 expert/projection 使用。动态 block quantization 不能替代 `input_scale`

从原始 activation 到 Tensor Core operand 的运行时处理顺序如下：

```text
BF16/FP16 hidden state
        │  router 已确定 expert_idx
        ▼
读取该 expert/projection 的 FP32 input_scale
        │  vLLM 取倒数，得到传给量化 kernel 的 SFScaleVal
        ▼
每 16 个元素计算绝对值最大值
        │  结合 SFScaleVal 和 E2M1 最大值 6
        ▼
生成并舍入为 FP8 E4M3 block scale
        │  用 block scale 归一化当前 16 个值
        ▼
舍入到 {-6, -4, -3, -2, -1.5, -1, -0.5, 0, ..., 6}
        │  两个 E2M1 值打包为一个 byte
        ▼
FP4 activation + FP8 block scale 进入 W4A4 GEMM
        │  GEMM scaling 同时纳入 FP32 activation/weight global scale
        ▼
恢复到正确的输出数值尺度
```

E4M3 舍入是两级缩放不可省略的原因。global scale 先把不同 block 的局部范围映射到 E4M3 的有效表示区间；block scale 完成舍入后，再与 FP32 global scale 一起恢复整体量级。错误的 `input_scale` 会改变 E4M3 block scale 的舍入或裁剪结果，并进一步改变 E2M1 编码。Weight 侧执行同样的数值重建，只是其 E2M1 数据、E4M3 block scale 和 FP32 global scale 已由 ModelOpt 写入 checkpoint。实际 CUTLASS kernel 会融合 block scaling、Tensor Core GEMM 与 global scaling，但上述数据依赖保持不变

下面是 vLLM 的关键实现，省略地址计算和边界处理。源码文件使用 CUDA intrinsic、template 和 `if constexpr`，属于 CUDA C++，因此代码块标记为 `cpp`，不是 C。实现位于 `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh` 的 `cvt_warp_fp16_to_fp4`：

```cpp
auto localMax = __habs2(vec.elts[0]);

#pragma unroll
for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
  localMax = __hmax2(localMax, __habs2(vec.elts[i]));
}

if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
  localMax = __hmax2(
      __shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
}

float vecMax = float(__hmax(localMax.x, localMax.y));

// Compute and store the E4M3 micro-block scale.
SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
__nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
SFValue = float(tmp);
if (SFout) *SFout = fp8SFVal;

// Derive the normalization factor used before E2M1 packing.
outputScale = SFValue != 0.0f
    ? reciprocal_approximate_ftz(
          SFValue * reciprocal_approximate_ftz(SFScaleVal))
    : 0.0f;

fp2Vals[i].x *= outputScale;
fp2Vals[i].y *= outputScale;
return pack_fp4(fp2Vals);
```

`vecMax` 是当前 16 元素 block 的绝对值最大值；`SFScaleVal` 是 checkpoint activation global scale 的倒数；转换为 `__nv_fp8_e4m3` 后的 `SFValue` 是该 block 的 E4M3 scale；`outputScale` 是写入 E2M1 前使用的归一化因子；`pack_fp4` 负责最终的 E2M1 舍入与打包

MoE 版本不会对所有 token 使用同一个 global scale，而是按照 router 已选中的 `expert_idx` 读取对应条目。关键代码位于 `csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu`：

```cpp
float const SFScaleVal =
    SFScale == nullptr ? 1.0f : SFScale[expert_idx];

out_pos = cvt_warp_fp16_to_fp4<
    Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
        quant_input, SFScaleVal, sf_out);
```

vLLM 的 MoE format conversion 在进入该内核前分别对 `w13_input_scale` 和 `w2_input_scale` 取倒数，生成 `a1_gscale` 和 `a2_gscale`。第一次 expert GEMM 前使用 `a1_gscale` 量化输入；SiLU/mul 后再使用 `a2_gscale` 量化中间 activation，然后执行第二次 expert GEMM

- **PTQ（Post-Training Quantization）：** 不重新训练基础模型，通过 calibration statistics 和量化规则生成低精度 checkpoint。本案例由 ModelOpt 对原始 BF16 Gemma 4 执行 PTQ

- **Calibration：** 使用代表性样本收集 activation range/statistics，以确定运行时不能仅由 weight 推导的量化参数。本案例中的自然 MoE routing 决定哪些 expert 能获得 `input_scale` 统计

- **Static weight quantization：** Weight 在 export 阶段完成量化，packed data 与 scale 固定写入 checkpoint。本案例对应 `w13/w2 weight`、`weight_scale` 和 `weight_scale_2`

- **Dynamic activation quantization：** 运行时依据当前 activation 生成低精度数据及局部 scale。本案例在每次 expert GEMM 前生成 FP4 activation 和 E4M3 block scale

- **Granularity：** 一个 scale 覆盖的数据范围，例如 per-tensor、per-channel、per-group、per-block 或 per-expert。NVFP4 micro-block 覆盖 16 个元素，global activation scale 则细化到 expert/projection

- **Packing：** 将多个低位宽元素编码进 byte 或 machine word；它是存储布局，不是另一种量化算法。本案例中两个 E2M1 值占一个 byte，8 个值由 `pack_fp4` 组成一个 `uint32_t`

- **Backend / kernel：** 消费量化权重与 scale 并执行 GEMM 的运行时实现。本案例中 CUTLASS 执行 W4A4，Marlin 执行不消费 activation `input_scale` 的路径

## 矩阵融合

共享输入的多个 projection 可以沿权重的输出维拼接，一次 GEMM 后再切分结果。以 PyTorch 的 `[out_features, in_features]` 布局为例，融合增加输出宽度，输入维度保持不变。下面用 `H` 表示 hidden size，`I` 表示 FFN intermediate size，`E` 表示 expert 数量；shape 均指量化打包前的逻辑布局。

| 模块 | 融合方式 | 融合后的权重或输出 |
| --- | --- | --- |
| Attention | `q_proj + k_proj + v_proj → qkv_proj` | 权重 `[Q_dim + K_dim + V_dim, H]` |
| Dense FFN | `gate_proj + up_proj → w13` | `w13 [2I,H]`；down 仍为 `w2 [H,I]` |
| MoE FFN | 每个 expert 分别融合 gate/up | `w13 [E,2I,H]`、`w2 [E,H,I]` |
| Gated DeltaNet | Q/K/V/Z 与 B/A 分成两组投影 | 输出分别切为 `[Q,K,V,Z]` 与 `[B,A]` |

**QKV projection** Q、K、V 共享 hidden state，因此三个 GEMM 可以合并为一个。GQA 的 K/V head 数少于 Q，输出必须按实际宽度切分：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

融合只合并输入投影；后续 normalization、RoPE、attention 和 `o_proj` 仍有各自的计算语义。

**Gated FFN** Gate 与 up 共享输入，融合为 `gate_up_proj` / `w13`。Down projection 依赖 activation/mul 的结果，仍需第二次 GEMM。`Gemma4MLP` 的主干只需三步：

```python
gate_up, _ = self.gate_up_proj(x)  # GEMM 1：同时生成 gate 和 up
x = self.act_fn(gate_up)           # activation(gate) × up
x, _ = self.down_proj(x)           # GEMM 2：down projection
```

**MoE FFN** 每个 expert 使用同样的 `w13/w2` 结构，但接收的 token 行数 `M_e` 不同。Backend 根据 routing 结果重排输入，把不同 expert 的矩阵乘组成 Grouped GEMM，减少逐 expert launch 的开销。

CUTLASS NVFP4 在两次 GEMM 前分别量化 activation：`w13_input_scale` 对应 gate/up 的输入，`w2_input_scale` 对应 SiLU/mul 后的中间结果；取倒数后分别得到 `a1_gscale`、`a2_gscale`。

```text
token 重排 → a1_gscale 量化 → Grouped w13 GEMM
         → SiLU/mul + a2_gscale 量化 → Grouped w2 GEMM
         → 恢复 token 顺序并按 routing weight 汇总
```

这里的几种融合发生在不同层次：

| 层次 | 本案例中的变化 |
| --- | --- |
| Projection fusion | Gate/up 权重沿输出维拼成 `w13` |
| Layout conversion | Packed weight 与 scale 转成 CUTLASS/Marlin 要求的排列 |
| Grouped scheduling | 把不同 expert 的变长 GEMM 一起提交 |
| Operator fusion | 把 SiLU、multiply、第二次量化合进同一个 kernel |

**Gated DeltaNet 的投影融合** Qwen3.5 MoE 中，GDN 负责序列混合，MoE 负责 FFN。vLLM 将 checkpoint 的 `in_proj_qkv` 与 `in_proj_z` 合为 `in_proj_qkvz`，将 `in_proj_b` 与 `in_proj_a` 合为 `in_proj_ba`，运行时执行两个输入投影 GEMM：

```python
mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
ba, _ = self.in_proj_ba(hidden_states)
```

QKVZ 的四段输出宽度为 `key_dim`、`key_dim`、`value_dim`、`value_dim`；B/A 的宽度均为 `num_v_heads`。Q/K/V 随后进入 convolution 和 recurrent update，Z 用于输出门控，B/A 控制状态更新。这些后续算子与投影 GEMM 分开，使用低精度 projection 不会自动改变它们的数值格式。

## Kernel Backend

vLLM 按算子分别选择后端。本案例的故障发生在 MoE expert，直接相关的配置是 `--moe-backend`。

| 配置 | 执行范围 |
| --- | --- |
| `--linear-backend` | 普通量化 Linear，如 `qkv_proj`、`o_proj`、Dense FFN |
| `--moe-backend` | MoE 的 token 重排、`w13/w2` expert GEMM、activation 与加权汇总 |

### 后端选择原则

1. **量化格式确定候选集。** NVFP4 Linear 与 MoE 使用独立的 selector，各自维护候选实现。
2. **`auto` 按优先级选择。** 依次检查 GPU 架构、dtype、shape 和并行配置，使用首个满足条件的实现。
3. **显式配置限制候选范围。** MoE 校验指定后端，不兼容时通常报错。Linear 若在指定后端中找不到该层类型的实现，会告警并对该层恢复自动选择。

选定后端后，vLLM 将 packed weight 与 scale 转成该实现要求的 layout，再交给具体 kernel：

```text
量化格式 → Linear / MoE 候选集 → 配置与支持条件检查 → layout conversion → kernel
```

### 本案例：CUTLASS 与 Marlin

| MoE 配置 | Activation 路径 | 缺失 `input_scale` 的影响 |
| --- | --- | --- |
| `--moe-backend cutlass` | W4A4：两次 expert GEMM 前均量化为 FP4 | 读取缺项 expert 的 scale 时，错误数值进入量化与 GEMM |
| `--moe-backend marlin` | W4A16：保持 BF16/FP16 | 不消费 activation `input_scale`，避开本次缺项 |

`cutlass` 对应内部名称 `VLLM_CUTLASS`，实际 expert class 是 `CutlassExpertsFp4`。Format conversion 对 `w13_input_scale`、`w2_input_scale` 取倒数，生成两次 activation quantization 使用的 `a1_gscale`、`a2_gscale`。Marlin 则在转换时将 activation scale 置为 `None`，使用 weight-only 路径。

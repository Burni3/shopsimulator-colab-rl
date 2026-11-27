# Colab 零点击多机训练基础设施与显存/环境调优实战手册

> 本文档记录了在 Google Colab（A100-40GB / Blackwell RTX PRO 6000 96GB G4）上跑通原生 `Slime + Megatron-LM + SGLang + Pi + ShopSimulator` 全栈的工程方案、关键排错实录与显存调优方法，供后续复现与面试复盘使用。

---

## 一、从 50 分钟编译到 33 秒零点击恢复的工程演进

### 1. 一次预编译 + 多架构缓存
- 原生 `Slime` 依赖 CUDA 12.9 + PyTorch 2.11 + FlashAttn + FLA + TransformerEngine + Apex + `torch_memory_saver` + 补丁版 Megatron-LM & SGLang，从源码编译耗时约 **50 分钟**（且原版 `Apex` 32 并发 nvcc 在 52 GB RAM 机器上会直接 OOM，已通过修改为 `--threads 2 --parallel 3` + `MAX_JOBS=2` 压至 10–12 GB 内存峰值）。
- 编译目标设为 `TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"`，实测兼容 **A100 (`sm_80`)、L4 (`sm_89`)、H100 (`sm_90`)** 以及 **Blackwell RTX PRO 6000 96GB (`sm_120`)**。
- 将 `ShopSimulator` 隔离在独立的 `micromamba/envs/shop_env`（Python 3.11 + OpenJDK 21）中，彻底避免其旧版 `flask==2.1.2` / `Werkzeug==2.1.2` / `requests==2.27.1` 污染 Slime/Ray 主环境。

### 2. 彻底告别 Google Drive 弹窗：私有 HF Dataset 零点击恢复
- **痛点**：Colab `drive.mount()` 绑定单机临时 `dfs_ephemeral` token（2–5 分钟过期），每开一台新 VM 必须人工点击浏览器授权，且多机并行极易过期空烧算力。
- **解法**：利用便宜的 Colab CPU 实例（仅消耗 `0.01 CU`）一次性将 `slime_env_bundle.tar.zst`（10.8 GB）与修复后的 `shopsimulator_ready.tar.zst`（653 MB）同步至私有 HF Dataset 仓库 `your-hf-username/pi-slime-shopsimulator-cache`。
- **实测提速**：
  - 配合 [restore_from_hf.sh](../scripts/restore_from_hf.sh) 与 [auto_launch_zero_click.ps1](../scripts/colab_cli/auto_launch_zero_click.ps1)，开卡后 **0 弹窗、0 点击** 自动拉取环境：
    - `slime_env_bundle.tar.zst`（10.8 GB）：**~30–33 秒**（~345 MB/s）；
    - `shopsimulator_ready.tar.zst`（653 MB）：**~2.1–3.3 秒**；
  - 从裸机到进入正式 GRPO/GSPO/OPSD 训练总准备时间仅 **4–6 分钟**，相比从头编译提速 **10–17×**。

---

## 二、40GB A100 与 96GB G4 显存账与零 OOM 调优配方

在 `--colocate` 单卡推理/训练协同模式下，Qwen3.5-2B（1.88B 参数、**248,320 超大词表**）面临三大显存挑战，我们通过实测逐一攻克：

### 1. 挑战一：超长单轨迹（14.8k tokens）+ 248k 词表导致 Autograd 显存撞墙
- **现象**：当多轮购物轨迹单条长达 **14,824 tokens**（`rollout 87` 实测）时，尽管启用了 `--log-probs-chunk-size 512`，Slime 的 `_VocabParallelLogProbEntropy` 仍会在每个 chunk 的 `ctx.save_for_backward` 中累积保存 FP32 `log_prob_softmax`（`14,824 × 248,320 × 4B = 13.71 GiB`），叠加 FP32 `logits`（`13.71 GiB`）+ FP32 梯度缓冲（`7.00 GiB`）+ BF16 参数（`3.50 GiB`）总计 `38.7 GiB`，刚好超出 40GB A100 约 `404 MiB`。
- **根治（`40g-bf16grad-chunked256`，实测释放 +3.75 GiB 永久显存）**：
  - 在 40GB A100 单卡分支移除冗余的 `--accumulate-allreduce-grads-in-fp32`，启用 **`--main-grads-dtype bf16`**（与 `--use-precision-aware-optimizer` 协同，直接永久省下 **3.50 GiB**）；
  - 配合 **`--log-probs-chunk-size 256`**（再降 243 MiB 瞬时峰值）与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，并加上 **`RESUME_IF_EXISTS=1` 自动断点续跑**（`--no-load-optim --no-load-rng`）；
  - **实测结果**：`gspo_native` 从 Step 80 检查点无缝续跑至 Step 100 并完成全量 200 题评测，全程零 OOM！

### 2. 挑战二：96GB Blackwell (G4) 满血显存压榨与关闭重计算甜点标定
- **动机**：在 96GB Blackwell（仅 `11.20 CU/hr`，性价比远超 A100 40GB 的 `8.16 CU/hr`）上若沿用 40GB 的激活重计算（`--recompute-granularity full`），49k 词表训练仅占用 `41.3 / 97.9 GB`，浪费超一半显存带宽。
- **实测极限探底（`DISABLE_RECOMPUTE=1` 关闭全量重计算）**：
  - `MAX_TOK=32768`（49k）→ `94.5 / 95.0 GiB` OOM；
  - `MAX_TOK=24576`（49k）→ Step 1 通过（`88.0 GiB`），Step 2 差 `0.42 GiB` OOM（`95.4 GiB`）；
  - **`MAX_TOK=16384`（49k）/ `12288`（248k）+ `DISABLE_RECOMPUTE=1` + `--log-probs-chunk-size 2048` + `SGLANG_MEM=0.75`** → **黄金甜点！显存稳锁 `76.6 / 97.9 GB（78.3%）`、GPU 利用率 `100%`、`actor_train` 提速 +30%（`1.17 → 1.50+ mb/s`），连续 67 步零 OOM，单步 Megatron 训练仅 `24–31s`！**

### 3. 挑战三：40GB A100 上 OPSD 在线 SGLang 自蒸馏并发 prefill 显存峰值
- **现象**：20 路 rollout 结束后若同时向 SGLang 发送 `return_logprob=True` 计算 248k 词表 logprobs，并发激活叠加容易打爆 40GB 显存（实测 Step 11 OOM）。
- **根治（三锁配方）**：`OPSD_CONCURRENCY=1` 串行流式前向 + `SGLANG_MEM_FRACTION_STATIC=0.45` + `--sglang-chunked-prefill-size 2048`，实测全量 500 题（100 步 / 5.52 小时）`opsd_gspo` 全程 **0 OOM 一次跑通**！

---

## 三、四个高价值踩坑与修复实录

| 问题 | 根因 | 修复方案与验证结果 |
| :--- | :--- | :--- |
| **1. SFT 基线分数腰斩（`r_loose=0.166`, `r_option=0.0`）** | 早期建 Lucene 索引误用英文 WebShop 大写键名（`Title/Description/Attributes`），中文 `items_eval_train.json` 实际是小写 `title/sub_title/shop_name/full_description/attribute/customization_options`，导致索引只剩 `category` | 用正确中文全字段重建 23,421 条文档并更新 HF 缓存包；修复后同一 SFT 模型 `r_loose` 从 **0.1659 → 0.5091（3.07×）**、`r_option` 从 **0 → 24%**、`r_hard` 从 **0 → 24%** |
| **2. Ray 启动报 `pyarrow requires NumPy 2.0`** | 环境中 `numpy==1.26.4`（兼容 Megatron/TE/onnx）与预装 `pyarrow 20+` 冲突 | 固定 `pyarrow==19.0.1 --no-deps` 并固化进 `restore_from_hf.sh` 与 `restore_from_drive.sh` |
| **3. SGLang FlashInfer JIT 报 `cannot find -lcudart`** | FlashInfer 链接器写死 `-L$CUDA_HOME/lib64`，而 micromamba 的 `libcudart.so` 在 `lib/` | 启动前自动建立 `lib64 -> lib` 及 `lib/stubs` 软链接 |
| **4. Colab 15 分钟空闲回收 & 本机休眠断连防护** | 仅 HTTP 轮询不重置 WebSocket 计时器；且 Windows 夜间自动休眠会中断 `watch_exp.ps1` 的 WebSocket 保活，导致云端 VM 约 30–90 分钟后被回收 | 脱离内核启动后台作业 + `watch_exp.ps1` 每 120s 发真实 WebSocket `colab exec` 保活（带 `-NoStop` 防误删机）+ `--save-interval 20` 断点自动续跑 + 每 15 分钟增量上传 `train_live.log` 至私有 HF 仓库；**后续夜间无人值守建议在 PowerShell 挂载 `SetThreadExecutionState` 防本机休眠** |

---

## 四、关键本地资产索引

- 算法与创新插件：[shop_innovations.py](../src/shop_innovations.py)
- 领域定向 BBPE 词表无损剪枝脚本：[prune_vocab_qwen35.py](../src/prune_vocab_qwen35.py)
- 统一实验与评测脚本：[run_experiment.sh](../scripts/run_experiment.sh)
- 零点击多机自动拉起：[auto_launch_zero_click.ps1](../scripts/colab_cli/auto_launch_zero_click.ps1) / [hunt_multi_gpu.ps1](../scripts/colab_cli/hunt_multi_gpu.ps1) / [watch_exp.ps1](../scripts/colab_cli/watch_exp.ps1)
- HF 私有仓库恢复与结果拉取：[restore_from_hf.sh](../scripts/restore_from_hf.sh) / [fetch_hf_results.py](../scripts/fetch_hf_results.py)
- 已同步到本地的实验评测结果：[hf_synced_results/](../results/)

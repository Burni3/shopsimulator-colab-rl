# Colab 零点击多机训练基础设施与显存/环境调优实战手册

> 本文档记录了在 Google Colab（A100-40GB / Blackwell RTX PRO 6000 96GB G4）上跑通原生 `Slime + Megatron-LM + SGLang + Pi + ShopSimulator` 全栈的工程方案、关键排错实录与显存调优方法，供后续复现与面试复盘使用。

---

## 一、从零构建全栈环境完整指南（避坑与 33 秒零点击恢复）

在单卡 Colab / Linux 云主机上从零拉起 **`Slime + Megatron-LM + SGLang + FlashAttn + FLA + TransformerEngine + Apex + Pi + ShopSimulator`** 是一套非常重的异构技术栈（Python 3.12 + Python 3.11 + OpenJDK 21 + Node.js 22 + CUDA 12.9），直接按默认命令安装极易触发编译内存爆炸、依赖冲突与索引静默失效。以下是我们踩完所有坑沉淀出的**可复现工程标准方案**：

### 1. 三套运行时严格物理隔离（彻底杜绝依赖地狱）
| 运行时 | 独立目录 | 核心版本 | 隔离原因 |
| :--- | :--- | :--- | :--- |
| **RL 主环境（Slime）** | `micromamba/envs/slime` | Python 3.12 + CUDA 12.9 + PyTorch 2.11 + Ray + SGLang + Megatron-LM + TE + Apex + `torch_memory_saver` | 高版本 CUDA/Torch 栈，需要最新 `aiohttp / requests / transformers` |
| **电商模拟环境（ShopSimulator）** | `micromamba/envs/shop_env` | Python 3.11 + OpenJDK 21 + Pyserini + Lucene + `flask==2.1.2` + `Werkzeug==2.1.2` | 旧版 Flask/Werkzeug/requests 与 Slime/Ray/HF Hub 严重冲突，必须独立进程通过 HTTP `:5000` 通信 |
| **Agent 执行壳（Pi）** | `pi-npm/node_modules/.bin/pi` | Node.js 22 + `@earendil-works/pi-coding-agent@0.84.2` | 独立 Node22 环境，不依赖宿主系统旧版 Node |

### 2. 源码编译防 OOM 与跨代 GPU 一次编译配方
- **Apex / CUDA 扩展防编译期 OOM（`exit 137`）**：
  - 原生 `Apex` / `FlashAttn` 编译默认拉起 32 路并发 `nvcc`，在 Colab（52–83 GB RAM）上瞬间打爆内存被内核 OOM Killer 杀掉；
  - **实战方案**：强制限制编译并发为 **`--threads 2 --parallel 3`** 与 **`MAX_JOBS=2`**，将整个编译过程内存峰值稳定压在 **10–12 GB**，耗时约 50 分钟一次成功。
- **跨代 GPU 多架构一次预编译**：
  - 编译前显式设置 **`TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"`**，只需在便宜的 L4/A100 上编译一次，实测即可直接兼容 **A100 (`sm_80`)、L4 (`sm_89`)、H100 (`sm_90`)** 以及 **Blackwell RTX PRO 6000 96GB (`sm_120`)**，再也无需换卡重编！

### 3. 三大必踩运行时雷区与自动化自愈脚本（已固化进 `restore_from_hf.sh` 与 `run_experiment.sh`）
1. **Ray 启动即崩：`pyarrow requires NumPy 2.0`**
   - **根因**：Megatron / TransformerEngine / ONNX 生态锁定 `numpy==1.26.4`，而新版 `pyarrow>=20` 强依赖 NumPy 2.0；
   - **1 行根治**：`pip install -q --no-deps --force-reinstall "pyarrow==19.0.1"`。
2. **SGLang FlashInfer JIT 链接失败：`cannot find -lcudart` / `libcuda.so`**
   - **根因**：FlashInfer JIT 写死 `-L$CUDA_HOME/lib64`，但 micromamba 安装的 CUDA 库位于 `$SLIME_ENV/lib/`，驱动 stubs 位于 `targets/x86_64-linux/lib/stubs`；
   - **2 行根治**：启动前幂等创建软链 `ln -s "${SLIME_ENV}/lib" "${SLIME_ENV}/lib64"` 及 `ln -s ".../stubs" "${SLIME_ENV}/lib/stubs"`。
3. **ShopSimulator 中文 Lucene 索引字段大小写错配导致分数腰斩 3×**
   - **根因**：上游英文 WebShop 脚本按大写键名（`Title / Description / Attributes`）建索引，但中文商品库 `items_eval_train.json` 实际是小写字段（`title / sub_title / category / shop_name / full_description / attribute / customization_options`），导致索引只剩 `category`（平均文档长度 `<30` 字符），同一 SFT 模型 `r_loose` 暴跌至 `0.1659`、`r_option = 0.0`；
   - **自动体检自愈**：`run_experiment.sh` 启动前自动抽检 `documents.jsonl` 前 20 条均长，若 `<60` 字符则自动用中文全字段重建 23,421 条 Lucene 索引并同步回缓存包；修复后同一 SFT 模型 `r_loose` 立刻从 **`0.1659 → 0.5091（3.07×）`**、`r_option` 从 **`0 → 24%`**！
4. **ShopSimulator 20 并发槽位泄漏自检**
   - 异常中断会导致模拟环境槽位被僵尸会话占用；脚本在每次实验前强制 POST `{"action":"release_all"}` 并断言 `status` 返回 `"free": 20`（不满足则自动重启 `pack_api.py`），确保 20 并发 rollout 零阻塞。

### 4. 彻底告别 Google Drive 弹窗：分层打包 + 私有 HF Dataset 33 秒零点击恢复
- **为什么放弃 Google Drive**：Colab `drive.mount()` 绑定单机临时 `dfs_ephemeral` token（2–5 分钟即过期），每开一台新 VM 必须人工点浏览器授权，多机并行与夜间重开极易卡死空烧 CU。
- **分层缓存架构**：
  - 用极低成本 Colab CPU 实例（仅消耗 `0.01 CU`）使用多线程 `zstd -T0 -3` 将编译好的环境拆分为两层上传至私有 HF Dataset：
    1. `slime_env_bundle.tar.zst`（10.8 GB：Slime + Megatron + SGLang + Pi + Node22 + `micromamba/envs/slime`）；
    2. `shopsimulator_ready.tar.zst`（653 MB：已修复全字段中文 Lucene 索引的 `ShopSimulator` + `micromamba/envs/shop_env`）。
  - 这样当只更新 ShopSimulator 索引或环境脚本时，无需重传 10.8 GB 主包！
- **实测恢复耗时**：
  - 配合 [restore_from_hf.sh](../scripts/restore_from_hf.sh) 与 [auto_launch_zero_click.ps1](../scripts/colab_cli/auto_launch_zero_click.ps1)，裸机开卡 **0 弹窗、0 点击**：
    - 拉取 `slime_env_bundle.tar.zst`（10.8 GB）：**28–33 秒**（实测下载速度 **345–381 MB/s**）；
    - 拉取 `shopsimulator_ready.tar.zst`（653 MB）：**2.1–3.3 秒**；
  - 从全新裸机到进入正式 GRPO / GSPO / OPSD 训练总准备时间仅 **4–6 分钟**，相比从头编译提速 **10–17×**！

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

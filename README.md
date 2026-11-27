# ShopSimulator 多轮购物 Agent 后训练实验室（Colab 单卡全栈）

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Base Model: Qwen3.5-2B](https://img.shields.io/badge/Model-Qwen3.5--2B-blue.svg)](https://huggingface.co/Qwen)
[![Framework: Slime + Megatron + SGLang](https://img.shields.io/badge/Stack-Slime%20%7C%20Megatron%20%7C%20SGLang-orange.svg)](https://github.com/THUDM/slime)

本项目是一套**零侵入上游源码**的多轮工具调用 Agent 强化学习（RL）与工程调优实验套件，基于 [`Piucente/pi-slime-shopsimulator`](https://github.com/Piucente/pi-slime-shopsimulator)（`Qwen3.5-2B` + `Pi` + `Slime` + `ShopSimulator`）构建，完整实现了 **Google Colab 单卡（A100 40GB / H100 80GB / Blackwell RTX PRO 6000 96GB）** 上的算法创新、领域词表无损剪枝与极限显存工程调优。

---

## 核心亮点与原创贡献

> 本仓库严格遵循**独立插件式架构**，所有算法与工程代码均为独立原创，不修改、不打包分发上游仓库源码与数据集（详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。

1. **多算法对比与原创 Privileged-Context OPSD（特权上下文在线自蒸馏）**（[src/shop_innovations.py](src/shop_innovations.py)）
   - 统一集成 **GSPO**（序列级几何平均重要性采样）、**CISPO + 复合硬奖励塑形**（针对性攻克 SKU 规格与严格全对率短板）以及 **原创 Privileged-Context OPSD**；
   - **Privileged-Context OPSD** 无需外部大模型、**零常驻显存增量**：在线 rollout 结束瞬间，将 ShopSimulator 返回的四维结构化诊断（`r_type / r_att / r_option / r_price`）精准注入 Qwen3.5 `system` 模板作为特权上下文，复用本地 colocate SGLang 引擎毫秒级计算同权重 Self-Teacher 的逐 token 条件对数概率，通过 token 级反向 KL 与 GSPO 优势正交融合，把稀疏终局奖励升级为稠密 token 级信用分配并盘活组内零方差样本；
   - 同时内置面向 ≥70GB 大显存卡的跨规模外部教师 **`Qwen3.5-4B` OPD** 扩展分支。

2. **领域定向 BBPE 词表无损剪枝（`248,320 → 49,152`，5.05× 压缩）与零样本意外红利**（[src/prune_vocab_qwen35.py](src/prune_vocab_qwen35.py)）
   - 零测试泄露扫描 ShopSimulator 全量商品库与训练集（**8,692 万 tokens**），发现整个中文电商环境仅命中 **38,907 个唯一 token（仅占原词表 15.67%）**；
   - 采用四级保底策略（256 字节原子 + 特殊控制符 + 领域命中词与 **BPE Merge Tree 传递闭包** + 全部单字汉字/ASCII + 预训练高频补齐至 128 对齐的 `49,152`），通过行切片将共享嵌入矩阵从 5.09 亿参数压至 1.01 亿，总参数量从 **1.88B → 1.47B（-21.7%）**；
   - 商品库 token 膨胀率仅 **+0.001%**，未见测试集仅 **+1.09%**，`MultiTurnLossMaskGenerator(qwen3_5)` 全链路通过，Megatron 与 SGLang `log_probs` 小数点后 4 位精准对齐；
   - **零样本意外发现（Domain-Constrained Tail Truncation）**：在 **0 步训练**下，49k 剪枝版 SFT 在全量 200 题测试集上直接反超原版 248k SFT（`r_loose` `0.4599 → 0.5123`，`r_success` `16.0% → 21.0%`，`right_product` `24.0% → 33.0%`），原因在于全词表自由采样（`T=1.0, top_p=1.0`）下裁剪 ~20 万非领域长尾词消除了长尾概率泄漏；
   - 配合 Blackwell 96GB 关重计算配方，单步 `log_probs` 低至 **4.6–5.9s**，单步训练耗时仅 **24–31s**（相比 40GB A100 248k 提速 **4–5×**）。

3. **单卡极限显存与 Colab 零点击多机基础设施**（[scripts/run_experiment.sh](scripts/run_experiment.sh)、[docs/02_Colab_ZeroClick_Infra_and_40G_Tuning.md](docs/02_Colab_ZeroClick_Infra_and_40G_Tuning.md)）
   - **33 秒零点击环境恢复**：预编译 Slime + Megatron + SGLang + 修复后中文全字段 Lucene 索引的 ShopSimulator 打包至私有 HF Dataset，新开卡 33 秒恢复，彻底告别 50 分钟编译与 Google Drive 人工弹窗授权；
   - **40GB A100 根治 14.8k 超长单轨迹 OOM（`40g-bf16grad-chunked256`）**：单卡移除冗余 `--accumulate-allreduce-grads-in-fp32` 并启用 `--main-grads-dtype bf16`（永久释放 **+3.50 GiB**）+ `--log-probs-chunk-size 256` + `expandable_segments:True` + 断点自动续跑（`--no-load-optim --no-load-rng`），实现全量 500 题零 OOM；
   - **40GB A100 OPSD 三锁稳态配方**：`OPSD_CONCURRENCY=1` + `SGLANG_MEM=0.45` + `--sglang-chunked-prefill-size 2048`，全量 100 步（5.52 小时）零 OOM 一次跑通；
   - **96GB Blackwell (G4) 满血关闭重计算标定**：实测探底 `MAX_TOK=16384 (49k) / 12288 (248k) + DISABLE_RECOMPUTE=1 + chunk=2048`，显存稳锁 **76.6–82.6 / 97.9 GB（~80%）**、GPU **100%**、实测训练算力达 **207 TFLOPS / 23,918 tok/s**；
   - **Windows 自动化抢卡、Modern Standby 防休眠保活与实时日志备份**：见 [scripts/colab_cli/](scripts/colab_cli/)。

---

## 实验结果汇总

> 所有实验均采用统一超参（`AdamW lr=1e-6`, `rollout_batch_size=5`, `n_samples_per_prompt=4`, `global_batch_size=20`, `max_model_turns=40`, `max_context_len=16384`, `T=1.0, top_p=1.0`），原始评测指标 JSON 均归档在 [results/](results/) 目录。

### 1. 全量 500 题正式实验（`rl_500.jsonl` 100 步 / 2,000 轨迹 → `official_test_200.jsonl` 全量 200 题）

| 实验项 | 算法与词表配置 | `done_rate` | 零奖励率↓ | `r_loose` | `r_att` | `r_option` | `r_price` | `r_hard` | 严格成功 `r_success` | `right_product` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eval_baseline_sft` | SFT 起跑线（248k / 1.88B，0 步） | 81.5% | 21.0% | 0.4599 | 0.4936 | 0.1975 | 0.6600 | 0.1747 | 16.0% (32/200) | 24.0% (48/200) |
| **`eval_vocab49k_sft`** | **BBPE 49k 剪枝 SFT（1.47B，0 步无训练）** | 83.5% | 20.0% | **0.5123** | 0.5548 | 0.2692 | 0.6950 | 0.2389 | **21.0% (42/200)** | **33.0% (66/200)** |
| **`gspo_native`** | **GSPO 全量 500 题（248k，100 步）** | **95.5%** | 6.0% | **0.6941** | **0.7404** | **0.4100** | **0.8700** | **0.3845** | **34.0% (68/200)** | **49.5% (99/200)** |
| **`opsd_gspo`** | **原创 OPSD + GSPO 全量（248k，100 步）** | **95.5%** | **5.5%** | 0.6722 | 0.7115 | 0.4083 | 0.8600 | 0.3798 | 33.5% (67/200) | 44.5% (89/200) |

- **核心结论**：
  - 全量 `gspo_native` 将严格全对率 **`r_success` 从 16.0% 提升至 34.0%（2.13×）**、命中正确商品 **`right_product` 从 24.0% 提升至 49.5%（2.06×）**、SKU 规格正确率 **`r_option` 从 19.75% 提升至 41.00%（2.08×）**；
  - 原创 `opsd_gspo` 紧咬各项核心指标并拿下**全组最低的完全零分率（5.5%）**；对比分析也揭示出固定常数 `opd_kl_coef=0.15` 在中后期偏强，后续升级为 **KL 余弦退火 / 零方差失败组门控（Gated OPSD）** 可进一步兼顾托底与探索上限。

### 2. 第一阶段 50 题快跑消融（`rl_50` 10 步 / 200 轨迹 → `official_test_50`）

| 实验项 | 算法与配置 | `done_rate` | `r_loose` | `r_att` | `r_option` | `r_price` | `r_hard` | `r_success` | `right_product` | 训练耗时 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eval_baseline_sft` | SFT 起跑线 | 80.0% | 0.5091 | 0.5204 | 0.2400 | 0.7800 | 0.2400 | 24.0% | 30.0% | — |
| `gspo_native` | GSPO + `r_loose` | **92.0%** | **0.5757** | 0.6062 | **0.2800** | 0.8200 | 0.2586 | 24.0% | 30.0% | 45.9 min |
| `opsd_gspo` | 原创 OPSD + GSPO + 复合硬奖励 | **92.0%** | 0.5738 | **0.6133** | 0.2500 | **0.8400** | 0.2500 | 24.0% | **32.0%** | 55.4 min |
| **`cispo_hard`** | **CISPO + 复合硬奖励** | 86.0% | 0.5744 | 0.6118 | **0.2800** | 0.8000 | **0.2700** | **26.0%** | **32.0%** | **35.3 min** |

---

## 目录结构

```text
shopsimulator-colab-rl/
├── README.md                                  # 项目说明与实验总览
├── LICENSE                                    # MIT 开源协议
├── THIRD_PARTY_NOTICES.md                     # 上游项目协议与致谢声明
├── .env.example                               # 环境变量模板（HF_TOKEN / HF_CACHE_REPO）
├── src/
│   ├── shop_innovations.py                    # Privileged-Context OPSD / 复合硬奖励 / 外部 4B OPD 插件
│   └── prune_vocab_qwen35.py                  # Qwen3.5 领域定向 BBPE 传递闭包词表剪枝工具
├── scripts/
│   ├── run_experiment.sh                      # 统一训练/评测与多卡显存自适应调度脚本
│   ├── restore_from_hf.sh                     # 私有 HF Dataset 预编译环境 33 秒零点击恢复脚本
│   ├── fetch_hf_results.py                    # HF 实验指标同步脚本
│   ├── queues/                                # 各实验队列配置文件
│   └── colab_cli/                             # Colab 多卡抢卡、后台拉起与防休眠监控脚本
├── results/
│   ├── exp_suite_v1/                          # 50 题快跑消融 4 组原始评测 JSON
│   └── full_rl500_v1/                         # 全量 200 题 4 组原始评测 JSON
└── docs/
    ├── 01_RL_OPSD_Architecture_and_Algorithms.md  # 算法原理、OPSD 与 BBPE 剪枝深度复盘
    └── 02_Colab_ZeroClick_Infra_and_40G_Tuning.md # Colab 零点击基建与 40GB/96GB 显存调优手册
```

---

## 环境构建与零踩坑核心经验（完整版见 [工程手册](docs/02_Colab_ZeroClick_Infra_and_40G_Tuning.md)）

搭建 `Slime + Megatron-LM + SGLang + Pi + ShopSimulator` 异构全栈本身就是整个实验最大的工程门槛之一，本项目总结了 5 条可直接复用的核心避坑配方：

1. **三环境物理隔离**：`micromamba/envs/slime`（Python 3.12 + CUDA 12.9 + PyTorch 2.11）、`micromamba/envs/shop_env`（Python 3.11 + OpenJDK 21 + 旧版 Flask/Werkzeug + Pyserini/Lucene）与独立 `Node.js 22`（Pi CLI）完全隔离，杜绝旧版依赖炸毁 Ray / SGLang；
2. **防编译期 OOM 与跨卡复用**：编译 Apex / FlashAttn 时限并发为 `--threads 2 --parallel 3` + `MAX_JOBS=2`（内存峰值压至 10–12 GB），并指定 `TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"`，一次编译通吃 **A100 / L4 / H100 / Blackwell 96GB**；
3. **运行时三大必踩雷自愈**：固定 `pyarrow==19.0.1 --no-deps`（解决 NumPy 1.26.4 与 Ray 冲突）、自动创建 `$SLIME_ENV/lib64 -> lib` 与 `lib/stubs` 软链（修复 FlashInfer JIT 找不到 `-lcudart`）、启动前自动抽检 `documents.jsonl` 均长并在检测到英文大写字段误建索引时自动重建 23,421 条中文全字段 Lucene 索引（直接将 SFT 基线从 `0.1659` 救回 `0.5091`）；
4. **分层打包 + 私有 HF Dataset 33 秒零点击恢复**：将主环境（10.8 GB）与模拟环境（653 MB）用 `zstd -T0 -3` 分层打包，借助私有 HF Dataset 实测 **345–381 MB/s** 拉取，彻底告别 Google Drive 弹窗授权与 50 分钟重复编译，裸机到开训仅 **4–6 分钟**；
5. **Colab 无人值守防断连**：`start_new_session=True` 脱离内核后台运行 + 每 120s 真实 WebSocket `colab exec` 保活 + Windows 11 Modern Standby 防休眠锁 + `--save-interval 20` 自动断点续跑。

---

## 快速复现指南

### 1. 准备前置依赖
请先按照上游仓库 [`Piucente/pi-slime-shopsimulator`](https://github.com/Piucente/pi-slime-shopsimulator) 的说明准备基础运行环境（或自行打包预编译环境至个人私有 HF Dataset），并参考 `.env.example` 配置环境变量：

```bash
export HF_TOKEN="hf_your_token_here"
export HF_CACHE_REPO="your-hf-username/pi-slime-shopsimulator-cache"
```

### 2. 运行离线领域词表剪枝（`248,320 → 49,152`）
```bash
python src/prune_vocab_qwen35.py \
  --src-hf /content/pi-slime-work/models/sft_hf \
  --dst-hf /content/pi-slime-work/models/sft_hf_vocab49k \
  --rl-jsonl /content/pi-slime-work/pi-slime-shopsimulator/slime/examples/ShopSimulator/data/tasks_v2/rl_500.jsonl \
  --eval-jsonl /content/pi-slime-work/pi-slime-shopsimulator/slime/examples/ShopSimulator/data/tasks_v2/official_test_200.jsonl \
  --shop-dir /content/pi-slime-work/ShopSimulator \
  --slime-dir /content/pi-slime-work/pi-slime-shopsimulator/slime \
  --target-vocab-sizes "49152" \
  --prune-vocab-size 49152
```

### 3. 运行统一全量 RL / OPSD / 评测流水线
脚本会自动识别当前 GPU 档位（`A100-40GB` / `H100-80GB` / `Blackwell-96GB`）并匹配对应显存优化配方：

```bash
# 示例 1：全量 500 题 GSPO 训练 + 200 题评测
EXP_NAME=gspo_native TRAIN_TASKS=500 EVAL_TASKS=200 RUN_SUITE=full_rl500_v1 SAVE_INTERVAL=20 \
  bash scripts/run_experiment.sh

# 示例 2：全量 500 题原创 Privileged-Context OPSD + GSPO
EXP_NAME=opsd_gspo TRAIN_TASKS=500 EVAL_TASKS=200 RUN_SUITE=full_rl500_v1 SAVE_INTERVAL=20 \
  bash scripts/run_experiment.sh

# 示例 3：49k 剪枝词表全量 GSPO（自动触发离线剪枝与权重转换）
EXP_NAME=gspo_vocab49k TRAIN_TASKS=500 EVAL_TASKS=200 RUN_SUITE=full_rl500_v1 SAVE_INTERVAL=20 \
  bash scripts/run_experiment.sh

# 示例 4：直接评测原作者公开 GRPO 模型（200 题同环境对比）
EXP_NAME=eval_author_grpo EVAL_TASKS=200 RUN_SUITE=full_rl500_v1 \
  bash scripts/run_experiment.sh
```

---

## 延伸阅读与技术文档

- [算法架构、Privileged-Context OPSD 与 BBPE 词表剪枝复盘](docs/01_RL_OPSD_Architecture_and_Algorithms.md)
- [Colab 零点击多机基建与 40GB / 96GB 显存调优实战手册](docs/02_Colab_ZeroClick_Infra_and_40G_Tuning.md)

---

## 致谢与版权声明（Acknowledgements & Attribution）

本项目深度受益于以下优秀开源工作，特此致谢：
- **[Piucente/pi-slime-shopsimulator](https://github.com/Piucente/pi-slime-shopsimulator)**（MIT License）：提供 `Qwen3.5-2B + Pi + Slime + ShopSimulator` 最小参考基线与公开 SFT 权重 `mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep`；
- **[THUDM/slime](https://github.com/THUDM/slime)**（Apache-2.0 License）：提供高性能 Megatron-LM + SGLang Colocate RL 训练框架；
- **[ShopAgent-Team/ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator)**：提供多轮中文电商模拟环境；
- **[Qwen3.5-2B](https://huggingface.co/Qwen)**（Apache-2.0 License）、**SGLang**、**Megatron-LM** 与 **Pi**。

本项目自身原创代码与文档以 **MIT License** 开源，详见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

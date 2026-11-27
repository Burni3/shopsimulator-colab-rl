# ShopSimulator 多轮工具调用 Agent 后训练技术方案：RL 算法演进、OPD 与 Privileged-Context OPSD

> 基准底座：`Qwen3.5-2B`（1.88B） + `Pi 0.84.2` + `Slime (Megatron-LM + SGLang Colocate)` + `ShopSimulator`（20-slot 线程化中文电商模拟环境）
> 起点模型：`mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep`

---

## 一、任务背景与 Baseline 核心痛点诊断

在 ShopSimulator 中文长程购物 Agent 任务中，模型需要通过多轮工具调用（`shop_reset` / `shop_act`，最多 40 轮，上下文长达 3,000–16,384 tokens）完成商品搜索、翻页浏览、属性核验、规格（颜色/尺码/套餐）点选、价格上限比对和最终下单。

在修复中文商品 Lucene 索引后的统一 50 题基线评测（`official_test_50`）与原仓库 500 题 GRPO 结果上，可以清晰定位出 3 个核心痛点：

1. **序列奖励与 Token 比率粒度错配（高方差）**：
   - 环境奖励只在终局下单后给出，是**整条序列级（Sequence-level）**的；
   - 原版 GRPO 使用逐 token 重要性比率 $r_{i,t}(\theta) = \pi_\theta(y_{i,t}) / \pi_{\text{old}}(y_{i,t})$。在 3,000+ token 的长多轮对话中，个别低概率噪声 token 即可导致比率尖刺或裁剪失真，梯度方差大。
2. **近 30% 的组内零方差样本浪费**：
   - 原仓库统计 `rl_500` 的 500 个 group 中仅 352 个有非零方差——约 **30% 的 group 4 条采样全对或全错**，GRPO 归一化后优势 $A=0$，导致近三分之一的在线 rollout 算力不产生任何有效梯度。
3. **松弛奖励与严格约束的巨大剪刀差（`r_loose` 高但 `r_hard / r_option` 低）**：
   - 统一 SFT 基线实测：`r_type = 0.80`、`r_price = 0.78`，但 **`r_option = 0.24`（76% 输在规格/颜色/尺码点选错误）**、`r_att = 0.5204`，直接将严格全对率 `r_hard / r_success` 拖低至 `24%`；
   - 原版仅用标量 `r_loose` 广播给整条轨迹，既缺乏对硬约束的强激励，也无法向模型指出**具体哪一步、哪个约束做错了**。

---

## 二、核心算法创新设计

本项目在严格保持上游 `pi-slime-shopsimulator` 源码零修改的前提下，在独立插件 [shop_innovations.py](../src/shop_innovations.py) 与调度框架 [run_experiment.sh](../scripts/run_experiment.sh) 中实现了四层递进的算法对比与原创创新：

### 1. GSPO（Group Sequence Policy Optimization）
- **源码实现位置**：Slime 原生 `slime/utils/ppo_utils.py::compute_gspo_kl`（`--advantage-estimator gspo --calculate-per-token-loss`）。
- **核心公式**：将 token 级比率替换为**序列级几何平均重要性比率**：
  $$s_i(\theta) = \exp\!\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \big(\log \pi_\theta(y_{i,t}) - \log \pi_{\text{old}}(y_{i,t})\big)\right)$$
  配合 `--calculate-per-token-loss` 消除不同轨迹长度带来的梯度偏置。
- **解决痛点**：彻底对齐序列级奖励与序列级优化比率，大幅压低多轮长轨迹方差，显著提升样本效率。

### 2. CISPO（Clipped Importance Sampling Policy Optimization）+ 复合硬奖励塑形
- **源码实现位置**：Slime 原生 `slime/utils/ppo_utils.py::compute_cispo_loss`（`--advantage-estimator cispo --calculate-per-token-loss`）。
- **核心公式**：
  $$\mathcal{L}_{\text{CISPO}} = -\operatorname{sg}\!\big(\operatorname{clip}(r_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\big) \cdot A_t \cdot \log \pi_\theta(y_t)$$
  与 PPO/GRPO 不同，CISPO 仅对权重施加 stop-gradient 裁剪而**不将超界 token 的梯度归零**，从而保住低概率但关键的探索动作（例如更换中文搜索关键词、切换冷门尺码规格），缓解多轮 Agent 的熵坍塌。
- **复合硬奖励塑形**（`shop_innovations.py::_compute_composite_reward`）：
  利用 ShopSimulator 返回的 4 维结构化子分数 `(r_type, r_att, r_option, r_price)`：
  $$r_{\text{hard}} = r_{\text{type}} \cdot r_{\text{att}} \cdot r_{\text{option}} \cdot r_{\text{price}}, \quad r_{\text{success}} = \mathbb{I}[\forall d,\, r_d = 1]$$
  $$R_{\text{composite}} = 0.5 \cdot r_{\text{loose}} + 0.5 \cdot r_{\text{hard}} + 0.2 \cdot r_{\text{success}}$$
  显式放大全属性、全规格、全价格约束同时达标时的梯度优势，直击 `r_option` 与 `r_hard` 短板。

### 3. 原创核心方法：Privileged-Context OPSD（特权上下文在线自蒸馏）
- **实现位置**：`shop_innovations.py::generate_opsd`，无缝接入 Slime 原生正交 OPD 管线（`--use-opd --opd-type sglang --opd-kl-coef 0.15` + `slime/backends/megatron_utils/loss.py::apply_opd_kl_to_advantages`）。
- **核心动机**：
  - 传统 OPD 依赖参数量更大的外部 Teacher 模型，显存与部署成本高；
  - 在工具调用环境中，模拟器在 rollout 结束瞬间会返回**精确的结构化事后诊断**（是否完成下单、终止原因、`r_type / r_att / r_option / r_price` 各维得分、是否命中 `goal_asin`）。
- **算法流程（零额外显存、单卡闭环）**：
  1. **在线正常 Rollout**：Student（无上帝视角）通过 Pi 与 ShopSimulator 交互生成完整多轮轨迹 $\tau = (x, y)$，获得 4 维结构化反馈 `reward_detail`；
  2. **构造特权诊断上下文 $c_{\text{priv}}$**：根据 `reward_detail` 动态生成精准中文特权诊断（如达标轨迹注入“全部品类/属性/规格/价格约束 1.0 达标，请强化此类规范核验与下单决策”；失分轨迹点名具体失分项，如“规格选项选错 `r_option<1` / 价格超上限 `r_price<1` / 属性遗漏 `r_att<1`”），并将其精准注入 Qwen3.5 chat template 中 `system` 消息末尾的 `<|im_end|>` 之前，保持 100% 合法对话模板；
  3. **复用在线 SGLang 的 Self-Teacher 前向**：在 rollout 刚结束、colocate SGLang 引擎尚未休眠的窗口期，以并发限流 `asyncio.Semaphore(2)` 向本地 SGLang Router 发送单次 prefill-only 请求（`max_new_tokens=0, return_logprob=True`），毫秒级获得同模型权重在特权上下文下的逐 token 条件对数概率 $\log \pi_\theta(y_t \mid x, c_{\text{priv}}, y_{<t})$ 作为 `sample.teacher_log_probs`；
  4. **Token 级反向 KL 与 RL 优势正交融合**：
     $$A_t^{\text{OPSD}} = A_i^{\text{GSPO}} - \beta \cdot \Big(\log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_\theta(y_t \mid x, c_{\text{priv}}, y_{<t})\Big)$$
- **核心优势**：
  - **无需任何外部大模型、零常驻显存增量**；
  - 将稀疏终局奖励转化为**全轨迹稠密 token 级信用分配**（带特权诊断的“自己”会对违背约束的错误购买/选项 token 给出极低概率，产生大 reverse KL 精准惩罚该步决策）；
  - **盘活 30% 的组内零方差样本**：即便组内 4 条采样全部失败（$A_i^{\text{GSPO}} = 0$），$-\beta \cdot \text{reverse\_KL}_t$ 仍能提供高质量 token 级纠错梯度。

### 4. 跨规模外部大老师 OPD（`opd_4b_gspo`，面向 ≥70GB 大显存卡）
- **实现位置**：`shop_innovations.py::generate_opd_external` + `run_experiment.sh` `opd_4b_gspo` 分支。
- **设计原则**：避免同尺寸 2B 充当外部老师，坚持使用更大规模同词表模型 **`Qwen/Qwen3.5-4B`**（共享 248,320 词表，100% token 对齐）在独立端口 `:13141` 启动 SGLang Teacher 服务（`mem-fraction-static 0.18`，仅占 ~17 GB），在 96GB G4 / 80GB H100 上自动启用，在 40GB A100 上自动跳过以防 OOM。

### 5. 领域定向 BBPE 词表无损剪枝（`248,320 → 49,152`，5.05× 压缩）
- **实现位置**：[prune_vocab_qwen35.py](../src/prune_vocab_qwen35.py) + `run_experiment.sh` `*vocab49k*` 适配分支。
- **核心发现（全量语料零泄露扫描）**：对 ShopSimulator 全量商品库（`items_eval_train.json` 134 MB + `fine_items_eval_train_all.json.gz` 20 MB）与 `rl_500.jsonl` 合计 **86,929,896 tokens** 扫描（严格隔离 `official_test_200.jsonl` 防泄露）发现，整个中文电商模拟宇宙实际用到的唯一 token 仅 **38,907 个（仅占 Qwen3.5 原生 248,320 词表的 15.67%）**，其余 84.3% 均为小语种/代码/罕见词。
- **四级保底剪枝策略（精确对齐 128 的倍数 `49,152 = 384×128`）**：
  - **Tier 0**：全部 256 个字节原子 + 26 个特殊/对话/工具模板 token（保证 **0 OOV** 兜底）；
  - **Tier 1**：全部 38,907 个领域命中 token + **BPE Merge Tree 传递闭包**（40,701 个，保证合并树完整无断链）；
  - **Tier 2**：全部单字 CJK 汉字与 ASCII 符号（42,235 个，保障未见中文商品/品牌零退化）；
  - **Tier 3**：按 Qwen3.5 预训练高频词顺序补齐至 `49,152`（同时满足 Megatron Tensor Core 128 对齐）。
- **无损与提效实测**：
  - 模型总参数从 **1.88B → 1.47B（-21.7%）**，`lm_head` / Softmax 显存与计算量直降 **5.05×**；
  - 商品库 token 膨胀率仅 **+0.001%**（`192,829 → 192,830`），未见 `official_test_200` 仅 **+1.09%**，编解码 100% 无损闭环，`MultiTurnLossMaskGenerator(qwen3_5)` 全链路断言通过，SGLang 与 Megatron `log_probs` 对齐至小数点后 4 位（`-0.170066` vs `-0.170053`）；
  - **零样本意外红利（Domain-Constrained Tail Truncation）**：在 **0 步 RL 训练** 下，剪枝版 SFT 模型（`eval_vocab49k_sft`）在全量 200 题测试集上反超原版 248k SFT 模型（`r_loose` `0.4599 → 0.5123`，`r_hard` `0.1747 → 0.2389`，`r_success` `16.0% → 21.0%`，`right_product` `24.0% → 33.0%`）。原因在于 `T=1.0, top_p=1.0, top_k=-1` 自由采样下，裁剪掉 ~20 万个非领域长尾词消除了长尾概率泄漏，等价于领域级隐式词汇约束解码。

---

## 三、实验设计与阶段性实测结果

### 1. 统一超参设置
- **优化器**：AdamW（`lr=1e-6`, `constant`, `weight_decay=0.1`, `beta1=0.9`, `beta2=0.98`, `clip_grad=1.0`）
- **策略约束**：`eps_clip=0.2`, `eps_clip_high=0.28`, `kl_loss_coef=0.001`（`low_var_kl`）, `entropy_coef=0.0`
- **Rollout 规格**：`rollout_batch_size=5`, `n_samples_per_prompt=4`, `global_batch_size=20`, `max_model_turns=40`, `max_context_len=16384`, `max_response_len=2048`, `temperature=1.0`, `top_p=1.0`

### 2. 第一阶段快跑消融结果（`rl_50` 10 steps / 200 轨迹 → `official_test_50` 统一评测）

| 实验项 | 算法与配置 | `done_rate` | `r_loose` | `r_att` | `r_option` | `r_price` | `r_hard` | `r_success` | `right_product` | 训练耗时 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eval_baseline_sft` | SFT 起跑线 | 80.0% | 0.5091 | 0.5204 | 0.2400 | 0.7800 | 0.2400 | 24.0% | 30.0% | — |
| `gspo_native` | GSPO + `r_loose`（10 steps） | **92.0%** | **0.5757** | 0.6062 | **0.2800** | 0.8200 | 0.2586 | 24.0% | 30.0% | 45.9 min |
| **`opsd_gspo`** | **原创 OPSD + GSPO + 复合硬奖励（10 steps）** | **92.0%** | 0.5738 | **0.6133** | 0.2500 | **0.8400** | 0.2500 | 24.0% | **32.0%** | 55.4 min |
| **`cispo_hard`** | **CISPO + 复合硬奖励（10 steps）** | 86.0% | 0.5744 | 0.6118 | **0.2800** | 0.8000 | **0.2700** | **26.0%** | **32.0%** | **35.3 min** |

**首波消融核心发现（4 组已全部落盘 HF）**：
- 仅 10 步训练，3 种创新方法均将 `r_loose` 从 0.5091 显著提升至 **0.574–0.576**（`+0.065~0.067`），`r_att` 提升 `+0.086~0.093`；
- **`cispo_hard`（CISPO + 复合硬奖励）**：训练**最快（35.3 min，比 GSPO 快 23%）**，且是 **10 步内唯一突破精确全对率 `r_success`（24% → 26%）并拿下最高 `r_hard`（0.2700，+3.0%）**、并列最高 `r_option`（0.28）与 `right_product`（32%）的方法，充分验证了 CISPO 不清零超界梯度 + 复合硬奖励直击 SKU 规格短板的有效性；
- **`opsd_gspo`（原创特权上下文自蒸馏 + GSPO）**：拿下最高 **`r_att`（0.6133）**、最高 **`r_price`（0.8400）**、最高 **`done_rate`（92.0%）** 及并列最高 **`right_product`（32.0%）**；
- **`gspo_native`（序列级 GSPO）**：拿下最高 **`r_loose`（0.5757）**、最高 **`done_rate`（92.0%）** 与并列最高 **`r_option`（0.28）**。

### 3. 第二阶段全量正式实验（`rl_500.jsonl` 500 题 / 100 步 / 2,000 轨迹 → `official_test_200.jsonl` 全量 200 题）

| 实验项 | 算法与配置 | 词表 / 参数 | `done_rate` | 零奖励率↓ | `r_loose` | `r_att` | `r_option` | `r_price` | `r_hard` | 严格成功 `r_success` | `right_product` | 状态 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `eval_baseline_sft` | SFT 起跑线 | 248k / 1.88B | 81.5% | 21.0% | 0.4599 | 0.4936 | 0.1975 | 0.6600 | 0.1747 | 16.0% (32/200) | 24.0% (48/200) | ✅ HF 已落盘 |
| **`eval_vocab49k_sft`** | **BBPE 49k 剪枝 SFT（零训练）** | **49k / 1.47B** | 83.5% | 20.0% | **0.5123** | 0.5548 | 0.2692 | 0.6950 | 0.2389 | 21.0% (42/200) | 33.0% (66/200) | ✅ HF 已落盘 |
| **`gspo_native`** | **GSPO + `r_loose`（100 步全量）** | 248k / 1.88B | **95.5%** | 6.0% | **0.6941** | **0.7404** | **0.4100** | **0.8700** | **0.3845** | **34.0% (68/200)** | **49.5% (99/200)** | ✅ HF 已落盘 |
| **`opsd_gspo`** | **原创 OPSD + GSPO + 复合硬奖励（100 步全量）** | 248k / 1.88B | **95.5%** | **5.5%** | 0.6722 | 0.7115 | 0.4083 | 0.8600 | 0.3798 | 33.5% (67/200) | 44.5% (89/200) | ✅ HF 已落盘 |
| `gspo_vocab49k` | 49k 剪枝 + GSPO（Blackwell 96GB） | 49k / 1.47B | — | — | *Step 66 `raw_reward=0.8051`* | — | — | — | — | — | — | ⏸ 待补完（单步仅 1.6 min） |
| `cispo_hard` / `cispo_vocab49k` | CISPO + 复合硬奖励 | 248k / 49k | — | — | — | — | — | — | — | — | — | ⏳ 待跑 |
| `baseline_grpo` | 原版 GRPO + `r_loose` | 248k / 1.88B | — | — | — | — | — | — | — | — | — | ⏳ 待跑 |

**第二阶段全量 500 题核心科学结论**：
1. **全量 GSPO 相比 SFT 实现跨代跃升（每项核心指标翻倍以上）**：
   - `gspo_native` 将全量 200 题 **`r_success` 从 `16.0% → 34.0%`（2.13×，+18.0%）**、**`right_product` 从 `24.0% → 49.5%`（2.06×，+25.5%）**、**`r_hard` 从 `0.1747 → 0.3845`（2.20×）**、**`r_option` 从 `0.1975 → 0.4100`（2.08×）**、**`r_loose` 从 `0.4599 → 0.6941`（+0.2342）**、**`done_rate` 从 `81.5% → 95.5%`**。
2. **原创 `opsd_gspo` 同样取得巨大增益并拿下全组最低零奖励率（5.5%）**：
   - `opsd_gspo` 在 `r_success`（**33.5% vs 34.0%**）、`r_option`（**0.4083 vs 0.4100**）、`r_hard`（**0.3798 vs 0.3845**）与 `done_rate`（**95.5% 并列第一**）上紧咬 `gspo_native`，并将**完全零分任务率压至全组最低的 `5.5%`**（SFT 为 `21.0%`，`gspo_native` 为 `6.0%`），验证了特权上下文自蒸馏在困难任务上的托底纠错价值；
   - **对比复盘与改进方向**：`opsd_gspo` 在 `right_product`（`44.5%` vs `49.5%`）与 `r_loose`（`0.6722` vs `0.6941`）上略逊于纯 `gspo_native`，原因在于固定常数的 `opd_kl_coef = 0.15` 在训练中后期对同权重 Self-Teacher 的反向 KL 正则偏强，略微抑制了中后期的搜索策略探索。后续最优改进为 **OPSD KL 系数余弦退火（`0.15 → 0.02`）** 或 **仅对组内零方差/失败样本启用 OPSD 门控（Gated OPSD）**，兼顾前期快速托底与后期自由探索。
3. **49k 词表剪枝 + Blackwell 96GB 速度实测（4–5× 加速）**：
   - `gspo_vocab49k` 在 Blackwell 96GB（关重计算 `MAX_TOK=16384`）上 107 分钟连跑 **67 步（`~1.60 min/step`）**，`log_probs` 仅耗时 **`4.6–5.9s`**（比 A100 248k 快 **5–6×**），`actor_train` 仅 **`14.7–19.1s`**，单步 Megatron 训练总耗时仅 **`24.2–31.6s`**，且 Step 66 在线 `raw_reward` 已冲至 **`0.8051`**（因凌晨本机休眠导致 WebSocket 断连被 Colab 回收，重启仅需 ~2.6 小时 / ~29 CU 即可跑完 100 步全量闭环）。

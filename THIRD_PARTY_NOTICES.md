# 第三方组件协议与致谢声明（Third-Party Notices）

本项目为**独立原创的算法插件、词表剪枝工具与 Colab 单卡实验调度框架**，**不包含也不再分发**任何上游仓库源码、商品数据集或模型权重文件。

用户在本地或云端运行本仓库脚本时，需自行遵循以下上游项目的开源协议与使用条款：

1. **Piucente/pi-slime-shopsimulator**
   - 仓库：<https://github.com/Piucente/pi-slime-shopsimulator>
   - 协议：MIT License（Copyright (c) 2026 Pi Slime ShopSimulator contributors）
   - 说明：本项目以零源码侵入（通过 `PYTHONPATH` 插件加载与临时参数包装）的方式调用其 `examples/ShopSimulator` 评测/训练入口，并使用其公开的 SFT 起跑线模型 [`mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep`](https://huggingface.co/mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep)。

2. **THUDM/slime**
   - 仓库：<https://github.com/THUDM/slime>
   - 协议：Apache License 2.0
   - 说明：提供 Megatron-LM + SGLang Colocate RL 训练框架及原生 GSPO / CISPO / OPD 管线接口。

3. **ShopAgent-Team/ShopSimulator**
   - 仓库：<https://github.com/ShopAgent-Team/ShopSimulator>
   - 说明：中文电商多轮交互模拟环境与商品数据；本仓库不分发其源码或数据，用户需按上游协议自行获取。

4. **Qwen3.5-2B**
   - 模型：<https://huggingface.co/Qwen>
   - 协议：Apache License 2.0

5. **核心基础设施依赖**
   - **Megatron-LM**（NVIDIA，BSD-3-Clause）
   - **SGLang**（Apache License 2.0）
   - **Pi Coding Agent (`@earendil-works/pi-coding-agent`)**（遵循其上游开源协议）

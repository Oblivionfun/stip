# 基于大模型微调的人类驾驶偏好自适应路径决策系统

> **项目状态**: 🚧 开发中
> **当前进度**: ✅ Persona建模完成 | ✅ 训练数据完成 | 🔄 模型训练准备中
> **最后更新**: 2025-12-04

---

## 📚 项目简介

本项目基于**GATSim**（Generative Agent Transport Simulation）架构和**Centaur**（Nature论文）的高效微调技术，构建一个能够根据人类驾驶偏好自适应做路径决策的智能体系统。

### 核心特性

- ✅ **数据驱动**：基于CN/UK/US/AU四国问卷数据（3000+样本）
- ✅ **偏好建模**：提取8个可解释的偏好因子
- ✅ **GATSim兼容**：输出格式兼容GATSim交通仿真系统
- ✅ **训练数据**：10,000+个高质量训练样本（含thinking思考过程）
- ✅ **高效微调**：采用unsloth + 4bit + LoRA技术
- ⏳ **自适应决策**：支持冷启动和在线偏好更新

---

## 🗂️ 项目结构

```
stip/
├── data/                      # 原始问卷数据
│   ├── CN_dataset.xlsx
│   ├── UK_dataset.xlsx
│   ├── US_dataset.xlsx
│   └── AU_dataset.xlsx (待添加)
│
├── configs/                   # 配置文件
│   ├── preference_config.yaml
│   └── scenario_config.yaml
│
├── src/                       # 源代码
│   ├── preference_modeling/   # Persona建模模块
│   │   ├── data_loader.py
│   │   ├── factor_analysis.py
│   │   ├── persona_generator.py
│   │   └── persona_clustering.py
│   ├── data_construction/     # 训练数据构造模块
│   │   ├── scenario_generator.py
│   │   ├── decision_simulator.py
│   │   └── sample_builder.py
│   ├── training/              # 模型训练和评估
│   │   ├── dataset.py
│   │   ├── sft_trainer.py
│   │   └── evaluator.py
│   └── utils/                 # 通用工具
│
├── outputs/                   # 输出数据
│   ├── cleaned_survey_data.csv
│   ├── preference_factors.csv
│   ├── personas.json
│   ├── persona_types.json
│   ├── scenarios.json
│   ├── decisions.json
│   ├── train_samples.jsonl
│   ├── validation_samples.jsonl
│   └── logs/
│
├── checkpoints/               # 模型权重
│   └── sft_model/            # 微调后的模型
│
├── run_persona_modeling.py   # Persona建模一键运行
├── run_data_construction.py  # 训练数据构造一键运行
├── run_model_training.py     # 模型训练一键运行
├── requirements.txt          # Python依赖
│
└── docs/                      # 文档
    ├── QUICKSTART.md         # 完整运行指南 ⭐
    ├── OPTIMIZATION_GUIDE.md # 优化说明
    ├── THINKING_FEATURE.md   # Thinking功能说明
    └── GPU_MEMORY_GUIDE.md   # GPU显存优化
```

---

## 🚀 快速开始

**详细步骤请查看 → [docs/QUICKSTART.md](docs/QUICKSTART.md) ⭐**

### 1. 环境准备

```bash
# 安装基础依赖
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml openpyxl scipy

# 安装深度学习依赖（模型训练需要）
pip install torch transformers accelerate datasets peft bitsandbytes
```

### 2. Persona建模 (约10秒)

```bash
python run_persona_modeling.py
```

**输出**: 1002个Persona对象 + 8个偏好因子 + 6种驾驶员类型

### 3. 训练数据构造 (约5秒)

```bash
python run_data_construction.py
```

**输出**: 9018个训练样本 + 1002个验证样本（含thinking思考过程）

### 4. 模型训练 (数小时)

```bash
python run_model_training.py
```

**输出**: 微调后的Qwen3模型 (checkpoints/sft_model/)

### 5. 模型评估

```bash
python src/training/evaluator.py
```

**评估指标**: 路径选择准确率、改道F1分数、Perplexity
- `outputs/preference_factors.csv` - 8个偏好因子得分
- `outputs/personas.json` - GATSim格式Persona对象（1002个）
- `outputs/persona_types.json` - 聚类类型标签
- 可视化图表（因子载荷热力图、聚类可视化等）

---

## 📊 主要输出文件

### Persona建模输出

| 文件名 | 大小 | 描述 |
|--------|------|------|
| `personas.json` | 1.5MB | 1002个GATSim格式Persona对象 |
| `preference_factors.csv` | 166KB | 8个偏好因子得分 |
| `persona_types.json` | 88KB | 6种驾驶员类型标签 |
| `persona_clustering.png` | 435KB | PCA聚类可视化 |

### 训练数据输出

| 文件名 | 大小 | 描述 |
|--------|------|------|
| `train_samples.jsonl` | 15.5MB | 9018个训练样本 |
| `validation_samples.jsonl` | 1.7MB | 1002个验证样本 |
| `scenarios.json` | ~20MB | 10020个场景 |

---

## 🎯 Response格式示例

模型输出包含完整的思考过程：

```json
{
  "thinking": "Let me analyze the current situation:\n- Route A (usual): 35min (delay: 10min, uncertainty: moderate)\n- Route B (alternative): 27min (delay: 0min, uncertainty: low)\n\nConsidering my preferences:\n- I prioritize time efficiency\n- I prefer predictable routes\n\nThe delay (10min) is within my tolerance threshold.",

  "reflection": "Based on my analysis, Route B is 8 minutes faster and has lower uncertainty. Although I prefer familiar routes, the time savings and predictability make Route B the better choice in this situation.",

  "plan": "update path: Side_St, Local_Road_2",
  "concepts": []
}
```

### 6个Persona类型

| Cluster | 类型标签 | 占比 | 主要特征 |
|---------|---------|------|---------|
| 0 | CAUTIOUS_FAMILIAR_INFO_DEPENDENT | 27.0% | 谨慎、偏好熟悉路线、依赖信息 |
| 1 | TIME_SENSITIVE_FLEXIBLE_REROUTER | 19.6% | 时间敏感、灵活改道 |
| 2 | RISK_TOLERANT_INDEPENDENT_NAVIGATOR | 6.8% | 风险容忍、独立导航 |
| 3 | BALANCED_RATIONAL_DRIVER | 8.3% | 平衡理性型 |
| 4 | CONSERVATIVE_PATIENT_FOLLOWER | 21.3% | 保守、耐心、跟随型 |
| 5 | PROACTIVE_TECH_SAVVY_OPTIMIZER | 17.1% | 主动、精通技术、优化型 |

---

## 📈 进度追踪

```
Persona建模        [████████████] 100% ✅ (约10秒)
训练数据构造       [████████████] 100% ✅ (约5秒)
模型训练           [░░░░░░░░░░░░]   0% 🔄 (数小时)
模型评估           [░░░░░░░░░░░░]   0% ⏳

总体进度：         [████████░░░░]  67%
```

---

## 🛠️ 技术栈

### 数据处理
- pandas, numpy - 数据处理
- scikit-learn - 因子分析、聚类
- matplotlib, seaborn - 可视化

### 模型训练
- PyTorch - 深度学习框架
- Transformers - 模型加载和推理
- PEFT - LoRA参数高效微调
- bitsandbytes - 4bit量化
- unsloth (可选) - 训练加速

### 训练配置
- 基座模型: Qwen3 (~15GB)
- LoRA rank: 16, alpha: 32
- 量化: 4bit
- 优化器: AdamW 8bit
- 学习率: 2e-4
- Batch size: 2 × 4 (梯度累积)
- Epochs: 3

---

## 📖 文档

- **[QUICKSTART.md](docs/QUICKSTART.md)** ⭐ - 完整运行指南（必读）
- **[OPTIMIZATION_GUIDE.md](docs/OPTIMIZATION_GUIDE.md)** - 时间抽象化优化说明
- **[THINKING_FEATURE.md](docs/THINKING_FEATURE.md)** - Thinking思考过程功能说明
- **[GPU_MEMORY_GUIDE.md](docs/GPU_MEMORY_GUIDE.md)** - GPU显存优化指南

---

## 🎓 核心创新

1. **时间信息抽象化** - 使用"morning rush hour"而非"08:42"，避免过拟合
2. **Thinking思考过程** - 模型输出包含详细的分析推理步骤
3. **偏好因子建模** - 8个可解释的偏好维度，支持个性化决策
4. **完善评估体系** - 路径准确率 + 改道F1 + Perplexity多维度评估

---

## 🤝 参考项目

- **GATSim** - Persona架构和决策prompt模板
- **Centaur (Nature 2025)** - unsloth + 4bit + LoRA高效微调
- **PNAS 100K Everyday Choices** - 大规模决策数据分析方法

---

**开始使用：查看 [docs/QUICKSTART.md](docs/QUICKSTART.md)** 🚀**核心模块**：
1. RouteDecisionAgent（智能体封装）
2. ActionParser（解析plan字段）
3. PreferenceAdapter（偏好冷启动和在线更新）

**API示例**：
```python
agent = RouteDecisionAgent(model_path="checkpoints/sft_model")
action = agent.decide(scenario, persona)
# 输出: {"route": "B", "reroute": True, "reasoning": "..."}
```

---

## 📈 项目进度

```
阶段1：Persona建模     [████████████] 100% ✅
阶段2：数据构造        [████████████] 100% ✅
阶段3：模型微调        [░░░░░░░░░░░░]   0% 🔄
阶段4：智能体API       [░░░░░░░░░░░░]   0% ⏳

总体进度：            [██████░░░░░░]  50%
```

**已完成**：
- ✅ 阶段1：Persona建模（约10秒运行时间）
- ✅ 阶段2：训练数据构造（约5秒运行时间）

**预计完成时间**：
- 阶段3：5-7天（包括模型训练）
- 阶段4：2-3天

---

## 🛠️ 技术栈

### 已使用
- **数据处理**：pandas, numpy
- **机器学习**：scikit-learn（因子分析、K-Means）
- **可视化**：matplotlib, seaborn
- **配置管理**：pyyaml

### 即将使用（阶段2-4）
- **深度学习**：PyTorch, Transformers
- **高效微调**：unsloth, peft, trl
- **模板引擎**：jinja2
- **基座模型**：Qwen2-7B-Instruct

---

## 📖 文档

详细文档请查看：

- 📋 [任务路线图](TASK_ROADMAP.md) - 原始技术路线
- 🚀 [优化方案](OPTIMIZED_ROADMAP.md) - 基于参考项目的优化
- 🗺️ [问卷映射](QUESTIONNAIRE_MAPPING.md) - 67个字段的完整映射（86.6%覆盖）
- 📊 [进度报告](PROGRESS_REPORT.md) - 详细进度和交付物

---

## 🤝 参考项目

本项目借鉴了以下优秀工作：

1. **GATSim** ([GitHub](https://github.com/tsinghua-fib-lab/GATSim))
   - Persona架构设计
   - 决策输出格式（reflection + plan + concepts）
   - Prompt模板设计

2. **Centaur** (Nature 2025)
   - unsloth高效微调技术
   - Completion-only训练方法
   - 4bit量化 + LoRA

3. **PNAS 100K Choices**
   - 大规模决策数据处理
   - 向量化 + 聚类pipeline

---

## ⚠️ 已知限制

1. **数据范围**：目前仅处理CN数据（1002样本），UK/US/AU需要Qualtrics列名映射
2. **因子旋转**：sklearn的FactorAnalysis不支持varimax，未来可考虑factor_analyzer库
3. **训练数据**：阶段2需要模拟生成，可选用GPT-4作为teacher模型提升质量

---

## 📝 TODO

- [ ] 实现阶段2：训练数据构造
  - [ ] 场景生成器
  - [ ] 决策模拟器
  - [ ] LLM样本构造器
- [ ] 实现阶段3：模型微调
  - [ ] unsloth + LoRA训练脚本
  - [ ] DPO偏好对齐（可选）
- [ ] 实现阶段4：智能体API
  - [ ] 推理接口封装
  - [ ] 动作解析器
  - [ ] 偏好适配器
- [ ] UK/US/AU数据处理
- [ ] 离线评估脚本
- [ ] 可视化界面（可选）

---

## 📧 联系方式

如有问题或建议，请查看项目日志：`outputs/logs/`

---

**License**: Apache 2.0
**Last Updated**: 2025-12-04

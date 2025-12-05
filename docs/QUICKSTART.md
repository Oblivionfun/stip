# 完整运行指南

> **目标**: 从问卷数据到可用的路径决策智能体
> **预计总时间**: 数据处理15秒 + 模型训练数小时

---

## 📋 前置准备

### 1. 环境要求

**硬件**：
- ✅ GPU: 至少16GB显存（推荐32GB）
- ✅ 内存: 至少32GB RAM
- ✅ 存储: 至少50GB可用空间

**软件**：
- Python 3.8+
- CUDA 11.7+ (用于GPU训练)

### 2. 检查数据文件

确保以下数据文件存在：
```bash
ls -lh data/
# 应该看到：
# CN_dataset.xlsx  (约1-2MB)
# UK_dataset.xlsx  (可选)
# US_dataset.xlsx  (可选)
```

### 3. 检查模型文件

确保Qwen3模型已下载：
```bash
ls -lh model/models/
# 应该看到：
# config.json
# model-00001-of-00005.safetensors (~3.8GB)
# model-00002-of-00005.safetensors (~3.8GB)
# ... (总计约15-16GB)
# tokenizer.json
```

---

## 🚀 完整运行流程

### 步骤1：安装依赖 (约2-5分钟)

```bash
# 基础依赖
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml openpyxl scipy

# 深度学习依赖
pip install torch transformers accelerate datasets peft bitsandbytes

# 可选：安装unsloth加速训练（需要特定GPU和CUDA版本）
# pip install unsloth
```

**验证安装**：
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

### 步骤2：Persona建模 (约10秒)

**功能**: 从问卷数据生成1002个驾驶员Persona

**运行**：
```bash
python run_persona_modeling.py
```

**输出文件**：
```
outputs/
├── cleaned_survey_data.csv      (299KB)   - 清洗后的问卷数据
├── preference_factors.csv       (166KB)   - 8个偏好因子得分
├── personas.json                (1.5MB)   - GATSim格式Persona对象
├── persona_types.json           (88KB)    - 6种驾驶员类型标签
├── factor_loadings_heatmap.png  (135KB)   - 因子载荷可视化
└── persona_clustering.png       (435KB)   - 聚类可视化
```

**验证**：
```bash
# 检查生成的Persona数量
python -c "import json; personas=json.load(open('outputs/personas.json')); print(f'Generated {len(personas)} personas')"

# 查看一个Persona示例
head -50 outputs/personas.json
```

**预期结果**：
- ✅ 1002个Persona对象
- ✅ 6种驾驶员类型（谨慎型、时间敏感型等）
- ✅ 每个Persona包含8个偏好因子

---

### 步骤3：训练数据构造 (约5秒)

**功能**: 为每个Persona生成10个场景，构造10,000个训练样本

**运行**：
```bash
python run_data_construction.py
```

**输出文件**：
```
outputs/
├── scenarios.json            (~20MB)  - 10,020个路径选择场景
├── decisions.json            (~3MB)   - 10,020个决策结果
├── train_samples.jsonl       (15.5MB) - 9,018个训练样本
├── validation_samples.jsonl  (1.7MB)  - 1,002个验证样本
└── *_statistics.json                  - 统计信息
```

**验证**：
```bash
# 检查样本数量
wc -l outputs/train_samples.jsonl outputs/validation_samples.jsonl

# 查看一个训练样本
head -1 outputs/train_samples.jsonl | python -m json.tool
```

**样本格式示例**：
```json
{
  "id": "CN_0001_S01",
  "prompt": "You play the role of the person:\nName: User_0001 | Age: 35...\nTime period: morning rush hour\n...",
  "response": {
    "thinking": "Let me analyze the current situation:\n- Route A (usual): 35min (delay: 10min)\n- Route B (alternative): 27min (delay: 0min)\n...",
    "reflection": "Based on my analysis, Route B is faster and more predictable...",
    "plan": "update path: Side_St, Local_Road_2",
    "concepts": []
  }
}
```

**预期结果**：
- ✅ 9,018个训练样本 + 1,002个验证样本
- ✅ 改道率约68%
- ✅ 包含thinking（思考过程）、reflection（推理）、plan（决策）

---

### 步骤4：模型训练 (数小时，取决于GPU性能)

**功能**: 使用LoRA微调Qwen3模型

**重要提示**：
- ⚠️ 训练需要数小时，建议在tmux/screen中运行
- ⚠️ 确保GPU显存足够（至少16GB，推荐32GB）
- ⚠️ 训练过程会占用GPU，确保没有其他任务

**运行**：
```bash
# 方式1：直接运行
python run_model_training.py

# 方式2：在tmux中运行（推荐）
tmux new -s training
python run_model_training.py
# 按Ctrl+B然后按D detach
# 稍后用 tmux attach -t training 重连

# 方式3：后台运行
nohup python run_model_training.py > training.log 2>&1 &
# 查看日志：tail -f training.log
```

**监控GPU使用**（另开一个终端）：
```bash
watch -n 1 nvidia-smi
```

**训练配置** (在 `configs/training_config.yaml` 中)：
- Epochs: 3
- Batch size: 2 × 4 (梯度累积) = 8有效batch
- LoRA rank: 16
- Learning rate: 2e-4
- 优化器: AdamW 8bit
- 量化: 4bit

**输出文件**：
```
checkpoints/sft_model/
├── adapter_model.safetensors  (~100MB)  - LoRA权重
├── adapter_config.json
├── tokenizer.json
├── config.json
└── training_args.bin
```

**训练日志**：
```
outputs/logs/sft_trainer.log
```

**预期训练时间**：
- A100: 2-3小时
- V100: 4-6小时
- RTX 3090: 3-5小时
- RTX 4090: 2-4小时

**验证**：
```bash
# 检查checkpoint是否生成
ls -lh checkpoints/sft_model/

# 查看训练日志
tail -100 outputs/logs/sft_trainer.log
```

---

### 步骤5：模型评估 (约10-30分钟)

**功能**: 评估模型的决策准确率、改道F1、Perplexity

**运行**：
```bash
python src/training/evaluator.py
```

**评估指标**：
1. **路径选择准确率** - 目标≥75%
2. **改道决策F1分数** - 目标≥0.80
3. **Perplexity** - 越低越好（<10为优秀）

**输出**：
```
================================================================================
模型评估结果
================================================================================

决策准确率指标:
  - 路径选择准确率: 78.50%
  - 改道决策F1分数: 0.8234
    - Precision: 0.8456
    - Recall: 0.8021

语言模型指标:
  - Perplexity: 4.23

详细统计:
  - 总样本数: 1002
  - 正确路径选择: 787
  - 改道TP: 552
  - 改道FP: 101
  - 改道FN: 136
================================================================================
```

**结果文件**：
```
outputs/evaluation_results.json
```

---

## 📊 运行进度追踪

### 快速检查进度

```bash
# 检查每个步骤的输出文件
ls -lh outputs/personas.json              # 步骤2完成
ls -lh outputs/train_samples.jsonl       # 步骤3完成
ls -lh checkpoints/sft_model/             # 步骤4完成
ls -lh outputs/evaluation_results.json   # 步骤5完成
```

### 详细日志位置

```bash
outputs/logs/
├── data_loader.log           - 数据加载日志
├── factor_analysis.log       - 因子分析日志
├── persona_generator.log     - Persona生成日志
├── persona_clustering.log    - 聚类日志
├── scenario_generator.log    - 场景生成日志
├── decision_simulator.log    - 决策模拟日志
├── sample_builder.log        - 样本构造日志
├── sft_trainer.log           - 模型训练日志（最重要）
└── evaluator.log             - 评估日志
```

---

## 🐛 常见问题排查

### 问题1：显存不足 (OOM)

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
```yaml
# 修改 configs/training_config.yaml

# 方案A：减小batch size
training:
  per_device_train_batch_size: 1  # 从2改为1
  gradient_accumulation_steps: 8  # 从4改为8

# 方案B：减小序列长度
model:
  max_seq_length: 1536  # 从2048改为1536

# 方案C：减小LoRA rank
lora:
  r: 8  # 从16改为8
```

---

### 问题2：训练速度很慢

**检查GPU使用**：
```bash
nvidia-smi
# GPU利用率应该>80%
```

**可能原因**：
- DataLoader workers不足
- Batch size太小
- 未启用混合精度

**解决方案**：
```yaml
# configs/training_config.yaml
training:
  dataloader_num_workers: 4  # 增加workers
  bf16: true                 # 启用混合精度
```

---

### 问题3：模型不收敛

**症状**：
- 训练loss不下降
- 验证loss持续上升
- 评估准确率很低

**检查**：
```bash
# 查看训练日志
grep "loss" outputs/logs/sft_trainer.log

# 检查学习率
grep "learning_rate" outputs/logs/sft_trainer.log
```

**解决方案**：
1. 降低学习率：`2e-4 -> 1e-4`
2. 增加warmup：`100 -> 200`
3. 检查数据质量：确认training data格式正确

---

### 问题4：评估结果不理想

**路径选择准确率<70%**：
- 检查训练数据质量
- 增加训练epochs
- 调整LoRA参数

**改道F1分数<0.70**：
- 数据不平衡（改道vs不改道）
- 增加数据多样性
- 调整决策阈值

---

## 🎯 性能优化建议

### 1. 数据预处理

**跳过已完成的步骤**：
```bash
# 如果personas.json已存在，直接跳到数据构造
python run_data_construction.py

# 如果train_samples.jsonl已存在，直接开始训练
python run_model_training.py
```

---

### 2. 训练加速

**使用更少的验证样本**：
```python
# src/training/evaluator.py
results = evaluator.run_evaluation(num_samples=100)  # 仅评估100个样本
```

**减少评估频率**：
```yaml
# configs/training_config.yaml
training:
  eval_steps: 1000  # 从500改为1000
```

---

### 3. 快速测试

**使用更少的数据快速验证流程**：
```yaml
# configs/training_config.yaml
data:
  max_samples: 1000  # 仅用1000个样本训练

training:
  num_train_epochs: 1  # 仅训练1个epoch
```

---

## 📁 文件结构总览

```
/root/stip/
├── run_persona_modeling.py      # 🔵 步骤2：Persona建模
├── run_data_construction.py     # 🔵 步骤3：训练数据构造
├── run_model_training.py        # 🔵 步骤4：模型训练
│
├── data/                         # 原始问卷数据
│   └── CN_dataset.xlsx
│
├── model/models/                 # Qwen3模型
│   ├── config.json
│   └── model-*.safetensors
│
├── configs/                      # 配置文件
│   ├── preference_config.yaml   - Persona建模配置
│   ├── scenario_config.yaml     - 场景生成配置
│   └── training_config.yaml     - 训练配置
│
├── src/                          # 源代码
│   ├── preference_modeling/     - Persona建模模块
│   ├── data_construction/       - 数据构造模块
│   ├── training/                - 训练和评估模块
│   └── utils/                   - 工具函数
│
├── outputs/                      # 所有输出文件
│   ├── personas.json            - Persona对象
│   ├── train_samples.jsonl      - 训练数据
│   ├── evaluation_results.json  - 评估结果
│   └── logs/                    - 日志文件
│
├── checkpoints/                  # 模型权重
│   └── sft_model/               - 微调后的模型
│
└── docs/                         # 文档
    ├── QUICKSTART.md            - 本文件
    ├── OPTIMIZATION_GUIDE.md    - 优化说明
    └── THINKING_FEATURE.md      - Thinking功能说明
```

---

## ✅ 运行检查清单

完成每个步骤后，在对应的框中打勾：

- [ ] **环境准备**
  - [ ] Python依赖已安装
  - [ ] GPU可用且显存足够
  - [ ] 问卷数据文件存在
  - [ ] Qwen3模型文件存在

- [ ] **步骤2：Persona建模** (约10秒)
  - [ ] 运行 `python run_persona_modeling.py`
  - [ ] 生成了 `outputs/personas.json` (1002个)
  - [ ] 生成了可视化图表

- [ ] **步骤3：训练数据构造** (约5秒)
  - [ ] 运行 `python run_data_construction.py`
  - [ ] 生成了 `outputs/train_samples.jsonl` (9018个)
  - [ ] 生成了 `outputs/validation_samples.jsonl` (1002个)
  - [ ] 样本格式包含thinking、reflection、plan

- [ ] **步骤4：模型训练** (数小时)
  - [ ] 运行 `python run_model_training.py`
  - [ ] 训练loss稳定下降
  - [ ] 生成了 `checkpoints/sft_model/`
  - [ ] 包含adapter权重和配置文件

- [ ] **步骤5：模型评估** (约10-30分钟)
  - [ ] 运行 `python src/training/evaluator.py`
  - [ ] 路径选择准确率 ≥ 70%
  - [ ] 改道F1分数 ≥ 0.70
  - [ ] 生成了 `outputs/evaluation_results.json`

---

## 🎓 下一步

模型训练完成后，可以：

1. **部署推理服务**
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM

   tokenizer = AutoTokenizer.from_pretrained("checkpoints/sft_model")
   model = AutoModelForCausalLM.from_pretrained("checkpoints/sft_model")

   # 推理
   prompt = "You play the role..."
   response = model.generate(...)
   ```

2. **集成到应用**
   - 构建API服务
   - 接入导航系统
   - 实时路径决策

3. **继续优化**
   - 收集真实反馈数据
   - 在线学习更新
   - A/B测试效果

---

## 📞 获取帮助

**日志查看**：
```bash
# 查看最近的错误
tail -100 outputs/logs/sft_trainer.log | grep -i error

# 实时监控训练
tail -f outputs/logs/sft_trainer.log
```

**常用命令**：
```bash
# 检查GPU使用
nvidia-smi

# 检查磁盘空间
df -h

# 检查进程
ps aux | grep python

# 杀死训练进程（谨慎使用）
pkill -f run_model_training
```

---

**祝您运行顺利！如有问题请查看日志文件或文档** 🚀

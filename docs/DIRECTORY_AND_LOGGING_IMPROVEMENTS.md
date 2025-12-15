# 更新日志和目录结构改进

## 📁 新的目录结构

```
outputs/
├── 1_persona_modeling/          # Stage 1: Persona建模
│   ├── personas.json
│   ├── preference_factors.csv
│   └── ...
│
├── 2_data_construction/         # Stage 2: 训练数据构造
│   ├── train_samples.jsonl
│   ├── validation_samples.jsonl
│   └── ...
│
├── 3_training/                  # Stage 3: 模型训练
│   └── runs/                    # TensorBoard日志（按时间戳）
│       ├── run_20241210_210543/
│       └── run_20241211_083421/
│
├── 4_evaluation/                # Stage 4: 模型评估
│   ├── baseline/                # Baseline模型评估结果
│   └── finetuned/               # 微调模型评估结果
│
└── logs/                        # 所有运行日志（按时间戳）
    ├── training_20241210_210543.log
    ├── evaluator_20241210_220134.log
    └── sft_trainer_20241210_210545.log
```

## ⏰ 时间戳功能

### 自动时间戳
所有日志文件现在都会自动添加时间戳，格式为：`YYYYMMDD_HHMMSS`

### 训练运行
每次训练会自动创建新的运行目录：
```bash
python run_model_training.py -y

# 自动创建：
# - outputs/3_training/runs/run_20241210_210543/  (TensorBoard日志)
# - outputs/logs/sft_trainer_20241210_210545.log  (训练日志)
```

### 评估结果
评估结果可以指定输出路径，建议使用描述性名称：
```bash
# Baseline评估
python src/training/evaluator.py \
  --model-path model/models \
  --output outputs/4_evaluation/baseline/baseline_20241210.json

# 微调模型评估
python src/training/evaluator.py \
  --model-path checkpoints/sft_model/checkpoint-2000 \
  --output outputs/4_evaluation/finetuned/checkpoint-2000_20241210.json
```

## 🔧 配置更新

### TensorBoard日志
现在会自动记录到带时间戳的目录：
```yaml
# configs/training_config.yaml
training:
  report_to: "tensorboard"  # 已启用
  logging_dir: "outputs/3_training/runs/{RUN_NAME}"  # 自动生成
```

### 查看TensorBoard
```bash
# 查看所有训练运行
tensorboard --logdir outputs/3_training/runs --port 6006 --bind_all

# 查看特定运行
tensorboard --logdir outputs/3_training/runs/run_20241210_210543 --port 6006
```

## 📊 路径工具使用

### 在代码中使用
```python
from src.utils.path_utils import (
    get_timestamp,
    get_log_path,
    get_output_path,
    get_training_run_dir,
    OutputPaths
)

# 生成时间戳
timestamp = get_timestamp()  # "20241210_210543"

# 获取日志路径（自动添加时间戳）
log_path = get_log_path('my_module')
# outputs/logs/my_module_20241210_210543.log

# 获取输出路径
output_path = get_output_path('results.json', '4_evaluation/finetuned')
# outputs/4_evaluation/finetuned/results_20241210_210543.json

# 获取训练运行目录
run_dir = get_training_run_dir()
# outputs/3_training/runs/run_20241210_210543/

# 使用路径常量
eval_dir = OutputPaths.EVAL_FINETUNED
# Path("outputs/4_evaluation/finetuned")
```

## 🚀 使用示例

### 完整训练流程
```bash
# 1. 运行训练（自动创建时间戳目录和日志）
python run_model_training.py -y

# 输出：
# - TensorBoard: outputs/3_training/runs/run_20241210_210543/
# - 日志: outputs/logs/sft_trainer_20241210_210545.log

# 2. 实时查看训练进度（TensorBoard）
tensorboard --logdir outputs/3_training/runs --port 6006 --bind_all

# 3. 训练完成后评估
python src/training/evaluator.py \
  --model-path checkpoints/sft_model/checkpoint-2000 \
  --num-samples 100 \
  --output outputs/4_evaluation/finetuned/checkpoint-2000_results.json

# 输出：
# - 评估结果: outputs/4_evaluation/finetuned/checkpoint-2000_results.json
# - 日志: outputs/logs/evaluator_20241210_220134.log

# 4. 对比baseline
python analyze_evaluation.py compare
```

### 查看日志
```bash
# 查看最新的训练日志
ls -t outputs/logs/sft_trainer_*.log | head -1 | xargs tail -f

# 查看所有评估日志
ls outputs/logs/evaluator_*.log
```

## ✨ 优点

### 1. 清晰分类
- 每个阶段的输出都在独立目录
- 一目了然知道文件属于哪个阶段

### 2. 自动时间戳
- 不需要手动修改输出路径
- 每次运行都有唯一的时间戳标识
- 方便追溯历史运行

### 3. 方便对比
- 可以轻松对比不同时间的训练结果
- TensorBoard支持同时加载多个运行

### 4. 不会覆盖
- 时间戳确保每次运行都是独立的
- 历史数据不会被意外覆盖

## 📝 配置迁移

### 旧路径 → 新路径
```
outputs/*.json                    → outputs/4_evaluation/*.json
outputs/*.log                     → outputs/logs/*.log
outputs/personas.json             → outputs/1_persona_modeling/personas.json
outputs/train_samples.jsonl       → outputs/2_data_construction/train_samples.jsonl
checkpoints/runs/                 → outputs/3_training/runs/
```

### 数据文件引用
配置文件中的路径已自动更新：
```yaml
data:
  train_file: "outputs/2_data_construction/train_samples.jsonl"
  val_file: "outputs/2_data_construction/validation_samples.jsonl"
```

## 🔍 故障排查

### 找不到文件？
检查新的目录结构：
```bash
cat outputs/DIRECTORY_STRUCTURE.md
```

### 旧日志文件？
旧日志文件仍然在 `outputs/logs/` 中，但没有时间戳。
新运行会自动添加时间戳。

### TensorBoard看不到数据？
确保配置已更新：
```bash
grep "report_to" configs/training_config.yaml
# 应该显示: report_to: "tensorboard"
```

---

最后更新: 2024-12-10

# 从Checkpoint恢复训练

## 📋 概述

当训练中断或者想要继续训练更多epochs时，可以从之前保存的checkpoint恢复训练。

## 🚀 使用方法

### 1. 查看可用的Checkpoint

```bash
ls -lh checkpoints/sft_model/
```

输出示例：
```
checkpoint-1000/
checkpoint-1500/
checkpoint-2000/  ← 最新的checkpoint
```

### 2. 检查Checkpoint的训练进度

```bash
python3 -c "
import json
data = json.load(open('checkpoints/sft_model/checkpoint-2000/trainer_state.json'))
print(f'当前步数: {data[\"global_step\"]}')
print(f'当前epoch: {data[\"epoch\"]:.2f}')
print(f'训练进度: {data[\"epoch\"]/3.0*100:.1f}%')
"
```

输出示例：
```
当前步数: 2000
当前epoch: 1.77
训练进度: 59.1%
```

### 3. 从Checkpoint恢复训练

**基本用法：**
```bash
python run_model_training.py --resume checkpoints/sft_model/checkpoint-2000 -y
```

**完整参数：**
```bash
python run_model_training.py \
  --resume checkpoints/sft_model/checkpoint-2000 \
  --no-unsloth \  # 可选：不使用unsloth加速
  -y              # 可选：跳过确认提示
```

## 📊 恢复训练的特点

### ✅ 会保留的状态
- ✅ **模型权重**：LoRA适配器参数
- ✅ **优化器状态**：Adam的momentum等
- ✅ **学习率调度**：warmup和decay的当前状态
- ✅ **训练步数**：从checkpoint的步数继续
- ✅ **随机数状态**：确保可重复性

### 📁 Checkpoint内容
```
checkpoint-2000/
├── adapter_model.safetensors  # LoRA权重
├── optimizer.pt               # 优化器状态
├── scheduler.pt               # 学习率调度器状态
├── trainer_state.json         # 训练状态（步数、epoch等）
├── training_args.bin          # 训练参数
└── rng_state.pth             # 随机数生成器状态
```

## 🔄 训练流程

### 从头开始训练
```bash
python run_model_training.py -y
```
- 从基础模型 `model/models` 开始
- Step: 0 → 3000 (假设3 epochs)
- Epoch: 0.0 → 3.0

### 从Checkpoint恢复
```bash
python run_model_training.py --resume checkpoints/sft_model/checkpoint-2000 -y
```
- 从checkpoint-2000恢复
- Step: 2000 → 3000
- Epoch: 1.77 → 3.0

## 📈 监控训练

### TensorBoard
```bash
# 查看所有训练运行（包括恢复的训练）
tensorboard --logdir outputs/3_training/runs --port 6006 --bind_all
```

**注意：** 恢复训练会创建新的TensorBoard日志目录（带新时间戳），但训练步数会从checkpoint的步数继续。

### 查看日志
```bash
# 查看最新的训练日志
ls -t outputs/logs/sft_trainer_*.log | head -1 | xargs tail -f
```

## 💡 使用场景

### 场景1：训练中断
```bash
# 训练在step 2000时中断
# 直接从最新checkpoint恢复
python run_model_training.py --resume checkpoints/sft_model/checkpoint-2000 -y
```

### 场景2：想要训练更多epochs
```bash
# 1. 修改配置文件增加epochs
vim configs/training_config.yaml
# 将 num_train_epochs: 3 改为 num_train_epochs: 5

# 2. 从checkpoint恢复，继续训练
python run_model_training.py --resume checkpoints/sft_model/checkpoint-2000 -y
```

### 场景3：调整学习率继续训练
```bash
# 1. 修改配置文件降低学习率
vim configs/training_config.yaml
# 将 learning_rate: 2.0e-4 改为 learning_rate: 1.0e-4

# 2. 从checkpoint恢复
python run_model_training.py --resume checkpoints/sft_model/checkpoint-2000 -y
```

## ⚠️ 注意事项

### 1. 配置一致性
恢复训练时，大部分训练配置会从checkpoint中的 `training_args.bin` 恢复。但某些配置会使用新的值：
- ✅ 可以修改：`num_train_epochs`, `learning_rate`, `logging_steps`
- ❌ 不建议修改：`batch_size`, `model_name`, `max_seq_length`

### 2. 数据一致性
- 确保训练数据文件没有改变
- 如果数据改变，可能导致训练不稳定

### 3. Checkpoint完整性
确保checkpoint目录完整，包含所有必要文件：
```bash
ls checkpoints/sft_model/checkpoint-2000/
# 应该看到：adapter_model.safetensors, optimizer.pt, scheduler.pt等
```

### 4. 磁盘空间
- 每个checkpoint约260MB
- 确保有足够空间保存新的checkpoint

## 🔍 故障排查

### 问题1：找不到checkpoint
```
Error: [Errno 2] No such file or directory: 'checkpoints/sft_model/checkpoint-2000'
```

**解决：**
```bash
# 检查checkpoint是否存在
ls -lh checkpoints/sft_model/
# 使用正确的路径
```

### 问题2：配置不匹配
```
ValueError: The model is not compatible with the checkpoint
```

**解决：**
- 不要修改模型架构相关配置（max_seq_length, dtype等）
- 确保使用相同的基础模型路径

### 问题3：显存不足
```
RuntimeError: CUDA out of memory
```

**解决：**
```bash
# 使用相同的batch size和gradient accumulation
# 或者减小batch size（但可能影响训练效果）
```

## 📚 参考命令汇总

```bash
# 查看checkpoint列表
ls -lh checkpoints/sft_model/

# 查看训练进度
python3 -c "import json; print(json.load(open('checkpoints/sft_model/checkpoint-2000/trainer_state.json'))['epoch'])"

# 从checkpoint恢复训练（推荐）
python run_model_training.py --resume checkpoints/sft_model/checkpoint-2000 -y

# 从头开始训练（对比）
python run_model_training.py -y

# 查看TensorBoard
tensorboard --logdir outputs/3_training/runs --port 6006 --bind_all

# 查看训练日志
ls -t outputs/logs/sft_trainer_*.log | head -1 | xargs tail -f
```

---

最后更新: 2024-12-10

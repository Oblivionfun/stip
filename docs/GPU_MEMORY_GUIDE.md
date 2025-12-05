# 显存优化建议（32GB GPU）

## 当前配置（推荐，适合32GB）
已经很优化，预计使用18-20GB显存

## 如果需要更多显存空间，可以调整：

### 方案1：减小序列长度（最有效）
model:
  max_seq_length: 1536  # 从2048降到1536，节省~3GB

### 方案2：减小batch size
training:
  per_device_train_batch_size: 1  # 从2降到1，节省~5GB
  gradient_accumulation_steps: 8  # 保持有效batch=8

### 方案3：减小LoRA rank
lora:
  r: 8  # 从16降到8，节省~0.2GB（但可能影响效果）

## 显存监控命令

训练时监控GPU使用：
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者在另一个终端
while true; do nvidia-smi; sleep 1; done
```

## 如果遇到OOM（内存不足）

1. 停止训练（Ctrl+C）
2. 调整configs/training_config.yaml中的参数
3. 重新运行python run_stage3.py

## 推荐的训练命令

```bash
# 方式1：直接运行（推荐）
python run_stage3.py

# 方式2：在tmux中运行（防止SSH断开）
tmux new -s training
python run_stage3.py
# 按Ctrl+B然后按D来detach
# 重新连接: tmux attach -t training

# 方式3：使用nohup后台运行
nohup python run_stage3.py > training.log 2>&1 &
# 查看日志: tail -f training.log
```

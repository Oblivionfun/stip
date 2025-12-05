# 数据优化与评估指标说明

> **问题**: 具体时间信息（如"08:42 Monday"）可能导致模型过拟合
> **解决**: 使用抽象时段 + 完善评估指标

---

## 🎯 优化1：时间信息抽象化（防止过拟合）

### 优化前（有过拟合风险）
```
Time: 08:42 (Monday)
```

**问题**：
- ❌ 模型可能学会"08:42 = 早高峰"这种简单规则
- ❌ 遇到训练中未见过的时间（如08:37）泛化能力差
- ❌ 具体到分钟的时间对决策帮助有限
- ❌ 星期几对通勤决策影响较小

### 优化后（更好的泛化）
```
Time period: morning rush hour
```

**优势**：
- ✅ 抽象化时段，减少过拟合
- ✅ 保留关键信息（是否高峰期）
- ✅ 更好的泛化能力
- ✅ 符合人类实际决策逻辑

### 时段类型
- **morning rush hour**: 早高峰通勤（7:00-9:00）
- **evening rush hour**: 晚高峰通勤（17:00-19:00）
- **off-peak hours**: 非高峰时段（其他时间）

### 示例对比

**优化前**：
```json
{
  "departure_time": "08:42",
  "day_of_week": "Monday"
}
```

**优化后**：
```json
{
  "time_period": "morning rush hour"
}
```

---

## 📊 优化2：完善的评估指标体系

### 核心评估指标

#### 1. 路径选择准确率 (Route Selection Accuracy)
**定义**: 模型选择的路径（A或B）与ground truth一致的比例

```python
accuracy = correct_predictions / total_samples
```

**目标**: ≥ 75%

**意义**: 衡量模型是否能做出正确的路径选择

---

#### 2. 改道决策F1分数 (Reroute F1 Score)
**定义**: 改道vs保持原路径的二分类指标

```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
```

**目标**: F1 ≥ 0.80

**意义**:
- 高precision: 改道决策准确，不乱改道
- 高recall: 该改道时不会错过
- F1综合评估改道决策质量

---

#### 3. Perplexity（困惑度）
**定义**: 语言模型在验证集上的困惑度

```python
perplexity = exp(average_loss)
```

**目标**: 尽可能低（通常 < 10 为优秀）

**意义**: 衡量模型对训练数据分布的拟合程度

---

#### 4. 偏好一致性 (Preference Consistency) [可选]
**定义**: 模型决策是否与Persona偏好因子一致

**示例检查**：
- 高`risk_aversion`的Persona → 倾向保守决策
- 高`time_sensitivity`的Persona → 倾向激进改道
- 高`familiar_route_preference`的Persona → 倾向保持原路径

**计算方式**：
```python
# 对于高风险厌恶的Persona
if persona['risk_aversion'] > 0.5:
    # 检查模型是否倾向于选择低不确定性路径
    consistency = check_conservative_bias(predictions)
```

---

### 评估运行方式

#### 快速评估（100样本）
```bash
python src/training/evaluator.py
```

#### 完整评估（全部验证集）
```python
from src.training.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model_path="checkpoints/sft_model")
results = evaluator.run_evaluation(num_samples=None)  # 全部样本
evaluator.print_results(results)
```

#### 输出示例
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

---

## 🔬 优化效果预期

### 时间抽象化带来的改进：
1. **泛化能力提升**: 模型不再依赖具体时间，而是理解"高峰期"这一概念
2. **样本效率更高**: 相同的3个时段（早高峰/晚高峰/非高峰）复用性更强
3. **减少过拟合**: 训练loss和验证loss差距减小
4. **更符合实际**: 人类决策也是基于"时段类型"而非具体分钟

### 评估指标的完善：
1. **多维度评估**: 不仅看loss，还看决策质量
2. **业务相关**: 改道F1直接反映实际应用效果
3. **可解释性强**: 每个指标都有明确的业务含义
4. **易于对比**: 可以与baseline模型、规则方法对比

---

## 📝 修改的文件

1. **src/data_construction/scenario_generator.py**
   - 修改`generate_departure_time()` → `generate_time_period()`
   - 场景对象：`departure_time` + `day_of_week` → `time_period`

2. **src/data_construction/sample_builder.py**
   - Prompt模板：使用`time_period`替代具体时间
   - Reflection生成：移除时间引用

3. **src/training/evaluator.py** (新增)
   - 完整的评估脚本
   - 路径选择准确率、改道F1、Perplexity

---

## 🚀 下一步操作

### 1. 重新生成训练数据
```bash
# 删除旧数据
rm outputs/scenarios.json outputs/decisions.json outputs/train_samples.jsonl outputs/validation_samples.jsonl

# 重新运行阶段2
python run_stage2.py
```

### 2. 运行训练
```bash
python run_stage3.py
```

### 3. 评估模型
```bash
python src/training/evaluator.py
```

### 4. 对比优化前后效果
- 对比训练前后的改道决策F1
- 检查路径选择准确率是否提升
- 观察perplexity是否更低

---

## ✅ 总结

| 方面 | 优化前 | 优化后 |
|------|--------|--------|
| 时间表示 | 具体时间（08:42） | 抽象时段（morning rush hour） |
| 过拟合风险 | 高 | 低 |
| 泛化能力 | 差 | 好 |
| 评估指标 | 仅Loss/Perplexity | 路径准确率 + 改道F1 + Perplexity |
| 业务相关性 | 弱 | 强 |

这些优化使模型更加robust，评估更加全面，更符合实际应用需求！

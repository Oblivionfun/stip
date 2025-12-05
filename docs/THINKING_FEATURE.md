# Thinking过程输出说明

> **新增功能**: 模型现在会在输出中显示思考过程，提高可解释性

---

## 📋 Response格式更新

### 优化前（仅有reflection）
```json
{
  "reflection": "I am currently at Home, planning to travel to Office during morning rush hour. My usual route shows a moderate 10-minute delay...",
  "plan": "update path: Side_St, Local_Road_2",
  "concepts": []
}
```

**问题**: 直接给出结论，缺乏推理过程

---

### 优化后（新增thinking）
```json
{
  "thinking": "Let me analyze the current situation:\n- Route A (usual): 35min (delay: 10min, uncertainty: moderate)\n- Route B (alternative): 27min (delay: 0min, uncertainty: low)\n\nConsidering my preferences:\n- I prioritize time efficiency\n- I prefer predictable routes over uncertain ones\n\nThe delay (10min) is within my tolerance threshold (10min).",

  "reflection": "Based on my analysis, although Route A has a 10-minute delay, it's within my tolerance. However, Route B is 8 minutes faster and has lower uncertainty, which aligns with my preference for predictability. Therefore, I will switch to Route B.",

  "plan": "update path: Side_St, Local_Road_2",
  "concepts": []
}
```

**优势**:
- ✅ 展示完整的推理链条
- ✅ 清晰的思考步骤
- ✅ 可验证的决策逻辑

---

## 🧠 Thinking字段的内容结构

### 1. 情况分析 (Situation Analysis)
```
Let me analyze the current situation:
- Route A (usual): 35min (delay: 10min, uncertainty: moderate)
- Route B (alternative): 27min (delay: 0min, uncertainty: low)
```

列出所有路径的关键信息：
- 当前旅行时间
- 延误时长
- 不确定性等级

---

### 2. 偏好考虑 (Preference Consideration)
```
Considering my preferences:
- I prioritize time efficiency
- I prefer predictable routes over uncertain ones
- I prefer sticking with familiar routes
```

根据Persona的偏好因子，列出相关的决策偏好：
- `time_sensitivity` > 0.3 → "I prioritize time efficiency"
- `risk_aversion` > 0.3 → "I prefer predictable routes"
- `familiar_route_preference` > 0.3 → "I prefer familiar routes"
- `rerouting_proactiveness` > 0.3 → "I actively seek alternatives"

---

### 3. 阈值判断 (Threshold Check)
```
The delay (10min) exceeds my tolerance threshold (8min).
```

或

```
The delay (5min) is within my tolerance threshold (10min).
```

对比当前延误与个人阈值，作为决策的关键依据。

---

### 4. 特殊情境 (Special Context)
```
- I have an urgent meeting, cannot afford to be late
```

如果有重要会议等特殊情况，会在thinking中明确指出。

---

## 🎯 Thinking vs Reflection的区别

| 维度 | Thinking | Reflection |
|------|----------|-----------|
| **目的** | 展示分析过程 | 说明最终决策 |
| **风格** | 列表式、结构化 | 叙述式、连贯 |
| **内容** | 客观信息 + 偏好 | 推理链 + 结论 |
| **长度** | 较短、要点式 | 较长、完整句子 |

### Thinking示例
```
Let me analyze:
- Route A: 35min, delay 10min
- Route B: 27min, no delay

My preferences:
- Time efficiency priority
- Risk aversion

Judgment: Delay exceeds threshold
```

### Reflection示例
```
I am currently at Home, heading to Office during morning rush hour.
My usual route (Route A) has a 10-minute delay, which exceeds my
8-minute tolerance. Given my preference for time efficiency and
predictability, I will switch to Route B, which is 8 minutes faster
and has no delay.
```

---

## 💡 可解释性提升

### 优化前（黑盒决策）
```
用户: 为什么选择了Route B？
模型: 因为Route B更快且延误更少
用户: 但是你的偏好是喜欢熟悉路线？
模型: ...（无法解释）
```

### 优化后（透明推理）
```
用户: 为什么选择了Route B？
模型:
Thinking过程显示：
1. Route A延误10分钟，超过我的8分钟阈值
2. 我的偏好是时间效率优先（time_sensitivity=0.8）
3. 虽然我偏好熟悉路线（0.6），但时间因素权重更高
4. Route B节省8分钟，且不确定性更低

因此选择Route B
```

---

## 📊 训练时的处理

### Completion-only训练
```yaml
completion_only:
  enabled: true
  response_template: '{\n    "thinking":'  # 从thinking开始计算loss
```

**效果**: 模型学习生成完整的thinking + reflection + plan

---

## 🔍 评估时的使用

评估脚本会解析thinking字段，但**主要评估plan字段**：

```python
def parse_model_output(output_text):
    result = json.loads(output_text)
    # result包含：
    # {
    #   "thinking": "...",     # 可用于人工检查
    #   "reflection": "...",   # 可用于质量评估
    #   "plan": "..."          # 用于准确率计算
    # }
    return result
```

**Thinking的作用**:
1. 提高用户信任度
2. 便于调试和错误分析
3. 可用于偏好一致性检查

---

## ✅ 使用方法

### 1. 重新生成训练数据
```bash
rm outputs/scenarios.json outputs/decisions.json outputs/train_samples.jsonl outputs/validation_samples.jsonl
python run_stage2.py
```

### 2. 检查生成的样本
```bash
head -1 outputs/train_samples.jsonl | python -m json.tool
```

应该看到：
```json
{
  "id": "CN_xxxx_Sxx",
  "prompt": "...",
  "response": {
    "thinking": "Let me analyze...",
    "reflection": "Based on my analysis...",
    "plan": "update path: ...",
    "concepts": []
  }
}
```

### 3. 训练模型
```bash
python run_stage3.py
```

### 4. 推理时查看thinking
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("checkpoints/sft_model")
model = AutoModelForCausalLM.from_pretrained("checkpoints/sft_model")

# 推理
response = model.generate(...)
output = json.loads(response)

print("Thinking:", output['thinking'])
print("Reflection:", output['reflection'])
print("Plan:", output['plan'])
```

---

## 🎓 示例对比

### 简单场景（无延误）
```json
{
  "thinking": "Let me analyze the current situation:\n- Route A (usual): 25min (delay: 0min, uncertainty: low)\n- Route B (alternative): 27min (delay: 0min, uncertainty: low)\n\nConsidering my preferences:\n- I prefer sticking with familiar routes\n\nThe delay (0min) is within my tolerance threshold (10min).",

  "reflection": "My usual route is clear with no delays. Since I prefer familiar routes and there's no compelling reason to change, I will stick with Route A.",

  "plan": "none"
}
```

### 复杂场景（严重延误+重要会议）
```json
{
  "thinking": "Let me analyze the current situation:\n- Route A (usual): 45min (delay: 20min, uncertainty: high)\n- Route B (alternative): 27min (delay: 0min, uncertainty: low)\n\nConsidering my preferences:\n- I prioritize time efficiency\n- I prefer predictable routes over uncertain ones\n- I have an urgent meeting, cannot afford to be late\n\nThe delay (20min) exceeds my tolerance threshold (10min).",

  "reflection": "I am facing a significant 20-minute delay on Route A due to severe congestion, which far exceeds my 10-minute tolerance. Given that I have an urgent meeting and cannot be late, and considering my preference for time efficiency and predictability, switching to Route B is the clear choice. It saves 18 minutes and has no uncertainty.",

  "plan": "update path: Side_St, Local_Road_2"
}
```

---

## 🔬 可解释性分析

Thinking字段使得我们可以验证：

1. **偏好一致性**: 决策是否符合Persona的偏好因子
2. **逻辑合理性**: 推理链是否完整
3. **阈值准确性**: 是否正确考虑了delay_tolerance
4. **情境感知**: 是否考虑了特殊情况（如重要会议）

这为模型的可靠性评估提供了额外维度！

---

**现在可以重新生成训练数据，训练带有思考过程的模型了！** 🧠✨

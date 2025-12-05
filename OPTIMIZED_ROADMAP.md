# 优化后的技术路线 - 基于参考项目的改进方案

> **参考项目分析总结**：
> - **GATSim**: 生成式Agent交通仿真系统，提供Persona架构和决策Prompt模板
> - **Centaur**: Nature论文的人类认知模型，展示unsloth+LoRA高效微调方法
> - **PNAS**: 大规模决策数据分析，提供向量化+聚类的偏好建模pipeline

---

## 🔄 核心优化点

### 1. **借鉴GATSim的Persona架构** ⭐⭐⭐
**原方案**：简单的偏好向量+离散标签
**优化方案**：完整的Persona属性系统

```python
# GATSim风格的Persona定义
persona = {
    # 个人属性
    'name': 'Zhang Wei',
    'age': 34,
    'gender': 'male',
    'occupation': 'software engineer',
    'education': 'bachelor',

    # 交通偏好（从问卷提取）
    'preferences_in_transportation':
        'prefer familiar routes and stable travel times; '
        'high risk aversion; rely heavily on navigation apps; '
        'willing to reroute to avoid >10min delays',

    # 驾驶特质（从问卷因子提取）
    'innate': 'cautious, time-sensitive, information-dependent',

    # 家庭属性
    'household_income': 'middle',
    'commute_time_morning': 25,  # 从问卷Q1/Q2
    'typical_congestion_duration': 8,  # 从问卷Q4/Q5

    # 国家标签
    'country': 'CN'
}
```

**优势**：
- 与GATSim格式兼容，便于借鉴其prompt模板
- 属性可解释性强，便于后期分析
- 自然语言描述便于LLM理解

---

### 2. **采用GATSim的决策输出格式** ⭐⭐⭐

**原方案**：
```
Plan decision: CHOOSE_ROUTE=A, REROUTE=false
```

**优化方案（GATSim格式）**：
```json
{
    "reflection": "I am currently at home, departing for work at 08:00. The usual route via Ave_1 shows moderate congestion with an estimated 9-minute delay at Ave_1_link_1. As someone who values travel time reliability and is risk-averse, I prefer to avoid uncertainty. Route B via Ave_2 takes 2 minutes longer nominally but offers more predictable travel time and avoids the congested segment.",

    "plan": "update path: Ave_2, St_3",

    "concepts": []
}
```

**plan字段支持4种格式**：
1. `"none"` - 保持当前计划
2. `"update path: shortest"` - 改用实时最短路径
3. `"update path: Ave_2, St_3"` - 指定路径（逗号分隔的路段名）
4. `"update departure time: 08:15"` - 调整出发时间

**优势**：
- `reflection`提供可解释的推理过程（便于评估模型是否学会了偏好→决策的映射）
- `plan`格式标准化，易于解析
- 与GATSim生态兼容（可直接对接仿真环境）

---

### 3. **使用Centaur的高效微调技术** ⭐⭐⭐

**原方案**：标准的LoRA微调
**优化方案**：Centaur风格的unsloth+4bit+completion-only训练

```python
# 使用unsloth库（比HF Trainer快2-5倍）
from unsloth import FastLanguageModel, UnslothTrainer
from trl import DataCollatorForCompletionOnlyLM

# 4bit量化加载模型（节省50%显存）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2-7B-Instruct",
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,  # 显存优化
)

# LoRA配置（借鉴Centaur参数）
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # 增加到16（Centaur用8，我们任务稍复杂）
    lora_alpha = 32,
    lora_dropout = 0.05,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_rslora = True,  # RSLoRA提升稳定性
)

# Completion-only训练（仅在输出部分计算loss）
# 定义response开始标记
response_template = '{\n    "reflection":'
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)
```

**关键改进**：
1. **仅在JSON输出部分计算loss**：避免模型在输入部分浪费学习能力
2. **4bit量化**：单卡V100即可训练Qwen2-7B
3. **unsloth优化**：训练速度提升2-5倍

---

### 4. **简化数据构造流程** ⭐⭐

**原方案**：构建完整仿真环境 → 离散选择模型模拟 → 生成训练数据
**优化方案**：问卷驱动 + 模板引擎 + 规则策略

```python
# 直接从问卷数据提取决策偏好
def extract_decision_preference(survey_row):
    """从问卷行提取决策偏好"""
    return {
        'reroute_threshold_delay': survey_row['Q21'],  # 多长延误会改道
        'reroute_willingness_accident': survey_row['Q20'],  # 遇事故改道意愿
        'reroute_willingness_congestion': survey_row['Q19'],  # 遇拥堵改道意愿
        'prefer_familiar_route': survey_row['Q22'],  # 熟悉路线偏好
        'prefer_smooth_over_fast': survey_row['Q29'],  # 宁可慢但顺畅
        'info_seeking_frequency': survey_row['Q12'],  # 信息查询频率
    }

# 场景+偏好 → 决策（基于规则+随机性）
def simulate_decision(scenario, preference):
    """根据偏好和场景模拟决策"""
    delay = scenario['route_A_delay']
    threshold = preference['reroute_threshold_delay']

    # 规则1：延误超过阈值 → 高概率改道
    if delay > threshold:
        reroute_prob = 0.8
    elif delay > threshold * 0.5:
        reroute_prob = 0.4
    else:
        reroute_prob = 0.1

    # 规则2：风险厌恶型 → 提高改道概率
    if preference['prefer_smooth_over_fast'] > 4:  # Likert 5分制
        reroute_prob += 0.2

    # 规则3：熟悉路线偏好 → 降低改道概率
    if preference['prefer_familiar_route'] > 4:
        reroute_prob -= 0.15

    # 随机采样
    reroute = np.random.rand() < reroute_prob

    return {
        'reroute': reroute,
        'chosen_route': 'B' if reroute else 'A'
    }

# 使用Jinja2模板生成GATSim格式文本
template = """
You play the role of the person:
{{ persona_description }}

Current time: {{ current_time }}
You are planning to travel from {{ origin }} to {{ destination }}.

Available routes:
- Route A (usual route): {{ route_a_description }}
- Route B (alternative): {{ route_b_description }}

Current traffic conditions:
{{ traffic_conditions }}

What route would you choose?
"""
```

**数据扩充策略**：
- 从3000问卷样本 → 生成10000训练样本
- 每个persona × 3-5个场景变体（不同延误程度/事件类型/时段）
- 可选：用GPT-4生成高质量reflection文本作为teacher

---

## 📋 优化后的任务清单（阶段1-4）

### **阶段1：Persona建模层** (3个任务)

#### 任务1.1：数据预处理
- **输入**：`data/*.xlsx`
- **输出**：`outputs/cleaned_survey_data.csv`
- **改进**：保留更多原始字段（通勤时间、拥堵时长等），用于Persona构建

#### 任务1.2：偏好因子提取 + Persona属性生成
- **输入**：`cleaned_survey_data.csv`
- **输出**：
  - `outputs/preference_factors.csv` (因子得分)
  - `outputs/personas.json` (**新增**，GATSim格式的完整Persona)
- **改进**：不仅提取因子，还生成自然语言描述的偏好特征

```json
// personas.json 示例
{
  "CN_0001": {
    "name": "Zhang Wei",
    "age": 34,
    "gender": "male",
    "preferences_in_transportation": "prefer familiar routes; high risk aversion; rely on navigation; willing to reroute if delay >10min",
    "innate": "cautious, time-sensitive, information-dependent",
    "commute_time_morning": 25,
    "country": "CN",
    ...
  }
}
```

#### 任务1.3：Persona聚类与类型标签
- **输入**：`preference_factors.csv`
- **输出**：`outputs/persona_types.json`
- **改进**：聚类后为每个类型定义描述性标签（如`"TYPE_RISK_AVOID_FAMILIAR"`）

---

### **阶段2：训练数据构造层** (3个任务)

#### 任务2.1：场景库构建
- **代码**：`src/data_construction/scenario_templates.py`
- **改进**：借鉴GATSim的场景描述格式

```python
# 场景模板（GATSim风格）
scenario_template = {
    "origin": "Home",
    "destination": "Office",
    "departure_time": "08:00",
    "trip_purpose": "commute_to_work",

    "routes": [
        {
            "id": "A",
            "description": "usual route via Ave_1 and Metro_2",
            "normal_travel_time": 25,
            "current_delay": 9,
            "current_travel_time": 34,
            "uncertainty": "moderate",
            "congestion_location": "Ave_1_link_1",
            "familiarity": 0.95
        },
        {
            "id": "B",
            "description": "alternative route via Ave_2 and St_3",
            "normal_travel_time": 27,
            "current_delay": 0,
            "current_travel_time": 27,
            "uncertainty": "low",
            "familiarity": 0.60
        }
    ],

    "traffic_events": [
        {
            "location": "Ave_1_link_1",
            "type": "congestion",
            "severity": "moderate",
            "expected_delay": 9
        }
    ]
}
```

#### 任务2.2：决策模拟器
- **代码**：`src/data_construction/decision_simulator.py`
- **改进**：基于问卷偏好的规则策略（不需要复杂的Logit模型）

```python
def simulate_decision(persona, scenario):
    """根据Persona偏好模拟决策"""
    # 提取关键偏好参数
    delay_threshold = persona['reroute_threshold_delay']  # 从Q21
    risk_aversion = persona['preference_factors']['risk_aversion']  # 从因子分析
    familiar_preference = persona['preference_factors']['familiarity']

    # 计算效用（简化版Logit）
    route_a = scenario['routes'][0]
    route_b = scenario['routes'][1]

    utility_a = (
        -0.5 * route_a['current_travel_time']
        - risk_aversion * route_a['uncertainty_score']
        + familiar_preference * route_a['familiarity']
    )

    utility_b = (
        -0.5 * route_b['current_travel_time']
        - risk_aversion * route_b['uncertainty_score']
        + familiar_preference * route_b['familiarity']
    )

    # 加入随机扰动（Gumbel noise）
    noise_a = np.random.gumbel()
    noise_b = np.random.gumbel()

    chosen = 'A' if (utility_a + noise_a) > (utility_b + noise_b) else 'B'

    return {
        'chosen_route': chosen,
        'reroute': chosen != 'A',  # A是默认路线
        'reasoning': generate_reasoning(persona, scenario, chosen)
    }
```

#### 任务2.3：GATSim格式样本生成
- **代码**：`src/data_construction/llm_sample_builder.py`
- **改进**：使用Jinja2模板 + GATSim JSON格式

```python
# Prompt模板（借鉴GATSim）
prompt_template = """
You play the role of the person:
Name: {{ persona.name }} | Age: {{ persona.age }} | Gender: {{ persona.gender }}
Occupation: {{ persona.occupation }}
Transportation preferences: {{ persona.preferences_in_transportation }}
Personality: {{ persona.innate }}

Current situation:
Time: {{ scenario.departure_time }}
Location: {{ scenario.origin }}
Destination: {{ scenario.destination }}
Purpose: {{ scenario.trip_purpose }}

Available routes:
{% for route in scenario.routes %}
- Route {{ route.id }}: {{ route.description }}
  * Normal travel time: {{ route.normal_travel_time }} minutes
  * Current delay: {{ route.current_delay }} minutes
  * Familiarity: {{ "High" if route.familiarity > 0.8 else "Medium" if route.familiarity > 0.5 else "Low" }}
{% endfor %}

Current traffic conditions:
{% for event in scenario.traffic_events %}
- {{ event.type }} at {{ event.location }}, severity: {{ event.severity }}, expected delay: {{ event.expected_delay }} minutes
{% endfor %}

What route would you choose and why? Respond in JSON format:
{
    "reflection": "your reasoning process...",
    "plan": "update path: Ave_2, St_3",
    "concepts": []
}
"""

# Response生成
response = {
    "reflection": f"I am currently at {scenario['origin']}, planning to depart for {scenario['destination']} at {scenario['departure_time']}. Route A via Ave_1 shows a {route_a_delay}-minute delay due to congestion. Given my preference for {preference_description}, I will {decision_reasoning}.",

    "plan": "update path: Ave_2, St_3" if chosen=='B' else "none",

    "concepts": []  # 训练阶段可为空
}
```

---

### **阶段3：大模型微调层** (3个任务)

#### 任务3.1：数据集准备
- **改进**：实现completion-only tokenization

```python
# 仅在JSON response部分计算loss
def tokenize_function(examples):
    # 找到response起始位置
    response_start = examples['text'].find('{\n    "reflection":')

    # Tokenize全文
    full_tokens = tokenizer(examples['text'])

    # 找到response的token起始位置
    prompt_tokens = tokenizer(examples['text'][:response_start])
    prompt_len = len(prompt_tokens['input_ids'])

    # 创建labels（仅在response部分计算loss）
    labels = full_tokens['input_ids'].copy()
    labels[:prompt_len] = [-100] * prompt_len  # -100 = ignore in loss

    return {
        'input_ids': full_tokens['input_ids'],
        'attention_mask': full_tokens['attention_mask'],
        'labels': labels
    }
```

#### 任务3.2：unsloth + LoRA微调
- **代码**：`src/training/sft_trainer.py`
- **改进**：使用unsloth库替代标准Trainer

```python
from unsloth import FastLanguageModel, UnslothTrainer

# 4bit加载
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2-7B-Instruct",
    max_seq_length = 4096,
    load_in_4bit = True,
)

# LoRA配置
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

# Trainer
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 4096,
    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,  # effective batch=16
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True,  # V100/A100支持
        logging_steps = 10,
        save_steps = 100,
        output_dir = "checkpoints/sft_model",
    ),
)

trainer.train()
```

#### 任务3.3：评估与验证
- **改进**：新增"plan字段准确率"指标

```python
def evaluate_plan_accuracy(predictions, labels):
    """评估plan字段的解析准确率"""
    correct = 0
    total = 0

    for pred, label in zip(predictions, labels):
        try:
            pred_json = json.loads(pred)
            label_json = json.loads(label)

            # 检查plan字段是否匹配
            if pred_json['plan'] == label_json['plan']:
                correct += 1
            total += 1
        except:
            total += 1

    return correct / total if total > 0 else 0
```

---

### **阶段4：决策智能体推理层** (3个任务)

#### 任务4.1：智能体封装
- **改进**：使用unsloth的推理优化

```python
from unsloth import FastLanguageModel

class RouteDecisionAgent:
    def __init__(self, model_path):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 4096,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.model)  # 推理加速

    def decide(self, scenario, persona):
        """生成路径决策"""
        prompt = self.build_prompt(scenario, persona)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens = 512,
            temperature = 0.7,
            do_sample = True,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 解析JSON
        decision = self.parse_response(response)

        return decision
```

#### 任务4.2：动作解析器（GATSim兼容）
- **改进**：支持GATSim的4种plan格式

```python
def parse_plan(plan_str):
    """解析plan字段为可执行动作"""
    if plan_str == "none":
        return {"action": "keep_current_plan"}

    elif plan_str.startswith("update path: "):
        path = plan_str.replace("update path: ", "").strip()
        if path == "shortest":
            return {"action": "reroute", "method": "shortest"}
        else:
            roads = [r.strip() for r in path.split(",")]
            return {"action": "reroute", "method": "specified", "roads": roads}

    elif plan_str.startswith("update departure time: "):
        time = plan_str.replace("update departure time: ", "").strip()
        return {"action": "delay_departure", "new_time": time}

    else:
        # 完整活动计划（暂不支持，返回保持原计划）
        return {"action": "keep_current_plan"}
```

#### 任务4.3：Persona冷启动适配器
- **改进**：从新问卷快速生成Persona

```python
def create_persona_from_survey(survey_response):
    """从问卷快速生成Persona对象"""
    # 使用预训练的因子模型提取偏好
    preference_factors = factor_model.transform(survey_response)

    # 聚类分配类型标签
    persona_type = cluster_model.predict(preference_factors)[0]
    type_label = persona_type_labels[persona_type]

    # 构建GATSim格式Persona
    persona = {
        'name': survey_response.get('name', f"User_{uuid.uuid4().hex[:8]}"),
        'age': survey_response['age'],
        'gender': survey_response['gender'],
        'preferences_in_transportation': generate_preference_description(preference_factors),
        'innate': persona_type_descriptions[type_label],
        'commute_time_morning': survey_response['Q1'],
        'country': survey_response['country'],
        ...
    }

    return persona
```

---

## 🎯 关键技术对比表

| 维度 | 原方案 | 优化方案 | 改进点 |
|-----|--------|---------|--------|
| **Persona建模** | 简单偏好向量 | GATSim风格完整Persona | 可解释性强、便于prompt构建 |
| **决策格式** | `CHOOSE_ROUTE=A` | `{"reflection": ..., "plan": ...}` | 提供推理过程、格式标准化 |
| **微调方法** | 标准LoRA | unsloth + 4bit + completion-only | 速度快2-5倍、显存省50% |
| **数据构造** | 离散选择模型模拟 | 规则策略 + 模板引擎 | 实现简单、与问卷紧密结合 |
| **模型选择** | 未定 | **Qwen2-7B-Instruct** | 中文强、社区支持好、7B可单卡训练 |

---

## 📅 优化后的开发时间线

| 阶段 | 任务 | 预计时间 | 关键交付物 |
|-----|------|---------|-----------|
| **阶段1** | Persona建模 | 3-4天 | `personas.json` (GATSim格式) |
| **阶段2** | 数据构造 | 4-5天 | `train_samples.jsonl` (10k样本) |
| **阶段3** | 模型微调 | 5-7天 | `checkpoints/sft_model/` (LoRA权重) |
| **阶段4** | 智能体API | 2-3天 | `RouteDecisionAgent` 类 |
| **总计** | | **14-19天** | 完整可用的决策智能体系统 |

---

## 💡 实施建议

### 立即行动项

1. **安装依赖**：
```bash
pip install unsloth transformers datasets peft trl
pip install pandas scikit-learn jinja2 networkx
```

2. **下载基座模型**（在微调前）：
```bash
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir models/Qwen2-7B
```

3. **准备GATSim prompt模板**：
   - 直接复用 `reference_getsim/gatsim/gatsim/agent/llm_modules/prompt_templates/update_daily_activity_plan_v1.txt`
   - 根据我们的场景微调变量

### 风险缓解

1. **数据量不足**：
   - 每个persona生成3-5个场景变体
   - 可选：用GPT-4生成高质量reflection文本作为增强

2. **显存不足**：
   - 使用4bit量化（40GB显存 → 20GB）
   - 降低batch size，增加gradient accumulation

3. **模型输出格式不稳定**：
   - completion-only训练聚焦JSON输出
   - 推理时用constrained decoding强制JSON格式

---

## ❓ 待确认问题

1. **模型选择**：确认使用 **Qwen2-7B-Instruct** ？（推荐，中文强且7B可单卡训练）

2. **数据规模**：目标生成 **10000个训练样本**（每个persona约3-4个场景）？

3. **评估方式**：
   - **离线评估**：在测试集上计算plan准确率
   - **人工评估**：抽样100个样本人工标注reflection质量
   - 暂不做仿真评估？

4. **teacher模型**：是否使用GPT-4/Claude生成高质量reflection文本？（可选，提升样本质量但增加成本）

**请确认以上优化方案，我将立即开始编写代码！** 🚀

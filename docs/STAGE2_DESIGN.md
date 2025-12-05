# 阶段2设计文档：训练数据构造

> **最后更新**: 2025-12-04
> **状态**: 设计中

---

## 📋 目标

从1002个Persona生成10,000个LLM训练样本（prompt-response pairs），格式兼容GATSim决策系统。

---

## 🏗️ 架构设计

### 数据流

```
personas.json (1002)
    ↓
[场景生成器] → 生成多样化场景 (每个Persona × 10个场景变体)
    ↓
[决策模拟器] → 基于偏好因子模拟决策
    ↓
[样本构造器] → 转换为GATSim格式的prompt-response
    ↓
train_samples.jsonl (10,000)
```

---

## 📦 核心模块

### 模块1：场景生成器 (scenario_generator.py)

**功能**：生成多样化的路径选择场景

**输入**：
- `personas.json` - Persona对象
- `configs/scenario_config.yaml` - 场景配置

**输出**：
- `outputs/scenarios.json` - 场景库

**场景结构**：
```python
scenario = {
    "scenario_id": "CN_0001_S01",
    "persona_id": "CN_0001",

    # 出行信息
    "origin": "Home",
    "destination": "Office",
    "departure_time": "08:00",
    "trip_purpose": "commute_to_work",
    "day_of_week": "Monday",

    # 路径选项（2-3条）
    "routes": [
        {
            "id": "A",
            "name": "Usual Route",
            "description": "via Ave_1 and Metro_2",
            "normal_travel_time": 25,  # 分钟
            "current_delay": 9,
            "current_travel_time": 34,
            "uncertainty_level": "moderate",  # low/moderate/high
            "familiarity": 0.95  # 0-1
        },
        {
            "id": "B",
            "name": "Alternative Route",
            "description": "via Ave_2 and St_3",
            "normal_travel_time": 27,
            "current_delay": 0,
            "current_travel_time": 27,
            "uncertainty_level": "low",
            "familiarity": 0.60
        }
    ],

    # 交通事件
    "traffic_events": [
        {
            "location": "Ave_1_link_1",
            "type": "congestion",  # congestion/accident/construction/weather
            "severity": "moderate",  # minor/moderate/severe
            "expected_delay": 9
        }
    ],

    # 情境因素
    "context": {
        "weather": "clear",  # clear/rain/snow
        "is_rush_hour": true,
        "has_important_meeting": false,
        "time_buffer": 10  # 剩余时间裕量（分钟）
    }
}
```

**场景变体生成策略**：

每个Persona生成10个场景，覆盖不同情况：

1. **延误程度变化** (4个场景)
   - 轻微延误 (5min)
   - 中度延误 (10min)
   - 严重延误 (20min)
   - 无延误对比

2. **事件类型变化** (3个场景)
   - 事故
   - 施工
   - 天气

3. **时段变化** (2个场景)
   - 早高峰
   - 非高峰

4. **情境因素** (1个场景)
   - 重要会议 + 时间紧张

**实现要点**：
- 根据Persona的通勤时间（Q1）设置normal_travel_time基准
- 熟悉路线设置familiarity=0.9+，替代路线0.5-0.7
- 事件严重程度与delay_tolerance匹配，确保有决策张力

---

### 模块2：决策模拟器 (decision_simulator.py)

**功能**：根据Persona偏好因子和场景，模拟路径选择决策

**输入**：
- `personas.json`
- `scenarios.json`

**输出**：
- `outputs/decisions.json` - 决策结果

**决策结构**：
```python
decision = {
    "scenario_id": "CN_0001_S01",
    "persona_id": "CN_0001",
    "chosen_route": "B",
    "reroute": True,
    "reasoning": "high delay exceeds tolerance threshold",
    "utility_scores": {
        "route_A": -12.5,
        "route_B": -8.3
    }
}
```

**决策模型**：简化的效用函数 + Gumbel噪声

```python
def calculate_utility(persona, route, scenario):
    """计算路径效用"""
    # 基础时间成本
    time_cost = -0.5 * route['current_travel_time']

    # 延误惩罚（基于time_sensitivity因子）
    delay_penalty = -persona['preference_factors']['time_sensitivity'] * route['current_delay']

    # 不确定性惩罚（基于risk_aversion因子）
    uncertainty_map = {'low': 0, 'moderate': 5, 'high': 10}
    uncertainty_penalty = -persona['preference_factors']['risk_aversion'] * uncertainty_map[route['uncertainty_level']]

    # 熟悉路线偏好（基于familiar_route_preference因子）
    familiarity_bonus = persona['preference_factors']['familiar_route_preference'] * route['familiarity'] * 3

    # 总效用
    utility = time_cost + delay_penalty + uncertainty_penalty + familiarity_bonus

    return utility

def simulate_decision(persona, scenario):
    """模拟决策"""
    routes = scenario['routes']

    # 计算各路径效用
    utilities = {}
    for route in routes:
        utility = calculate_utility(persona, route, scenario)
        noise = np.random.gumbel(0, 1)  # Gumbel噪声模拟随机性
        utilities[route['id']] = utility + noise

    # 选择效用最高的路径
    chosen = max(utilities, key=utilities.get)

    return {
        'chosen_route': chosen,
        'reroute': chosen != 'A',  # A为默认路线
        'utility_scores': {k: v for k, v in utilities.items()}
    }
```

**关键偏好因子映射**：
- `time_sensitivity` → 延误权重
- `risk_aversion` → 不确定性权重
- `familiar_route_preference` → 熟悉路线偏好
- `rerouting_proactiveness` → 改道倾向基准
- `information_dependency` → 是否参考实时信息

**质量控制**：
- 确保决策分布合理（改道率30-70%）
- 极端偏好的Persona应有明显决策倾向
- 添加阈值规则：delay > delay_tolerance_planned → 强制改道概率0.8+

---

### 模块3：样本构造器 (sample_builder.py)

**功能**：将场景+决策转换为GATSim格式的LLM训练样本

**输入**：
- `personas.json`
- `scenarios.json`
- `decisions.json`

**输出**：
- `outputs/train_samples.jsonl` - 训练样本（10,000行）

**样本格式（JSONL，每行一个样本）**：
```json
{
    "id": "CN_0001_S01",
    "persona_id": "CN_0001",
    "scenario_id": "CN_0001_S01",
    "prompt": "<完整的GATSim格式prompt>",
    "response": {
        "reflection": "<推理过程>",
        "plan": "update path: Ave_2, St_3",
        "concepts": []
    }
}
```

**Prompt模板**（Jinja2）：
```jinja2
You play the role of the person:
Name: {{ persona.name }} | Age: {{ persona.age }} | Gender: {{ persona.gender }} | Country: {{ persona.country }}
Occupation: {{ persona.occupation }} | Education: {{ persona.education }}

Transportation preferences: {{ persona.preferences_in_transportation }}
Personality traits: {{ persona.innate }}

Current situation:
Time: {{ scenario.departure_time }} ({{ scenario.day_of_week }})
Location: {{ scenario.origin }}
Destination: {{ scenario.destination }}
Purpose: {{ scenario.trip_purpose }}
{% if scenario.context.has_important_meeting %}
⚠️ Important: You have an urgent meeting and cannot be late.
{% endif %}

Available routes:
{% for route in scenario.routes %}
Route {{ route.id }} ({{ route.name }}):
- Description: {{ route.description }}
- Normal travel time: {{ route.normal_travel_time }} minutes
- Current travel time: {{ route.current_travel_time }} minutes (delay: {{ route.current_delay }} min)
- Familiarity: {{ "High - your usual route" if route.familiarity > 0.8 else "Medium" if route.familiarity > 0.5 else "Low - unfamiliar" }}
- Uncertainty: {{ route.uncertainty_level }}
{% endfor %}

Current traffic conditions:
{% for event in scenario.traffic_events %}
- {{ event.type|capitalize }} at {{ event.location }}: {{ event.severity }} severity, expected {{ event.expected_delay }}-minute delay
{% endfor %}

{% if scenario.context.weather != 'clear' %}
Weather: {{ scenario.context.weather }}
{% endif %}

What route would you choose and why? Respond in JSON format:
{
    "reflection": "your reasoning process considering your preferences and the current situation",
    "plan": "your route choice (use 'none' to keep Route A, or 'update path: <route_name>' to switch)",
    "concepts": []
}
```

**Response生成**（reflection文本生成）：

基于决策结果生成自然语言推理：

```python
def generate_reflection(persona, scenario, decision):
    """生成reflection文本"""
    parts = []

    # 1. 情境描述
    parts.append(f"I am currently at {scenario['origin']}, planning to depart for {scenario['destination']} at {scenario['departure_time']}.")

    # 2. 路况分析
    route_a = scenario['routes'][0]
    if route_a['current_delay'] > 0:
        parts.append(f"My usual route (Route A) shows a {route_a['current_delay']}-minute delay due to {scenario['traffic_events'][0]['type']}.")

    # 3. 偏好考量
    key_factor = get_dominant_factor(persona)
    if key_factor == 'risk_aversion' and persona['preference_factors']['risk_aversion'] > 0.5:
        parts.append("As someone who values travel time reliability, I prefer predictable routes over potentially faster but uncertain options.")
    elif key_factor == 'time_sensitivity':
        parts.append("Time efficiency is my priority, so I will choose the faster option even if it means taking an unfamiliar route.")
    elif key_factor == 'familiar_route_preference':
        parts.append("I generally prefer my usual route unless the delay is significant.")

    # 4. 决策结论
    if decision['reroute']:
        chosen_route = next(r for r in scenario['routes'] if r['id'] == decision['chosen_route'])
        parts.append(f"Therefore, I will switch to Route {decision['chosen_route']}, which saves {route_a['current_travel_time'] - chosen_route['current_travel_time']} minutes.")
    else:
        parts.append("The delay is within my tolerance, so I will stick with my usual route.")

    return " ".join(parts)

def generate_plan(decision, scenario):
    """生成plan字段"""
    if not decision['reroute']:
        return "none"

    chosen_route = next(r for r in scenario['routes'] if r['id'] == decision['chosen_route'])
    return f"update path: {chosen_route['description'].replace('via ', '')}"
```

---

## ⚙️ 配置文件

### configs/scenario_config.yaml

```yaml
scenario_generation:
  # 每个Persona生成的场景数
  scenarios_per_persona: 10

  # 场景模板
  templates:
    - name: "commute_morning"
      origin: "Home"
      destination: "Office"
      time_range: ["07:00", "09:00"]
      trip_purpose: "commute_to_work"
      weight: 0.5

    - name: "commute_evening"
      origin: "Office"
      destination: "Home"
      time_range: ["17:00", "19:00"]
      trip_purpose: "commute_from_work"
      weight: 0.3

    - name: "errand"
      origin: "Home"
      destination: "Shopping_Mall"
      time_range: ["10:00", "15:00"]
      trip_purpose: "shopping"
      weight: 0.2

  # 延误分布
  delay_scenarios:
    - level: "minor"
      delay_range: [3, 7]
      weight: 0.3
    - level: "moderate"
      delay_range: [8, 15]
      weight: 0.4
    - level: "severe"
      delay_range: [16, 30]
      weight: 0.2
    - level: "none"
      delay_range: [0, 0]
      weight: 0.1

  # 事件类型
  event_types:
    - congestion: 0.5
    - accident: 0.2
    - construction: 0.15
    - weather: 0.1
    - special_event: 0.05

  # 路径配置
  routes:
    default_num: 2  # 默认2条路径（A和B）
    familiarity:
      usual_route: [0.90, 0.98]
      alternative_route: [0.50, 0.75]

    travel_time_variation: 0.15  # ±15% 随机变化

# 数据质量控制
quality_control:
  # 决策分布目标
  target_reroute_rate: [0.35, 0.65]  # 改道率应在35-65%

  # 确保多样性
  min_scenarios_per_category:
    delay_level: 2
    event_type: 2
    time_period: 2
```

---

## 📊 预期输出

### 文件清单

| 文件名 | 大小 | 行数 | 描述 |
|--------|------|------|------|
| `scenarios.json` | ~15MB | 10,020 | 场景库（1002 persona × 10） |
| `decisions.json` | ~3MB | 10,020 | 决策结果 |
| `train_samples.jsonl` | ~80MB | 10,000 | LLM训练样本 |
| `data_construction.log` | ~1MB | - | 运行日志 |

### 数据统计

- **总样本数**: 10,000
- **Persona覆盖**: 1002 (每个Persona约10个样本)
- **改道率**: 40-60%
- **场景多样性**:
  - 延误等级分布: minor(30%), moderate(40%), severe(20%), none(10%)
  - 事件类型分布: congestion(50%), accident(20%), construction(15%), weather(10%), other(5%)
  - 时段分布: morning(50%), evening(30%), midday(20%)

---

## 🔧 实现计划

### Task 2.1: 场景生成器 (2-3小时)
- [ ] 创建scenario_generator.py
- [ ] 实现场景模板系统
- [ ] 实现变体生成逻辑
- [ ] 添加场景验证

### Task 2.2: 决策模拟器 (2-3小时)
- [ ] 创建decision_simulator.py
- [ ] 实现效用函数
- [ ] 实现Gumbel采样
- [ ] 添加质量控制（改道率检查）

### Task 2.3: 样本构造器 (3-4小时)
- [ ] 创建sample_builder.py
- [ ] 设计Jinja2模板
- [ ] 实现reflection生成逻辑
- [ ] 实现plan字段转换
- [ ] 添加输出格式验证

### Task 2.4: 一键运行脚本 (1小时)
- [ ] 创建run_stage2.py
- [ ] 集成三个模块
- [ ] 添加进度显示
- [ ] 添加统计报告

---

## ✅ 验证标准

1. **数据量**: 生成10,000个有效样本
2. **格式正确性**: 100%符合GATSim JSON格式
3. **决策合理性**: 改道率在35-65%之间
4. **多样性**: 每个场景类型至少1000个样本
5. **一致性**: Persona偏好与决策行为一致

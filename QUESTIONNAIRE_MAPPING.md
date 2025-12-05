# 问卷字段完整映射方案

> **数据概况**：CN/UK/US/AU 四国问卷数据，每个数据集约1000样本，共67个字段

---

## 📊 问卷结构总览

| 类别 | 问题编号 | 字段数 | 用途 |
|-----|---------|-------|------|
| 元数据 | - | 6 | 数据质量控制（不参与建模） |
| 通勤基本信息 | Q1-Q5 | 5 | → Persona基本属性 + 场景构建 |
| 拥堵认知与改道行为 | Q6-Q15 | 10 | → 因子分析 + 改道意愿偏好 |
| 不同情境改道意愿 | Q16-Q21 | 6 | → 因子分析 + 情境敏感度 |
| 路线偏好与决策因素 | Q22-Q42 | 21 | → 因子分析（核心） |
| 不同出行类型改道意愿 | Q43-Q48 | 6 | → 出行目的偏好建模 |
| 人口统计信息 | Q49-Q58 | 10 | → Persona人口属性 + 分组分析 |
| 汇总分数 | 总分 | 1 | 数据质量检查 |

---

## 🔄 完整映射方案

### 1️⃣ 元数据字段（不参与建模）

| 字段名 | 用途 | 处理方式 |
|-------|------|---------|
| 序号 | 样本ID | 转换为统一ID格式：`{country}_{序号}` |
| 提交答卷时间 | 数据质量检查 | 用于检测批量填写/异常时间 |
| 所用时间 | 数据质量检查 | 过滤填写过快（<180秒）的样本 |
| 来源 | 数据来源标记 | 仅记录，不参与建模 |
| 来源详情 | - | 缺失，忽略 |
| 来自IP | 地理位置验证 | 验证country标签一致性 |

---

### 2️⃣ 通勤基本信息（Q1-Q5）→ Persona基本属性

#### 映射到GATSim Persona格式：

```python
persona = {
    # 直接映射
    'commute_time_morning': Q1,  # 早晨通勤时间（分钟）
    'commute_time_afternoon': Q2,  # 下午通勤时间（分钟）
    'main_route_type': decode_Q3(Q3),  # 主要路线类型
    'congestion_duration_morning': Q4,  # 早晨拥堵时长（分钟）
    'congestion_duration_afternoon': Q5,  # 下午拥堵时长（分钟）

    # 衍生特征
    'congestion_ratio_morning': Q4 / Q1,  # 拥堵占比
    'congestion_ratio_afternoon': Q5 / Q2,
    'commute_asymmetry': abs(Q1 - Q2),  # 通勤时间不对称性
}

def decode_Q3(value):
    """Q3编码解析"""
    mapping = {
        1: "highway",      # 高速公路
        2: "arterial",     # 主干道
        3: "local_street"  # 地方街道
    }
    return mapping.get(value, "unknown")
```

#### 用于场景构建：
- **Q1, Q2** → 生成符合真实通勤时长的路径选项
- **Q4, Q5** → 生成符合真实拥堵程度的延误场景
- **Q3** → 生成对应道路类型的路径描述

---

### 3️⃣ 拥堵认知与改道行为（Q6-Q15）→ 偏好因子

#### 因子1：**改道主动性** (Rerouting Proactiveness)

| 问题 | 原始问题 | 编码 | 因子载荷预期 |
|-----|---------|------|-------------|
| Q7 | 我在通勤时经常因为交通拥堵而改道 | 1-5 (Likert) | 正向 ++ |
| Q6 | 通勤中拥堵是一个问题 | 1-5 | 正向 + |
| Q15 | 我经常因拥堵信息取消行程或改变目的地 | 1-5 | 正向 + |

#### 因子2：**改道决策信心** (Rerouting Confidence)

| 问题 | 原始问题 | 编码 | 因子载荷预期 |
|-----|---------|------|-------------|
| Q8 | 改道时我很相信自己在做最好的选择 | 1-5 | 正向 ++ |
| Q9 | 改道后可能返回正常路线 | 1-5 | 正向 + |

#### 因子3：**信息依赖度** (Information Dependency)

| 问题 | 原始问题 | 编码 | 因子载荷预期 |
|-----|---------|------|-------------|
| Q12 | 我经常寻求交通信息/路线引导 | 1-5 | 正向 ++ |
| Q11 | 何时寻求交通信息 | 1-6 | 正向 + |
| Q10 | 如何获取交通信息 | 多选 | 需one-hot处理 |

#### 因子4：**信息独立性** (Information Independence)

| 问题 | 原始问题 | 编码 | 因子载荷预期 |
|-----|---------|------|-------------|
| Q13 | 我经常不同意所得到的路线引导 | 1-5 | 正向 ++ |
| Q14 | 我通常不理会路线引导，自己选择路线 | 1-5 | 正向 ++ |

**Q10特殊处理**：
```python
# Q10是多选题，格式如 "1,2,8"
def process_Q10(value):
    """解析信息获取渠道"""
    channels = {
        '1': 'radio',
        '2': 'navigation_app',
        '3': 'vms',  # 可变信息板
        '4': 'web',
        '5': 'social_media',
        '6': 'friends',
        '7': 'experience',
        '8': 'other'
    }
    selected = value.split(',') if isinstance(value, str) else []
    return {
        'info_channels': [channels.get(c.strip(), 'unknown') for c in selected],
        'info_channel_count': len(selected),
        'use_nav_app': '2' in selected,  # 重点特征
        'use_vms': '3' in selected
    }
```

---

### 4️⃣ 不同情境改道意愿（Q16-Q21）→ 情境敏感度因子

#### 因子5：**事件响应度** (Event Responsiveness)

| 问题 | 情境类型 | 编码 | 特征名 |
|-----|---------|------|--------|
| Q16 | 作业区（施工） | 1-5 | `reroute_construction` |
| Q17 | 特殊事件（音乐会等） | 1-5 | `reroute_special_event` |
| Q18 | 天气事件 | 1-5 | `reroute_weather` |
| Q19 | 高峰期拥堵 | 1-5 | `reroute_peak_congestion` |
| Q20 | 事故 | 1-5 | `reroute_accident` |

#### 关键阈值特征：

| 问题 | 原始问题 | 编码 | 用途 |
|-----|---------|------|------|
| **Q21** | 多长延误会导致寻找另一条路线 | 连续值（分钟） | **核心阈值参数** |
| **Q42** | 在拥堵中等待多长时间后改道 | 连续值（分钟） | **实时决策阈值** |

**Q21 vs Q42 的区别**：
- **Q21**：预期延误阈值（出发前看到信息）
- **Q42**：实时等待阈值（已经在路上）

```python
# 用于决策模拟的关键参数
decision_params = {
    'delay_tolerance_planned': Q21,  # 出发前容忍延误
    'delay_tolerance_realtime': Q42,  # 路上容忍等待
    'delay_tolerance_ratio': Q42 / Q21 if Q21 > 0 else 1.0  # 沉没成本效应
}
```

---

### 5️⃣ 路线偏好与决策因素（Q22-Q42）→ 核心偏好因子

#### 因子6：**熟悉路线偏好** (Familiar Route Preference)

| 问题 | 原始问题 | 编码 | 反向题 |
|-----|---------|------|-------|
| Q22 | 我更倾向于在熟悉的路线上行驶 | 1-5 | ❌ |
| Q23 | 如果对该地区熟悉，更愿意改道 | 1-5 | ✅ 反向 |

**反向题处理**：Q23需要反转 → `6 - Q23`

#### 因子7：**时间可靠性偏好** (Time Reliability Preference)

| 问题 | 原始问题 | 编码 | 因子载荷 |
|-----|---------|------|---------|
| Q25 | 旅行时间的一致性/可靠性对我很重要 | 1-5 | 正向 ++ |
| Q29 | 即使更长时间，也选顺畅路线而非拥堵路线 | 1-5 | 正向 ++ |
| Q27 | 如果改道更长时间，就不会改道 | 1-5 | 负向 - |

#### 因子8：**时间敏感度** (Time Sensitivity)

| 问题 | 原始问题 | 编码 | 因子载荷 |
|-----|---------|------|---------|
| Q30 | 只要能节省时间，就会立即改道 | 1-5 | 正向 ++ |
| Q27 | 如果改道更长时间，就不会改道 | 1-5 | 正向 + |

#### 因子9：**路线类型偏好** (Route Type Preference)

| 问题 | 原始问题 | 编码 | 含义 |
|-----|---------|------|------|
| Q24 | 相比地方街道，更倾向改道到高速 | 1-5 | 高速偏好 |
| Q32 | 替代路线上红绿灯数量影响选择意愿 | 1-5 | 信号灯敏感度 |

#### 因子10：**选择灵活性偏好** (Choice Flexibility)

| 问题 | 原始问题 | 编码 |
|-----|---------|------|
| Q26 | 有多种改道选择时更愿意改道 | 1-5 |
| Q33 | 离目的地远时更愿意改变路线 | 1-5 |

#### 因子11：**外部信息敏感度** (External Information Sensitivity)

| 问题 | 原始问题 | 编码 |
|-----|---------|------|
| Q36 | 认为可变信息板有用且准确 | 1-5 |
| Q37 | 如有可变信息板建议，更愿意改道 | 1-5 |
| Q38 | 看到其他驾驶员改道，也更愿意改道 | 1-5 |

#### 因子12：**风险厌恶度** (Risk Aversion)

| 问题 | 原始问题 | 编码 | 反向题 |
|-----|---------|------|-------|
| Q34 | 我避免开车经过有作业区的地区 | 1-5 | ❌ |
| Q28 | 关闭车道数量影响改道意愿 | 1-5 | ❌ |
| Q35 | 我知道特殊事件可能影响交通 | 1-5 | ❌ |
| Q40 | 我认为我是激进的驾驶员 | 1-5 | ✅ 反向 |
| Q41 | 我认为我是谨慎的驾驶员 | 1-5 | ❌ |

#### 其他特征：

| 问题 | 原始问题 | 用途 |
|-----|---------|------|
| Q31 | 请选择"有点不同意" | ⚠️ 注意力检查题，不参与因子分析 |
| Q39 | 车上有乘客时驾驶方式不同 | 社会行为特征，可选 |

---

### 6️⃣ 不同出行类型改道意愿（Q43-Q48）→ 出行目的建模

#### 用途：验证偏好一致性 + 场景权重调整

| 问题 | 出行类型 | 编码 | 映射到场景 |
|-----|---------|------|----------|
| Q43 | 通勤（早晨） | 1-5 | `trip_purpose: "commute_morning"` |
| Q44 | 通勤（下午） | 1-5 | `trip_purpose: "commute_afternoon"` |
| Q45 | 购物出行 | 1-5 | `trip_purpose: "shopping"` |
| Q46 | 休闲社交 | 1-5 | `trip_purpose: "leisure"` |
| Q47 | 长途度假 | 1-5 | `trip_purpose: "vacation"` |
| Q48 | 紧急疏散 | 1-5 | `trip_purpose: "emergency"` |

**用于数据一致性检查**：
```python
# 检查逻辑一致性
def check_consistency(row):
    """检查改道意愿的一致性"""
    # 通勤改道意愿应该与Q7相关
    commute_reroute_avg = (row['Q43'] + row['Q44']) / 2
    general_reroute = row['Q7']

    # 相关系数应 > 0.3
    if abs(commute_reroute_avg - general_reroute) > 3:
        return "inconsistent"

    # 紧急疏散改道意愿应该最高
    if row['Q48'] < max(row['Q43'], row['Q44'], row['Q45']):
        return "suspicious"

    return "ok"
```

**用于场景权重**：
```python
# 根据出行目的调整改道概率
trip_purpose_weights = {
    'commute_morning': row['Q43'] / 5.0,  # 归一化到[0,1]
    'commute_afternoon': row['Q44'] / 5.0,
    'shopping': row['Q45'] / 5.0,
    'leisure': row['Q46'] / 5.0,
    'vacation': row['Q47'] / 5.0,
    'emergency': row['Q48'] / 5.0,
}
```

---

### 7️⃣ 人口统计信息（Q49-Q58）→ Persona人口属性

#### 直接映射到GATSim Persona格式：

```python
persona = {
    # 基本人口属性
    'gender': decode_gender(Q49),  # 1=男, 2=女
    'age': Q50,  # 20-66岁
    'ethnicity': decode_ethnicity(Q51),  # 民族
    'education': decode_education(Q54),  # 注意：这里是Q54而非Q32！
    'occupation': decode_occupation(Q53),
    'household_income': decode_income(Q52),

    # 车辆与技术
    'vehicle_type': decode_vehicle(Q55),  # 所有人都选1，可能是单选题缺陷
    'has_navigation': Q56 == 1,  # 1=有, 2=无

    # 空间位置
    'home_location_type': decode_location(Q57),  # 1=市中心, 2=市区, 3=郊区, 4=农村
    'work_location_type': decode_location(Q58),

    # 国家标签
    'country': country_code  # 'CN', 'UK', 'US', 'AU'
}

def decode_gender(value):
    return "male" if value == 1 else "female"

def decode_education(value):
    """Q54（实际是Q32字段）"""
    mapping = {
        1: "high_school",
        2: "bachelor",
        3: "master",
        4: "phd"
    }
    return mapping.get(value, "unknown")

def decode_income(value):
    """Q52: 1-7级收入"""
    if value <= 2:
        return "low"
    elif value <= 5:
        return "middle"
    else:
        return "high"

def decode_location(value):
    mapping = {
        1: "downtown",
        2: "urban",
        3: "suburban",
        4: "rural"
    }
    return mapping.get(value, "unknown")

def decode_occupation(value):
    """Q53: 需要查看编码本"""
    # TODO: 需要问卷编码本确定具体类别
    mapping = {
        1: "manager",
        2: "professional",
        3: "technician",
        4: "clerk",
        5: "service",
        6: "other"
    }
    return mapping.get(value, "other")
```

---

## 🎯 完整的Persona生成Pipeline

### Step 1: 数据预处理

```python
def preprocess_survey(df, country_code):
    """清洗并标准化问卷数据"""

    # 1. 过滤无效样本
    df = df[df['所用+C1:D851时间'] >= 180]  # 至少3分钟
    df = df[df['Q31'] == 2]  # 注意力检查题正确

    # 2. 反向题处理
    df['Q23_reversed'] = 6 - df['Q23']
    df['Q40_reversed'] = 6 - df['Q40']

    # 3. 添加衍生特征
    df['congestion_ratio_morning'] = df['Q4'] / df['Q1'].replace(0, 1)
    df['congestion_ratio_afternoon'] = df['Q5'] / df['Q2'].replace(0, 1)
    df['delay_tolerance_ratio'] = df['Q42'] / df['Q21'].replace(0, 1)

    # 4. Q10多选题处理
    df['info_channel_count'] = df['Q10'].str.split(',').str.len()
    df['use_nav_app'] = df['Q10'].str.contains('2', na=False)

    # 5. 添加国家标签
    df['country'] = country_code

    # 6. 生成唯一ID
    df['persona_id'] = country_code + '_' + df['序号'].astype(str).str.zfill(4)

    return df
```

### Step 2: 因子分析

```python
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

def extract_preference_factors(df):
    """提取偏好因子"""

    # 选择态度题（Likert量表）
    attitude_cols = [
        # 改道主动性
        'Q6', 'Q7', 'Q15',
        # 改道信心
        'Q8', 'Q9',
        # 信息依赖
        'Q11', 'Q12',
        # 信息独立
        'Q13', 'Q14',
        # 情境响应
        'Q16', 'Q17', 'Q18', 'Q19', 'Q20',
        # 熟悉偏好（Q23已反转）
        'Q22', 'Q23_reversed',
        # 时间可靠性
        'Q25', 'Q29', 'Q27',
        # 时间敏感度
        'Q30',
        # 路线类型
        'Q24', 'Q32',
        # 灵活性
        'Q26', 'Q33',
        # 外部信息
        'Q36', 'Q37', 'Q38',
        # 风险厌恶
        'Q34', 'Q28', 'Q35', 'Q40_reversed', 'Q41'
    ]

    X = df[attitude_cols].fillna(df[attitude_cols].median())

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 因子分析（确定最优因子数）
    n_factors = 8  # 根据碎石图/累计方差确定
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    factors = fa.fit_transform(X_scaled)

    # 因子命名（根据载荷矩阵）
    factor_names = [
        'rerouting_proactiveness',
        'information_dependency',
        'time_reliability_preference',
        'risk_aversion',
        'familiar_route_preference',
        'time_sensitivity',
        'external_info_sensitivity',
        'route_flexibility'
    ]

    # 创建因子DataFrame
    factor_df = pd.DataFrame(
        factors,
        columns=factor_names,
        index=df.index
    )

    return factor_df, fa.components_  # 返回因子得分和载荷矩阵
```

### Step 3: 生成GATSim格式Persona

```python
def generate_persona(row, factors):
    """从问卷行生成GATSim格式Persona"""

    # 1. 基本属性
    persona = {
        'name': generate_name(row['persona_id'], row['Q49'], row['country']),
        'age': int(row['Q50']),
        'gender': 'male' if row['Q49'] == 1 else 'female',
        'education': decode_education(row['Q54']),
        'occupation': decode_occupation(row['Q53']),

        # 2. 交通偏好（自然语言描述）
        'preferences_in_transportation': generate_preference_description(row, factors),

        # 3. 个性特质
        'innate': generate_personality_description(factors),

        # 4. 通勤属性
        'commute_time_morning': int(row['Q1']),
        'commute_time_afternoon': int(row['Q2']),
        'congestion_duration_morning': int(row['Q4']),
        'congestion_duration_afternoon': int(row['Q5']),
        'main_route_type': decode_route_type(row['Q3']),

        # 5. 家庭属性
        'household_income': decode_income(row['Q52']),
        'home_location': decode_location(row['Q57']),
        'work_location': decode_location(row['Q58']),

        # 6. 车辆与技术
        'has_navigation': row['Q56'] == 1,
        'licensed_driver': True,  # 问卷对象都是驾驶员

        # 7. 国家标签
        'country': row['country'],

        # 8. 决策关键参数（用于模拟）
        'delay_tolerance_planned': int(row['Q21']),
        'delay_tolerance_realtime': int(row['Q42']),
        'reroute_threshold_construction': int(row['Q16']),
        'reroute_threshold_accident': int(row['Q20']),
        'reroute_threshold_congestion': int(row['Q19']),

        # 9. 偏好因子得分
        'preference_factors': {
            name: float(factors[name])
            for name in factors.index
        }
    }

    return persona

def generate_preference_description(row, factors):
    """生成自然语言偏好描述"""
    parts = []

    # 熟悉路线偏好
    if row['Q22'] >= 4:
        parts.append("prefer familiar routes")

    # 风险厌恶
    if factors['risk_aversion'] > 0.5:
        parts.append("high risk aversion")
    elif factors['risk_aversion'] < -0.5:
        parts.append("risk-tolerant")

    # 信息依赖
    if factors['information_dependency'] > 0.5:
        parts.append("rely heavily on navigation apps")
    elif factors['information_dependency'] < -0.5:
        parts.append("prefer self-judgment over navigation")

    # 时间可靠性
    if row['Q29'] >= 4:
        parts.append("prefer smooth routes over fast but uncertain ones")

    # 改道阈值
    if row['Q21'] <= 10:
        parts.append(f"willing to reroute if delay >{row['Q21']}min")
    else:
        parts.append(f"patient with delays up to {row['Q21']}min")

    # 时间敏感度
    if row['Q30'] >= 4:
        parts.append("immediately reroute to save time")

    return "; ".join(parts)

def generate_personality_description(factors):
    """生成个性描述"""
    traits = []

    if factors['risk_aversion'] > 0.5:
        traits.append("cautious")
    elif factors['risk_aversion'] < -0.5:
        traits.append("bold")

    if factors['time_sensitivity'] > 0.5:
        traits.append("time-sensitive")

    if factors['information_dependency'] > 0.5:
        traits.append("information-dependent")

    if factors['route_flexibility'] > 0.5:
        traits.append("flexible")

    if not traits:
        traits = ["rational", "balanced"]

    return ", ".join(traits)
```

---

## 📋 未映射字段汇总

### 完全未使用的字段：

| 字段 | 原因 |
|-----|------|
| 来源详情 | 全部缺失 |
| 来自IP | 仅用于数据验证 |
| 请在如下表述中选择一个（×2） | 值全为1，无区分度 |
| Q55（车辆类型） | 值全为1，无区分度 |
| Q31（注意力检查题） | 仅用于过滤样本 |
| 总分 | 仅用于数据质量检查 |

### 部分使用的字段：

| 字段 | 使用情况 | 备注 |
|-----|---------|------|
| Q10（信息渠道） | 衍生特征：是否使用导航、渠道数量 | 原始多选值不直接用于因子分析 |
| Q11（何时寻求信息） | 用于因子分析 | 但具体类别信息未充分利用 |
| Q39（乘客影响） | 可选特征 | 暂未纳入核心因子 |

---

## ✅ 映射覆盖率统计

| 维度 | 字段数 | 已映射 | 覆盖率 |
|-----|-------|-------|-------|
| **通勤基本信息** | 5 | 5 | 100% ✅ |
| **拥堵与改道行为** | 10 | 9 | 90% ✅ (Q10部分使用) |
| **情境改道意愿** | 6 | 6 | 100% ✅ |
| **路线偏好** | 21 | 20 | 95% ✅ (Q31仅用于过滤) |
| **出行类型意愿** | 6 | 6 | 100% ✅ |
| **人口统计** | 10 | 8 | 80% ✅ (Q55无区分度，IP仅验证) |
| **元数据** | 6 | 2 | 33% (仅ID和时间用于质控) |
| **汇总** | 1 | 0 | 0% (仅质检) |
| **总计** | 67 | 58 | **86.6%** ✅ |

---

## 🔍 数据质量检查清单

```python
def quality_check(df):
    """数据质量检查"""

    checks = {
        '填写时长过短': df['所用+C1:D851时间'] < 180,
        '注意力检查失败': df['Q31'] != 2,
        'Q21与Q42不一致': (df['Q42'] / df['Q21']) > 3,  # Q42应>=Q21
        '紧急疏散改道意愿过低': df['Q48'] < 3,  # 应该较高
        '通勤时间异常': (df['Q1'] > 120) | (df['Q2'] > 120),  # 超2小时
        'Q40与Q41矛盾': (df['Q40'] >= 4) & (df['Q41'] >= 4),  # 不能同时激进和谨慎
    }

    for check_name, mask in checks.items():
        invalid_count = mask.sum()
        print(f"{check_name}: {invalid_count} ({invalid_count/len(df)*100:.1f}%)")

    return checks
```

---

## 💡 下一步操作建议

1. **立即确认**：
   - Q10（信息渠道）的具体编码对应关系
   - Q53（职业）的具体编码对应关系
   - 是否需要利用Q39（乘客影响）

2. **因子分析调优**：
   - 运行EFA确定最优因子数（建议6-10个）
   - 检查因子载荷矩阵，命名因子
   - 计算Cronbach's α验证信度

3. **跨国等价性验证**：
   - 多组CFA检查四国因子结构是否一致
   - 如果不一致，考虑分国家建模

---

**请确认此映射方案是否符合您的研究设计！** 🎯

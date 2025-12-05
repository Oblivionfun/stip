# 基于大模型微调的人类驾驶偏好自适应路径决策系统 - 任务清单

## 📊 数据概况
- **数据集**：CN/UK/US/AU 四国问卷数据（共约3000+样本）
- **问卷维度**：67个字段，包含通勤习惯、改道意愿、信息依赖、风险偏好等
- **核心变量**：
  - Q6-Q15：拥堵认知与改道行为
  - Q16-Q20：不同情境改道意愿
  - Q22-Q30：路线偏好与决策因素
  - Q43-Q48：不同出行类型改道倾向
  - 人口统计：性别、年龄、收入、职业等

---

## 🎯 技术路线流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    阶段1：人类偏好建模层                        │
├─────────────────────────────────────────────────────────────┤
│ 输入：问卷数据（xlsx）                                          │
│   ↓                                                          │
│ [1.1] 数据预处理与清洗                                         │
│   ↓                                                          │
│ [1.2] 探索性因子分析（EFA）→ 提取K个偏好因子                    │
│   ↓                                                          │
│ [1.3] 偏好向量计算 + 聚类 → 驾驶员画像（Personas）              │
│   ↓                                                          │
│ 输出：preference_vectors.pkl, persona_labels.json             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 阶段2：训练数据构造层                           │
├─────────────────────────────────────────────────────────────┤
│ 输入：偏好向量 + 路网场景模拟数据                               │
│   ↓                                                          │
│ [2.1] 路网场景生成器（O-D对、候选路径、事件注入）                │
│   ↓                                                          │
│ [2.2] 行为决策模拟（基于离散选择模型 + 偏好向量）                │
│   ↓                                                          │
│ [2.3] 生成LLM训练样本（Prompt + Response）                     │
│      格式：场景描述JSON → 决策推理文本 + Plan decision          │
│   ↓                                                          │
│ 输出：train_samples.jsonl (5000-10000条样本)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  阶段3：大模型微调层                            │
├─────────────────────────────────────────────────────────────┤
│ 输入：train_samples.jsonl + 基座模型（Qwen/LLaMA）             │
│   ↓                                                          │
│ [3.1] 数据集划分与预处理（train/val/test）                     │
│   ↓                                                          │
│ [3.2] SFT监督微调（LoRA/QLoRA）                               │
│      - 行为克隆损失                                            │
│      - Plan decision token加权                                │
│   ↓                                                          │
│ [3.3] 偏好对齐微调（DPO/ORPO可选）                             │
│      - 构造正负样本对                                          │
│      - 偏好对比学习                                            │
│   ↓                                                          │
│ 输出：fine_tuned_model/ (LoRA权重 + adapter_config.json)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 阶段4：决策智能体推理层                         │
├─────────────────────────────────────────────────────────────┤
│ 输入：微调模型 + 实时场景状态                                   │
│   ↓                                                          │
│ [4.1] 智能体接口封装                                           │
│      - 加载模型 + LoRA权重                                     │
│      - Prompt模板管理                                         │
│   ↓                                                          │
│ [4.2] 推理与动作解析                                           │
│      - 生成决策文本                                            │
│      - 解析Plan decision（选路/改道）                          │
│   ↓                                                          │
│ [4.3] 自适应机制                                              │
│      - 冷启动：问卷 → 偏好标签                                 │
│      - 在线更新：经历 → 偏好调整（可选）                        │
│   ↓                                                          │
│ 输出：action = {route_id, reroute, reasoning}                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 ✅ 阶段4完成 - 可直接使用智能体                 │
│                                                              │
│ 后续扩展选项：                                                │
│ - 接入真实仿真环境（SUMO/MATSIM）                             │
│ - 实现在线评估模块                                            │
│ - 开发可视化界面                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 详细任务清单（阶段1-4，共12个任务）

> **注意**：暂时不实现阶段5仿真评估层，聚焦在偏好建模→数据构造→模型微调→智能体推理

### **阶段1：人类偏好建模层** (3个任务)

#### 任务1.1：数据预处理与清洗
- **输入**：`data/*.xlsx` (四国问卷原始数据)
- **代码文件**：`src/preference_modeling/data_loader.py`
- **功能**：
  - 读取Excel并合并四国数据
  - 反向题处理（如Q40 vs Q41）
  - 缺失值填充/删除
  - 标准化数值型变量
  - 添加country标签
- **输出**：`outputs/cleaned_survey_data.csv`

#### 任务1.2：偏好因子提取
- **输入**：`outputs/cleaned_survey_data.csv`
- **代码文件**：`src/preference_modeling/factor_analysis.py`
- **功能**：
  - 选择关键态度题（Q6-Q30, Q43-Q48）
  - 探索性因子分析（EFA）+ 主成分分析（PCA）
  - 确定因子数量（碎石图/累计方差）
  - 因子旋转与命名（如：风险厌恶、信息依赖、熟悉偏好等）
  - 计算每个被试的因子得分
- **输出**：
  - `outputs/factor_loadings.csv` (因子载荷矩阵)
  - `outputs/preference_vectors.pkl` (每个样本的偏好向量)

#### 任务1.3：驾驶员画像聚类
- **输入**：`outputs/preference_vectors.pkl`
- **代码文件**：`src/preference_modeling/persona_clustering.py`
- **功能**：
  - K-Means/GMM聚类（选择最优K值）
  - 为每个cluster定义可解释标签（如：TYPE_RISK_AVOID_FAMILIAR）
  - 生成偏好标签映射表
  - 跨国分布对比可视化
- **输出**：
  - `outputs/persona_labels.json` (样本ID → 偏好标签)
  - `outputs/persona_statistics.png` (聚类分布图)

---

### **阶段2：训练数据构造层** (3个任务)

#### 任务2.1：路网场景生成器
- **输入**：自定义路网配置
- **代码文件**：`src/data_construction/scenario_generator.py`
- **功能**：
  - 定义简化路网（节点+边+路径集合）
  - 生成多样化O-D对
  - 为每个场景随机注入事件（拥堵/事故/施工）
  - 为每条候选路径计算特征：
    - 预期时间（ETA）、不确定性
    - 与习惯路线相似度
    - 路段类型分布（高速/主干/支路）
- **输出**：`outputs/scenarios.json` (1000+场景)

#### 任务2.2：行为决策模拟
- **输入**：`scenarios.json` + `preference_vectors.pkl`
- **代码文件**：`src/data_construction/behavior_simulator.py`
- **功能**：
  - 基于多项Logit模型 + 偏好向量模拟路径选择
  - 效用函数设计：U = β₁·时间 + β₂·不确定性 + β₃·熟悉度 + ...
  - β系数由偏好向量决定
  - 生成改道决策（初始选择 vs 遇事件后调整）
- **输出**：`outputs/simulated_decisions.json`

#### 任务2.3：LLM训练样本生成
- **输入**：`scenarios.json` + `simulated_decisions.json` + `persona_labels.json`
- **代码文件**：`src/data_construction/llm_sample_builder.py`
- **功能**：
  - 构造Prompt（场景JSON + 偏好标签）
  - 生成Response（决策推理文本 + Plan decision行）
  - 使用模板引擎（Jinja2）确保格式一致
  - 可选：调用GPT-4/Claude生成高质量推理文本作为teacher
- **输出**：`outputs/train_samples.jsonl` (格式化训练数据)

**样本格式示例**：
```json
{
  "prompt": "You are a route decision agent simulating a driver with preferences: <TAG_CN> <TYPE_RISK_AVOID> <INFO_HIGH>...\n\nScenario:\n{scenario_json}\n\nWhat route would this driver choose?",
  "response": "As a cautious commuter who values time predictability, I notice Route A has a 9-minute congestion delay. Although Route B takes 2 minutes longer nominally, it offers more certainty...\n\nPlan decision: CHOOSE_ROUTE=B, REROUTE=true"
}
```

---

### **阶段3：大模型微调层** (3个任务)

#### 任务3.1：数据集准备
- **输入**：`outputs/train_samples.jsonl`
- **代码文件**：`src/training/dataset.py`
- **功能**：
  - 划分train/val/test（8:1:1）
  - Tokenization（适配Qwen/LLaMA tokenizer）
  - 构造attention mask和labels
  - DataLoader封装（支持batch processing）
- **输出**：`outputs/processed_dataset/`

#### 任务3.2：SFT监督微调
- **输入**：`processed_dataset/` + 基座模型（如Qwen-7B）
- **代码文件**：`src/training/sft_trainer.py`
- **功能**：
  - 加载预训练模型（from HuggingFace）
  - LoRA配置（r=16, alpha=32, target_modules=[q_proj, v_proj]）
  - 训练循环（AdamW + Cosine LR）
  - Plan decision token损失加权（×2-3倍）
  - 验证集评估（perplexity + action accuracy）
  - 模型checkpoint保存
- **输出**：`checkpoints/sft_model/` (LoRA adapter)
- **配置文件**：`configs/sft_config.yaml`

#### 任务3.3：DPO偏好对齐微调（可选高级功能）
- **输入**：`train_samples.jsonl` + SFT模型
- **代码文件**：`src/training/dpo_trainer.py`
- **功能**：
  - 为每个场景构造正负样本对（chosen vs rejected route）
  - DPO损失函数实现
  - 在SFT模型基础上继续训练
- **输出**：`checkpoints/dpo_model/`

---

### **阶段4：决策智能体推理层** (3个任务)

#### 任务4.1：智能体接口封装
- **输入**：微调模型权重
- **代码文件**：`src/agent/route_agent.py`
- **功能**：
  - 模型加载（base model + LoRA merge）
  - Prompt模板管理（支持不同偏好标签组合）
  - 批量推理接口
  - GPU/CPU自适应
- **API**：
  ```python
  agent = RouteDecisionAgent(model_path="checkpoints/sft_model")
  action = agent.decide(scenario, preference_tags)
  ```

#### 任务4.2：动作解析器
- **代码文件**：`src/agent/action_parser.py`
- **功能**：
  - 从生成文本中提取 `Plan decision` 行
  - 解析 `CHOOSE_ROUTE=A`, `REROUTE=true`
  - 容错处理（正则匹配 + 后备规则）
  - 返回结构化动作字典

#### 任务4.3：自适应偏好管理器
- **代码文件**：`src/agent/preference_adapter.py`
- **功能**：
  - 冷启动：新驾驶员问卷 → 偏好向量 → 标签映射
  - 在线更新：根据历史决策更新偏好权重（贝叶斯更新）
  - 支持手动偏好调整（A/B测试场景）

---

### **阶段5：工程化与文档** (3个任务)

#### 任务5.1：配置管理
- **代码文件**：
  - `configs/sft_config.yaml`
  - `configs/dpo_config.yaml`
  - `configs/simulation_config.yaml`
- **功能**：集中管理所有超参数、路径配置

#### 任务5.2：依赖管理
- **文件**：`requirements.txt` 或 `pyproject.toml`
- **主要依赖**：
  - pandas, numpy, scipy, scikit-learn
  - torch, transformers, peft (LoRA)
  - networkx (路网仿真)
  - matplotlib, seaborn (可视化)

#### 任务5.3：文档编写
- **文件**：`README.md`
- **内容**：
  - 项目简介与技术路线
  - 快速开始指南
  - 数据准备说明
  - 训练与评估命令
  - API文档

---

## 🎯 里程碑与交付物

| 阶段 | 交付物 | 验证标准 |
|-----|--------|---------|
| 阶段1 | 偏好向量 + 驾驶员画像 | 因子可解释性、聚类轮廓系数>0.4 |
| 阶段2 | 5000+训练样本 | 样本多样性、格式合规率100% |
| 阶段3 | 微调模型权重 | Validation accuracy > 75%, Plan decision准确率>85% |
| 阶段4 | 智能体API | 推理延迟<2s, 动作解析成功率>95% |

---

## ⚠️ 关键技术难点与解决方案

1. **问题**：问卷数据量有限（~3000样本），训练数据不足
   - **方案**：通过离散选择模型模拟扩充到10000+样本，使用teacher LLM增强文本质量

2. **问题**：跨国文化差异如何建模
   - **方案**：在偏好标签中显式加入国家标签，微调时分国家加权

3. **问题**：LLM生成的Plan decision格式不稳定
   - **方案**：在训练时加大action token loss权重，推理时用正则+后备规则解析

---

## 📅 建议开发顺序（阶段1-4）

1. **第1周**：阶段1（偏好建模）→ 验证因子可解释性
2. **第2周**：阶段2（数据构造）→ 人工检查样本质量
3. **第3-4周**：阶段3（模型微调）→ 监控训练指标
4. **第5周**：阶段4（智能体推理）→ API测试与调优
5. **后续扩展**：如需评估可再添加阶段5仿真模块

---

**下一步请确认（阶段1-4）：**
1. ✅ 已移除阶段5仿真评估模块
2. 技术路线（偏好建模→数据构造→微调→智能体）是否符合预期？
3. 模型选择倾向：Qwen2-7B（中文强）还是 LLaMA-3-8B（英文生态好）？
4. 数据扩充方式：是否接受用离散选择模型模拟扩充训练样本？
5. 确认后立即开始编写代码！

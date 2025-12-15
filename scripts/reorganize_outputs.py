#!/usr/bin/env python3
"""
重新组织项目目录结构
将outputs下的文件按功能分类到子目录
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def create_organized_structure():
    """创建组织良好的目录结构"""

    base_dir = Path("outputs")

    # 定义新的目录结构
    directories = {
        "1_persona_modeling": "Stage 1: Persona建模输出",
        "2_data_construction": "Stage 2: 训练数据构造输出",
        "3_training": "Stage 3: 模型训练输出",
        "4_evaluation": "Stage 4: 模型评估输出",
        "logs": "运行日志（按时间戳组织）",
    }

    # 创建子目录
    for dir_name, description in directories.items():
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

        # 创建README说明
        readme = dir_path / "README.md"
        if not readme.exists():
            readme.write_text(f"# {description}\n\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"✓ 创建目录: {dir_path}")

    # 创建评估子目录
    eval_dir = base_dir / "4_evaluation"
    (eval_dir / "baseline").mkdir(exist_ok=True)
    (eval_dir / "finetuned").mkdir(exist_ok=True)
    print(f"✓ 创建评估子目录")

    return base_dir


def move_files(base_dir):
    """移动文件到对应目录"""

    # 定义文件映射：源文件 -> 目标目录
    file_mappings = {
        # Stage 1: Persona建模
        "1_persona_modeling": [
            "cleaned_survey_data.csv",
            "preference_factors.csv",
            "factor_loadings.csv",
            "factor_loadings_heatmap.png",
            "personas.json",
            "persona_clustering.png",
            "cluster_distributions.png",
            "persona_types.json",
            "cluster_statistics.json",
        ],

        # Stage 2: 数据构造
        "2_data_construction": [
            "scenarios.json",
            "scenario_statistics.json",
            "decisions.json",
            "decision_statistics.json",
            "train_samples.jsonl",
            "validation_samples.jsonl",
            "sample_statistics.json",
        ],

        # Stage 3: 训练
        "3_training": [
            "training_curves.png",
        ],

        # Stage 4: 评估
        "4_evaluation": [
            "evaluation_results.json",
        ],

        # Baseline评估结果
        "4_evaluation/baseline": [
            "baseline_results_fixed.json",
            "baseline_test.json",
        ],

        # 微调模型评估
        "4_evaluation/finetuned": [
            "checkpoint-2000-results.json",
        ],
    }

    moved_count = 0

    for target_dir, files in file_mappings.items():
        target_path = base_dir / target_dir

        for filename in files:
            source = base_dir / filename
            destination = target_path / filename

            if source.exists():
                # 如果目标已存在，添加时间戳
                if destination.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    name, ext = os.path.splitext(filename)
                    destination = target_path / f"{name}_{timestamp}{ext}"

                shutil.move(str(source), str(destination))
                print(f"  移动: {filename} -> {target_dir}/")
                moved_count += 1
            else:
                print(f"  跳过: {filename} (不存在)")

    print(f"\n✓ 共移动 {moved_count} 个文件")


def organize_logs(base_dir):
    """组织日志文件"""

    logs_source = base_dir / "logs"

    if logs_source.exists():
        log_files = list(logs_source.glob("*.log"))
        print(f"\n整理 {len(log_files)} 个日志文件...")

        for log_file in log_files:
            # 日志文件保持在原位置，但可以添加说明
            print(f"  保留: logs/{log_file.name}")

    return True


def create_directory_tree(base_dir):
    """创建目录树文档"""

    tree_file = base_dir / "DIRECTORY_STRUCTURE.md"

    content = """# 输出目录结构

## 📁 目录说明

```
outputs/
├── 1_persona_modeling/          # Stage 1: Persona建模
│   ├── cleaned_survey_data.csv        # 清洗后的问卷数据
│   ├── preference_factors.csv         # 偏好因子得分
│   ├── factor_loadings.csv            # 因子载荷矩阵
│   ├── factor_loadings_heatmap.png    # 因子载荷热力图
│   ├── personas.json                  # 生成的personas
│   ├── persona_clustering.png         # 聚类可视化
│   ├── cluster_distributions.png      # 聚类分布
│   ├── persona_types.json             # Persona类型定义
│   └── cluster_statistics.json        # 聚类统计
│
├── 2_data_construction/         # Stage 2: 训练数据构造
│   ├── scenarios.json                 # 生成的场景
│   ├── scenario_statistics.json       # 场景统计
│   ├── decisions.json                 # 模拟的决策
│   ├── decision_statistics.json       # 决策统计
│   ├── train_samples.jsonl            # 训练样本
│   ├── validation_samples.jsonl       # 验证样本
│   └── sample_statistics.json         # 样本统计
│
├── 3_training/                  # Stage 3: 模型训练
│   ├── runs/                          # TensorBoard日志
│   │   └── run_YYYYMMDD_HHMMSS/      # 按时间戳组织的训练运行
│   └── training_curves.png            # 训练曲线可视化
│
├── 4_evaluation/                # Stage 4: 模型评估
│   ├── baseline/                      # Baseline模型评估
│   │   ├── baseline_results_*.json
│   │   └── baseline_test.json
│   ├── finetuned/                     # 微调模型评估
│   │   └── checkpoint-*_results.json
│   └── evaluation_results.json        # 最新评估结果
│
└── logs/                        # 运行日志
    ├── persona_modeling_*.log         # Persona建模日志
    ├── data_construction_*.log        # 数据构造日志
    ├── training_*.log                 # 训练日志
    └── evaluation_*.log               # 评估日志
```

## 🔄 文件命名规范

### 时间戳格式
- 日志文件: `MODULE_YYYYMMDD_HHMMSS.log`
- 评估结果: `MODEL_YYYYMMDD_HHMMSS.json`
- 训练运行: `run_YYYYMMDD_HHMMSS/`

### 示例
```
logs/training_20241210_205243.log
4_evaluation/finetuned/checkpoint-2000_20241210_204530.json
3_training/runs/run_20241210_205243/
```

## 📝 使用说明

1. **Stage 1-2**: 输出文件自动保存到对应目录
2. **Stage 3**: 训练日志按时间戳自动创建新目录
3. **Stage 4**: 评估结果按模型类型分类保存

## 🔧 配置文件

主要配置文件：`configs/training_config.yaml`

关键路径设置：
```yaml
paths:
  persona_output_dir: "outputs/1_persona_modeling"
  data_output_dir: "outputs/2_data_construction"
  training_output_dir: "outputs/3_training"
  eval_output_dir: "outputs/4_evaluation"
  logs_dir: "outputs/logs"
```

---

生成时间: {timestamp}
    """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    tree_file.write_text(content)
    print(f"\n✓ 创建目录结构文档: {tree_file}")


def main():
    print("=" * 80)
    print("重新组织项目目录结构")
    print("=" * 80)

    # 1. 创建目录结构
    print("\n[1/4] 创建新目录结构...")
    base_dir = create_organized_structure()

    # 2. 移动文件
    print("\n[2/4] 移动现有文件到对应目录...")
    move_files(base_dir)

    # 3. 整理日志
    print("\n[3/4] 整理日志文件...")
    organize_logs(base_dir)

    # 4. 创建文档
    print("\n[4/4] 创建目录结构文档...")
    create_directory_tree(base_dir)

    print("\n" + "=" * 80)
    print("✓ 目录重组完成！")
    print("=" * 80)
    print("\n查看新的目录结构:")
    print("  cat outputs/DIRECTORY_STRUCTURE.md")
    print("\n或者:")
    print("  tree outputs/ -L 2")


if __name__ == "__main__":
    main()

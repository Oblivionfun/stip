#!/usr/bin/env python3
"""
一键运行阶段1：Persona建模

用法：
    python run_stage1.py
    python run_stage1.py --skip-data-loading  # 如果已有清洗数据
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse


def print_banner(text):
    """打印横幅"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def run_script(script_name, script_path):
    """运行Python脚本"""
    print(f"[运行] {script_name}...")
    print(f"  脚本: {script_path}")
    print("-" * 80)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )

        elapsed = time.time() - start_time
        print("-" * 80)
        print(f"✓ {script_name} 完成 (用时: {elapsed:.2f}秒)\n")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print("-" * 80)
        print(f"✗ {script_name} 失败 (用时: {elapsed:.2f}秒)")
        print(f"错误码: {e.returncode}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='运行阶段1：Persona建模')
    parser.add_argument(
        '--skip-data-loading',
        action='store_true',
        help='跳过数据加载步骤（如果已有cleaned_survey_data.csv）'
    )
    parser.add_argument(
        '--skip-factor-analysis',
        action='store_true',
        help='跳过因子分析步骤（如果已有preference_factors.csv）'
    )
    args = parser.parse_args()

    print_banner("阶段1：Persona建模 - 一键运行")

    print("包含以下步骤：")
    stages = [
        ("1.1", "数据预处理", "src/preference_modeling/data_loader.py", args.skip_data_loading),
        ("1.2", "偏好因子提取", "src/preference_modeling/factor_analysis.py", args.skip_factor_analysis),
        ("1.3", "Persona生成", "src/preference_modeling/persona_generator.py", False),
        ("1.4", "Persona聚类", "src/preference_modeling/persona_clustering.py", False),
    ]

    for stage_num, stage_name, _, skip in stages:
        status = "跳过" if skip else "执行"
        print(f"  [{status}] 阶段{stage_num}: {stage_name}")

    print("\n按Enter键开始...")
    input()

    # 记录总开始时间
    total_start = time.time()

    # 依次运行每个阶段
    for stage_num, stage_name, script_path, skip in stages:
        if skip:
            print(f"\n[跳过] 阶段{stage_num}: {stage_name}")
            continue

        print_banner(f"阶段{stage_num}: {stage_name}")

        success = run_script(f"阶段{stage_num}", script_path)

        if not success:
            print_banner(f"阶段1失败于步骤{stage_num}")
            print(f"\n请检查日志文件：outputs/logs/")
            print(f"修复错误后，可以使用 --skip-xxx 参数跳过已完成的步骤\n")
            sys.exit(1)

    # 总结
    total_elapsed = time.time() - total_start

    print_banner("✓ 阶段1 全部完成！")

    print("生成的文件：")
    output_files = [
        "outputs/cleaned_survey_data.csv",
        "outputs/preference_factors.csv",
        "outputs/factor_loadings.csv",
        "outputs/factor_loadings_heatmap.png",
        "outputs/personas.json",
        "outputs/persona_types.json",
        "outputs/persona_clustering.png",
        "outputs/cluster_distributions.png",
        "outputs/cluster_statistics.json",
    ]

    for file_path in output_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / 1024  # KB
            print(f"  ✓ {file_path} ({size:.1f} KB)")
        else:
            print(f"  ✗ {file_path} (未生成)")

    print(f"\n总用时: {total_elapsed:.2f}秒")
    print(f"日志位置: outputs/logs/")

    print("\n" + "=" * 80)
    print("下一步：运行阶段2（训练数据构造）".center(80))
    print("命令：python run_stage2.py".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

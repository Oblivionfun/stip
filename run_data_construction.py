#!/usr/bin/env python3
"""
一键运行阶段2：训练数据构造

用法：
    python run_stage2.py
    python run_stage2.py --skip-scenarios  # 如果已有场景数据
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
    parser = argparse.ArgumentParser(description='运行阶段2：训练数据构造')
    parser.add_argument(
        '--skip-scenarios',
        action='store_true',
        help='跳过场景生成步骤（如果已有scenarios.json）'
    )
    parser.add_argument(
        '--skip-decisions',
        action='store_true',
        help='跳过决策模拟步骤（如果已有decisions.json）'
    )
    args = parser.parse_args()

    print_banner("阶段2：训练数据构造 - 一键运行")

    print("包含以下步骤：")
    stages = [
        ("2.1", "场景生成", "src/data_construction/scenario_generator.py", args.skip_scenarios),
        ("2.2", "决策模拟", "src/data_construction/decision_simulator.py", args.skip_decisions),
        ("2.3", "样本构造", "src/data_construction/sample_builder.py", False),
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
            print_banner(f"阶段2失败于步骤{stage_num}")
            print(f"\n请检查日志文件：outputs/logs/")
            print(f"修复错误后，可以使用 --skip-xxx 参数跳过已完成的步骤\n")
            sys.exit(1)

    # 总结
    total_elapsed = time.time() - total_start

    print_banner("✓ 阶段2 全部完成！")

    print("生成的文件：")
    output_files = [
        "outputs/scenarios.json",
        "outputs/scenario_statistics.json",
        "outputs/decisions.json",
        "outputs/decision_statistics.json",
        "outputs/train_samples.jsonl",
        "outputs/validation_samples.jsonl",
        "outputs/sample_statistics.json",
    ]

    for file_path in output_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {file_path} ({size:.2f} MB)")
        else:
            print(f"  ✗ {file_path} (未生成)")

    print(f"\n总用时: {total_elapsed:.2f}秒")
    print(f"日志位置: outputs/logs/")

    # 显示数据统计
    print("\n" + "=" * 80)
    print("数据统计")
    print("=" * 80)

    import json
    try:
        with open("outputs/sample_statistics.json", 'r') as f:
            stats = json.load(f)

        print(f"\n训练样本统计：")
        print(f"  - 总样本数: {stats['total_samples']}")
        print(f"  - 平均prompt长度: {stats['avg_prompt_length']} 词")
        print(f"  - 平均reflection长度: {stats['avg_reflection_length']} 词")
        print(f"  - plan分布: {stats['plan_distribution']}")

        # 读取决策统计
        with open("outputs/decision_statistics.json", 'r') as f:
            decision_stats = json.load(f)

        print(f"\n决策统计：")
        print(f"  - 改道率: {decision_stats['reroute_rate']}%")
        print(f"  - 改道决策: {decision_stats['reroute_count']}")
        print(f"  - 保持原路径: {decision_stats['keep_route_count']}")

    except Exception as e:
        print(f"无法读取统计信息: {e}")

    print("\n" + "=" * 80)
    print("下一步：运行阶段3（模型微调）".center(80))
    print("命令：python run_stage3.py".center(80))
    print("==" * 80 + "\n")


if __name__ == "__main__":
    main()

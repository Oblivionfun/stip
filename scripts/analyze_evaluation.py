#!/usr/bin/env python3
"""
评估结果分析脚本
用于查看和对比baseline vs 微调模型的性能
"""

import json
import sys
from pathlib import Path


def load_results(filepath):
    """加载评估结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_metrics(results, title="评估结果"):
    """打印评估指标"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    metrics = results['decision_metrics']

    print(f"\n📊 决策准确率指标:")
    print(f"  • 路径选择准确率: {metrics['route_selection_accuracy']*100:.2f}%")
    print(f"  • 改道决策F1分数: {metrics['reroute_f1']:.4f}")
    print(f"    - Precision: {metrics['reroute_precision']:.4f}")
    print(f"    - Recall: {metrics['reroute_recall']:.4f}")

    print(f"\n📝 语言模型指标:")
    print(f"  • Perplexity: {results['perplexity']:.2f}")

    print(f"\n📈 详细统计:")
    print(f"  • 总样本数: {metrics['total_samples']}")
    print(f"  • 正确路径选择: {metrics['correct_route']}")
    print(f"  • 改道TP (正确改道): {metrics['reroute_tp']}")
    print(f"  • 改道FP (错误改道): {metrics['reroute_fp']}")
    print(f"  • 改道FN (应改未改): {metrics['reroute_fn']}")

    # 如果有预测样例
    if 'predictions' in results and len(results['predictions']) > 0:
        print(f"\n🔍 预测样例 (前3个):")
        for i, pred in enumerate(results['predictions'][:3], 1):
            print(f"\n  [{i}] 预测:")
            print(f"      {json.dumps(pred, ensure_ascii=False, indent=6)}")

    print("\n" + "=" * 80)


def compare_results(baseline_path, finetuned_path):
    """对比baseline和微调后的结果"""
    baseline = load_results(baseline_path)
    finetuned = load_results(finetuned_path)

    print("\n" + "=" * 80)
    print("📊 性能对比：Baseline vs 微调后")
    print("=" * 80)

    b_metrics = baseline['decision_metrics']
    f_metrics = finetuned['decision_metrics']

    # 路径选择准确率
    b_acc = b_metrics['route_selection_accuracy'] * 100
    f_acc = f_metrics['route_selection_accuracy'] * 100
    acc_improvement = f_acc - b_acc

    print(f"\n🎯 路径选择准确率:")
    print(f"  Baseline:    {b_acc:6.2f}%")
    print(f"  微调后:      {f_acc:6.2f}%")
    print(f"  提升:        {acc_improvement:+6.2f}% {'✅' if acc_improvement > 0 else '❌'}")

    # 改道F1
    b_f1 = b_metrics['reroute_f1']
    f_f1 = f_metrics['reroute_f1']
    f1_improvement = f_f1 - b_f1

    print(f"\n🔄 改道F1分数:")
    print(f"  Baseline:    {b_f1:.4f}")
    print(f"  微调后:      {f_f1:.4f}")
    print(f"  提升:        {f1_improvement:+.4f} {'✅' if f1_improvement > 0 else '❌'}")

    # Perplexity
    b_ppl = baseline['perplexity']
    f_ppl = finetuned['perplexity']
    ppl_change = f_ppl - b_ppl

    print(f"\n📝 Perplexity (越低越好):")
    print(f"  Baseline:    {b_ppl:.2f}")
    print(f"  微调后:      {f_ppl:.2f}")
    print(f"  变化:        {ppl_change:+.2f} {'✅' if ppl_change < 0 else '❌'}")

    # 目标达成情况
    print(f"\n🎯 目标达成情况:")
    print(f"  路径准确率 ≥75%:  {'✅ 达成' if f_acc >= 75 else '❌ 未达成'} ({f_acc:.1f}%)")
    print(f"  改道F1 ≥0.80:      {'✅ 达成' if f_f1 >= 0.80 else '❌ 未达成'} ({f_f1:.3f})")
    print(f"  Perplexity <10:    {'✅ 达成' if f_ppl < 10 else '❌ 未达成'} ({f_ppl:.2f})")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) == 1:
        # 只查看baseline
        baseline_file = "outputs/baseline_results.json"
        if Path(baseline_file).exists():
            results = load_results(baseline_file)
            print_metrics(results, "Baseline 评估结果 (未微调)")
        else:
            print(f"❌ 文件不存在: {baseline_file}")
            print("\n使用方法:")
            print("  python analyze_evaluation.py                    # 查看baseline")
            print("  python analyze_evaluation.py <result_file>      # 查看指定结果")
            print("  python analyze_evaluation.py compare            # 对比baseline和微调后")

    elif len(sys.argv) == 2:
        arg = sys.argv[1]

        if arg == "compare":
            # 对比模式
            baseline_file = "outputs/baseline_results.json"
            finetuned_file = "outputs/finetuned_results.json"

            if not Path(baseline_file).exists():
                print(f"❌ Baseline文件不存在: {baseline_file}")
                return

            if not Path(finetuned_file).exists():
                print(f"❌ 微调结果文件不存在: {finetuned_file}")
                print("   请先运行训练和评估")
                return

            compare_results(baseline_file, finetuned_file)
        else:
            # 查看指定文件
            if Path(arg).exists():
                results = load_results(arg)
                print_metrics(results)
            else:
                print(f"❌ 文件不存在: {arg}")


if __name__ == "__main__":
    main()

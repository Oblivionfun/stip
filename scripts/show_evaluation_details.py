#!/usr/bin/env python3
"""
显示评估的详细结果，包括thinking过程和决策统计
"""

import json
import sys
from pathlib import Path
from collections import Counter


def load_results(filepath):
    """加载评估结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_predictions(results):
    """分析预测结果"""

    predictions = results.get('predictions', [])
    ground_truth = results.get('ground_truth', [])

    if not predictions:
        print("❌ 评估结果中没有保存predictions")
        return

    # 统计决策分布
    pred_plans = []
    gt_plans = []

    for pred in predictions:
        if pred:
            plan = pred.get('plan', '')
            pred_plans.append('改道' if 'update path' in plan.lower() else '不改道')

    for gt in ground_truth:
        if gt and 'response' in gt:
            plan = gt['response'].get('plan', '')
            gt_plans.append('改道' if 'update path' in plan.lower() else '不改道')

    print("\n" + "=" * 80)
    print("决策分布统计")
    print("=" * 80)

    print(f"\n📊 预测决策分布 (前{len(pred_plans)}个样本):")
    pred_counter = Counter(pred_plans)
    for decision, count in pred_counter.items():
        pct = count / len(pred_plans) * 100 if pred_plans else 0
        print(f"  • {decision}: {count} ({pct:.1f}%)")

    print(f"\n✅ Ground Truth决策分布 (前{len(gt_plans)}个样本):")
    gt_counter = Counter(gt_plans)
    for decision, count in gt_counter.items():
        pct = count / len(gt_plans) * 100 if gt_plans else 0
        print(f"  • {decision}: {count} ({pct:.1f}%)")

    # 显示thinking示例
    print("\n" + "=" * 80)
    print("Thinking 过程示例 (前3个)")
    print("=" * 80)

    valid_preds = [p for p in predictions if p is not None]

    for i, (pred, gt) in enumerate(zip(valid_preds[:3], ground_truth[:3]), 1):
        print(f"\n{'='*80}")
        print(f"样本 {i}: {gt.get('id', 'N/A')}")
        print(f"{'='*80}")

        # Ground Truth
        gt_resp = gt.get('response', {})
        gt_thinking = gt_resp.get('thinking', 'N/A')
        gt_reflection = gt_resp.get('reflection', 'N/A')
        gt_plan = gt_resp.get('plan', 'N/A')

        print(f"\n✅ Ground Truth:")
        print(f"   Thinking (前150字符):")
        print(f"   {gt_thinking[:150]}...")
        print(f"\n   Reflection (前100字符):")
        print(f"   {gt_reflection[:100]}...")
        print(f"\n   Plan: {gt_plan}")

        # Prediction
        pred_thinking = pred.get('thinking', 'N/A')
        pred_reflection = pred.get('reflection', 'N/A')
        pred_plan = pred.get('plan', 'N/A')

        print(f"\n🤖 模型预测:")
        print(f"   Thinking (前150字符):")
        print(f"   {pred_thinking[:150]}...")
        print(f"\n   Reflection (前100字符):")
        print(f"   {pred_reflection[:100]}...")
        print(f"\n   Plan: {pred_plan}")

        # 对比
        match = '✅ 匹配' if gt_plan == pred_plan else '❌ 不匹配'
        print(f"\n   决策对比: {match}")

    print("\n" + "=" * 80)

    # 计算thinking字段的覆盖率
    pred_with_thinking = sum(1 for p in predictions if p and p.get('thinking'))
    total_pred = len([p for p in predictions if p])

    print(f"\n📊 Thinking字段统计:")
    print(f"  • 包含thinking的预测: {pred_with_thinking}/{total_pred}")
    if total_pred > 0:
        print(f"  • 覆盖率: {pred_with_thinking/total_pred*100:.1f}%")


def main():
    if len(sys.argv) < 2:
        result_file = "outputs/evaluation_results.json"
    else:
        result_file = sys.argv[1]

    if not Path(result_file).exists():
        print(f"❌ 文件不存在: {result_file}")
        print("\n使用方法:")
        print(f"  python {sys.argv[0]} [结果文件路径]")
        print(f"  默认: {result_file}")
        return

    print("=" * 80)
    print(f"分析评估结果: {result_file}")
    print("=" * 80)

    results = load_results(result_file)

    # 显示整体指标
    print(f"\n📊 整体性能:")
    metrics = results.get('decision_metrics', {})
    print(f"  • 路径选择准确率: {metrics.get('route_selection_accuracy', 0)*100:.2f}%")
    print(f"  • 改道F1分数: {metrics.get('reroute_f1', 0):.4f}")
    print(f"  • Perplexity: {results.get('perplexity', 0):.2f}")
    print(f"  • 总样本数: {results.get('num_samples', 0)}")

    # 分析predictions
    analyze_predictions(results)

    print("\n✓ 分析完成")


if __name__ == "__main__":
    main()

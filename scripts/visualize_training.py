#!/usr/bin/env python3
"""
可视化训练历史（从checkpoint的trainer_state.json）
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_training_history(checkpoint_path):
    """从checkpoint加载训练历史"""
    trainer_state_file = Path(checkpoint_path) / "trainer_state.json"

    if not trainer_state_file.exists():
        raise FileNotFoundError(f"找不到 {trainer_state_file}")

    with open(trainer_state_file, 'r') as f:
        data = json.load(f)

    return data['log_history']


def plot_training_curves(log_history, output_path=None):
    """绘制训练曲线"""

    # 提取数据
    train_steps = []
    train_losses = []
    learning_rates = []
    grad_norms = []

    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            # 训练loss
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])

            if 'learning_rate' in entry:
                learning_rates.append(entry['learning_rate'])

            if 'grad_norm' in entry:
                grad_norms.append(entry['grad_norm'])

        elif 'eval_loss' in entry and 'step' in entry:
            # 评估loss
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # 1. 训练Loss
    ax1 = axes[0, 0]
    ax1.plot(train_steps, train_losses, 'b-', linewidth=2, label='Train Loss')
    if eval_losses:
        ax1.plot(eval_steps, eval_losses, 'r--', linewidth=2, label='Eval Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Evaluation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 学习率
    ax2 = axes[0, 1]
    if learning_rates:
        ax2.plot(train_steps[:len(learning_rates)], learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # 3. 梯度范数
    ax3 = axes[1, 0]
    if grad_norms:
        ax3.plot(train_steps[:len(grad_norms)], grad_norms, 'orange', linewidth=2)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Norm')
        ax3.grid(True, alpha=0.3)

    # 4. Loss趋势（滑动平均）
    ax4 = axes[1, 1]
    if len(train_losses) > 10:
        window_size = min(50, len(train_losses) // 10)
        smoothed_loss = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = train_steps[window_size-1:]
        ax4.plot(train_steps, train_losses, 'b-', alpha=0.3, label='Raw')
        ax4.plot(smoothed_steps, smoothed_loss, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Loss')
        ax4.set_title('Loss Trend (Smoothed)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存到: {output_path}")

    return fig


def print_statistics(log_history):
    """打印训练统计信息"""
    train_losses = [entry['loss'] for entry in log_history if 'loss' in entry]

    if not train_losses:
        print("❌ 没有找到训练loss数据")
        return

    print("\n" + "=" * 80)
    print("训练统计")
    print("=" * 80)

    print(f"\n📊 Loss统计:")
    print(f"  • 初始Loss: {train_losses[0]:.4f}")
    print(f"  • 最终Loss: {train_losses[-1]:.4f}")
    print(f"  • 最低Loss: {min(train_losses):.4f}")
    print(f"  • Loss下降: {train_losses[0] - train_losses[-1]:.4f} ({(1 - train_losses[-1]/train_losses[0])*100:.1f}%)")

    # 最近N步的平均loss
    recent_n = min(100, len(train_losses) // 10)
    recent_avg = np.mean(train_losses[-recent_n:])
    print(f"  • 最近{recent_n}步平均Loss: {recent_avg:.4f}")

    # 训练进度
    total_steps = log_history[-1].get('step', 0)
    epochs = log_history[-1].get('epoch', 0)

    print(f"\n📈 训练进度:")
    print(f"  • 总步数: {total_steps}")
    print(f"  • 训练轮数: {epochs:.2f} epochs")

    # 评估结果
    eval_entries = [entry for entry in log_history if 'eval_loss' in entry]
    if eval_entries:
        print(f"\n🎯 评估结果:")
        print(f"  • 评估次数: {len(eval_entries)}")
        eval_losses = [entry['eval_loss'] for entry in eval_entries]
        print(f"  • 最佳Eval Loss: {min(eval_losses):.4f}")
        print(f"  • 最新Eval Loss: {eval_losses[-1]:.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="可视化训练历史")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/sft_model/checkpoint-2000',
        help='Checkpoint路径（默认：checkpoint-2000）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/training_curves.png',
        help='输出图片路径'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='显示图表（需要GUI环境）'
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"加载训练历史: {args.checkpoint}")
    print("=" * 80)

    try:
        # 加载训练历史
        log_history = load_training_history(args.checkpoint)

        # 打印统计信息
        print_statistics(log_history)

        # 绘制曲线
        fig = plot_training_curves(log_history, args.output)

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        print(f"\n✓ 完成！")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

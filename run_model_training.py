#!/usr/bin/env python3
"""
一键运行阶段3：大模型微调

用法：
    python run_stage3.py
    python run_stage3.py --use-unsloth  # 使用unsloth加速（需要GPU和unsloth库）
    python run_stage3.py --no-unsloth   # 使用标准Transformers

注意：
    - 训练需要GPU支持
    - 建议至少16GB GPU显存
    - 使用4bit量化可以减少显存需求
    - 训练时间取决于GPU性能，大约需要数小时
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.training.sft_trainer import SFTTrainer


def print_banner(text):
    """打印横幅"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_system_info():
    """打印系统信息"""
    import torch

    print("系统信息:")
    print(f"  - Python: {sys.version.split()[0]}")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"      Memory: {mem_gb:.2f} GB")
    else:
        print("  ⚠ WARNING: No CUDA available, training will be very slow!")


def check_dependencies():
    """检查依赖"""
    missing = []

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import datasets
    except ImportError:
        missing.append("datasets")

    try:
        import peft
    except ImportError:
        missing.append("peft")

    try:
        import accelerate
    except ImportError:
        missing.append("accelerate")

    try:
        import bitsandbytes
    except ImportError:
        missing.append("bitsandbytes (needed for 4bit)")

    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='运行阶段3：大模型微调')
    parser.add_argument(
        '--use-unsloth',
        action='store_true',
        help='使用unsloth加速（默认尝试使用）'
    )
    parser.add_argument(
        '--no-unsloth',
        action='store_true',
        help='不使用unsloth，使用标准Transformers'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='跳过确认提示，直接开始训练'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从checkpoint恢复训练（如：checkpoints/sft_model/checkpoint-2000）'
    )
    args = parser.parse_args()

    print_banner("阶段3：大模型微调")

    # 打印系统信息
    print_system_info()
    print()

    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺失的依赖")
        sys.exit(1)

    # 确定是否使用unsloth
    use_unsloth = not args.no_unsloth

    print("训练配置:")
    print(f"  - 使用unsloth加速: {'是' if use_unsloth else '否'}")
    print(f"  - 配置文件: configs/training_config.yaml")
    print(f"  - 模型路径: model/models")
    if args.resume:
        print(f"  - 从checkpoint恢复: {args.resume}")
    print(f"  - 训练数据: outputs/train_samples.jsonl (9,018样本)")
    print(f"  - 验证数据: outputs/validation_samples.jsonl (1,002样本)")
    print(f"  - 输出目录: checkpoints/sft_model")

    print("\n" + "=" * 80)
    print("训练将开始，这可能需要数小时...".center(80))
    print("建议在tmux或screen会话中运行，避免中断".center(80))
    print("=" * 80)

    if not args.yes:
        print("\n按Enter键开始训练...")
        input()
    else:
        print("\n自动开始训练...\n")

    try:
        # 创建带时间戳的训练运行目录
        from src.utils.path_utils import get_training_run_dir, get_log_path
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"run_{timestamp}"

        # 初始化训练器
        print_banner("初始化训练器")
        print(f"训练运行: {run_name}")
        print(f"TensorBoard日志: outputs/3_training/runs/{run_name}\n")

        trainer = SFTTrainer(use_unsloth=use_unsloth, run_name=run_name)

        # 运行训练
        print_banner("开始训练")
        train_result, eval_metrics = trainer.run(resume_from_checkpoint=args.resume)

        # 总结
        print_banner("✓ 训练完成！")

        print("训练指标:")
        print(f"  - Loss: {train_result.training_loss:.4f}")

        print("\n评估指标:")
        for key, value in eval_metrics.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")

        print(f"\n模型保存位置: {trainer.config['training']['output_dir']}")
        print(f"可以使用此模型进行推理和评估")

        print("\n" + "=" * 80)
        print("下一步：运行阶段4（决策智能体API）".center(80))
        print("命令：python run_stage4.py".center(80))
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\n训练被中断")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

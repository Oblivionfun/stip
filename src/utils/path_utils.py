"""
路径和时间戳工具
提供统一的路径管理和时间戳生成
"""

from pathlib import Path
from datetime import datetime
import os


def get_timestamp(format_type="full"):
    """
    生成时间戳

    Args:
        format_type: 时间戳格式类型
            - "full": YYYYMMDD_HHMMSS (默认)
            - "date": YYYYMMDD
            - "readable": YYYY-MM-DD_HH-MM-SS
            - "compact": YYYYMMDDHHMMSS

    Returns:
        时间戳字符串
    """
    now = datetime.now()

    formats = {
        "full": "%Y%m%d_%H%M%S",
        "date": "%Y%m%d",
        "readable": "%Y-%m-%D_%H-%M-%S",
        "compact": "%Y%m%d%H%M%S",
    }

    return now.strftime(formats.get(format_type, formats["full"]))


def get_log_path(module_name, logs_dir="outputs/logs"):
    """
    生成带时间戳的日志路径

    Args:
        module_name: 模块名称 (如 "training", "evaluation")
        logs_dir: 日志目录

    Returns:
        日志文件路径 (Path对象)
    """
    timestamp = get_timestamp()
    log_filename = f"{module_name}_{timestamp}.log"

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    return logs_path / log_filename


def get_output_path(filename, subdir, output_base="outputs", add_timestamp=True):
    """
    生成输出文件路径

    Args:
        filename: 文件名
        subdir: 子目录 (如 "4_evaluation/finetuned")
        output_base: 输出基础目录
        add_timestamp: 是否添加时间戳

    Returns:
        输出文件路径 (Path对象)
    """
    output_dir = Path(output_base) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if add_timestamp and not has_timestamp(filename):
        name, ext = os.path.splitext(filename)
        timestamp = get_timestamp()
        filename = f"{name}_{timestamp}{ext}"

    return output_dir / filename


def get_training_run_dir(run_name=None, base_dir="outputs/3_training/runs"):
    """
    生成训练运行目录

    Args:
        run_name: 运行名称 (可选，默认使用时间戳)
        base_dir: 基础目录

    Returns:
        训练运行目录路径 (Path对象)
    """
    if run_name is None:
        timestamp = get_timestamp()
        run_name = f"run_{timestamp}"

    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def has_timestamp(filename):
    """
    检查文件名是否已包含时间戳

    Args:
        filename: 文件名

    Returns:
        bool: 是否包含时间戳
    """
    import re

    # 匹配 YYYYMMDD_HHMMSS 格式
    pattern = r'\d{8}_\d{6}'
    return bool(re.search(pattern, filename))


def get_latest_file(directory, pattern="*"):
    """
    获取目录中最新的文件

    Args:
        directory: 目录路径
        pattern: 文件模式 (glob格式)

    Returns:
        最新文件路径 (Path对象) 或 None
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        return None

    files = list(dir_path.glob(pattern))

    if not files:
        return None

    # 按修改时间排序，返回最新的
    return max(files, key=lambda p: p.stat().st_mtime)


def ensure_output_dirs():
    """
    确保所有输出目录存在
    """
    directories = [
        "outputs/1_persona_modeling",
        "outputs/2_data_construction",
        "outputs/3_training/runs",
        "outputs/4_evaluation/baseline",
        "outputs/4_evaluation/finetuned",
        "outputs/logs",
        "checkpoints",
        ".cache",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# 路径常量
class OutputPaths:
    """输出路径常量"""

    BASE = Path("outputs")

    # Stage outputs
    PERSONA = BASE / "1_persona_modeling"
    DATA = BASE / "2_data_construction"
    TRAINING = BASE / "3_training"
    EVALUATION = BASE / "4_evaluation"

    # Evaluation subdirs
    EVAL_BASELINE = EVALUATION / "baseline"
    EVAL_FINETUNED = EVALUATION / "finetuned"

    # Logs
    LOGS = BASE / "logs"

    # Training runs
    TRAINING_RUNS = TRAINING / "runs"

    # Checkpoints
    CHECKPOINTS = Path("checkpoints")


if __name__ == "__main__":
    # 测试
    print("时间戳测试:")
    print(f"  Full: {get_timestamp('full')}")
    print(f"  Date: {get_timestamp('date')}")
    print(f"  Compact: {get_timestamp('compact')}")

    print("\n路径生成测试:")
    print(f"  Log path: {get_log_path('training')}")
    print(f"  Output path: {get_output_path('results.json', '4_evaluation/finetuned')}")
    print(f"  Training run: {get_training_run_dir()}")

    print("\n确保输出目录存在...")
    ensure_output_dirs()
    print("  ✓ 完成")

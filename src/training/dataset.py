"""
阶段3.1：数据集加载器
功能：
1. 加载JSONL格式的训练数据
2. 格式化为Qwen3的chat格式
3. 支持completion-only训练
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets import Dataset
from transformers import PreTrainedTokenizer
from src.utils.common import load_config, setup_logger


class RouteDecisionDataset:
    """路径决策数据集加载器"""

    def __init__(
        self,
        train_config_path: str = "configs/training_config.yaml"
    ):
        self.config = load_config(train_config_path)
        self.logger = setup_logger(
            'RouteDecisionDataset',
            f"{self.config['paths']['logs_dir']}/dataset.log"
        )
        self.data_config = self.config['data']
        self.prompt_config = self.config['prompt_format']

    def load_jsonl(self, file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        加载JSONL文件

        Args:
            file_path: JSONL文件路径
            max_samples: 最大样本数（None表示全部加载）

        Returns:
            样本列表
        """
        self.logger.info(f"Loading data from {file_path}...")

        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break

                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line {idx+1}: {e}")

        self.logger.info(f"Loaded {len(samples)} samples")
        return samples

    def format_sample(self, sample: Dict) -> Dict:
        """
        格式化单个样本为Chat格式

        Args:
            sample: 原始样本 {"prompt": "...", "response": {"reflection": "...", "plan": "..."}}

        Returns:
            格式化后的样本 {"text": "<formatted_chat>"}
        """
        # 提取prompt和response
        prompt = sample['prompt']
        response = sample['response']

        # 将response转换为JSON字符串
        response_json = json.dumps(response, ensure_ascii=False, indent=2)

        # 使用chat_template格式化
        chat_template = self.prompt_config['chat_template']
        formatted_text = chat_template.format(
            prompt=prompt,
            response_json=response_json
        )

        return {
            "text": formatted_text,
            "id": sample.get('id', ''),
            "scenario_id": sample.get('scenario_id', '')
        }

    def create_dataset(
        self,
        file_path: str,
        max_samples: Optional[int] = None,
        shuffle: bool = True
    ) -> Dataset:
        """
        创建Hugging Face Dataset

        Args:
            file_path: JSONL文件路径
            max_samples: 最大样本数
            shuffle: 是否打乱

        Returns:
            Dataset对象
        """
        # 加载数据
        samples = self.load_jsonl(file_path, max_samples)

        # 格式化
        self.logger.info("Formatting samples...")
        formatted_samples = [self.format_sample(s) for s in samples]

        # 创建Dataset
        dataset = Dataset.from_list(formatted_samples)

        # 打乱
        if shuffle:
            dataset = dataset.shuffle(seed=self.config['training']['seed'])

        self.logger.info(f"Created dataset with {len(dataset)} samples")

        return dataset

    def load_train_val_datasets(self) -> tuple:
        """
        加载训练集和验证集

        Returns:
            (train_dataset, val_dataset)
        """
        max_samples = self.data_config.get('max_samples')
        shuffle = self.data_config.get('shuffle', True)

        # 加载训练集
        train_dataset = self.create_dataset(
            self.data_config['train_file'],
            max_samples=max_samples,
            shuffle=shuffle
        )

        # 加载验证集
        val_dataset = self.create_dataset(
            self.data_config['val_file'],
            max_samples=None,  # 验证集使用全部数据
            shuffle=False  # 验证集不打乱
        )

        self.logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

        return train_dataset, val_dataset

    def get_completion_only_collator(self, tokenizer: PreTrainedTokenizer):
        """
        获取Completion-only数据collator

        Args:
            tokenizer: 分词器

        Returns:
            DataCollatorForCompletionOnlyLM
        """
        if not self.config['completion_only']['enabled']:
            return None

        try:
            from trl import DataCollatorForCompletionOnlyLM
        except ImportError:
            self.logger.warning("trl not installed, falling back to standard collator")
            return None

        response_template = self.config['completion_only']['response_template']

        # 找到response_template在tokenizer中的token ids
        response_template_ids = tokenizer.encode(
            response_template,
            add_special_tokens=False
        )

        self.logger.info(f"Response template: '{response_template}'")
        self.logger.info(f"Response template token IDs: {response_template_ids}")

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer
        )

        return collator


def main():
    """测试数据集加载"""

    print("=" * 80)
    print("测试数据集加载器")
    print("=" * 80)

    # 初始化加载器
    dataset_loader = RouteDecisionDataset()

    # 加载数据集
    print("\n[1/2] 加载训练集和验证集...")
    train_dataset, val_dataset = dataset_loader.load_train_val_datasets()

    print(f"\n✓ 数据集加载完成！")
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 验证集: {len(val_dataset)} 样本")

    # 展示示例
    print("\n" + "=" * 80)
    print("训练样本示例")
    print("=" * 80)

    sample = train_dataset[0]
    print(f"\n[ID] {sample['id']}")
    print(f"\n[TEXT]")
    print(sample['text'][:500] + "..." if len(sample['text']) > 500 else sample['text'])

    print("\n" + "=" * 80)
    print("✓ 测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

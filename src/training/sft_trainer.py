"""
阶段3.2：SFT训练器
功能：
1. 使用unsloth（如果可用）或标准Transformers进行模型微调
2. 应用LoRA进行参数高效微调
3. Completion-only训练
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.training.dataset import RouteDecisionDataset
from src.utils.common import load_config, setup_logger, ensure_dir


class SFTTrainer:
    """监督微调训练器"""

    def __init__(
        self,
        training_config_path: str = "configs/training_config.yaml",
        use_unsloth: bool = True
    ):
        self.config = load_config(training_config_path)
        self.logger = setup_logger(
            'SFTTrainer',
            f"{self.config['paths']['logs_dir']}/sft_trainer.log"
        )
        self.use_unsloth = use_unsloth

        # 尝试导入unsloth
        self.unsloth_available = False
        if use_unsloth:
            try:
                from unsloth import FastLanguageModel
                self.FastLanguageModel = FastLanguageModel
                self.unsloth_available = True
                self.logger.info("✓ Unsloth available, will use accelerated training")
            except ImportError:
                self.logger.warning("⚠ Unsloth not available, falling back to standard training")

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        model_config = self.config['model']
        model_name = model_config['model_name']

        self.logger.info(f"Loading model from {model_name}...")

        if self.unsloth_available:
            # 使用unsloth加载（更快）
            self.logger.info("Using Unsloth FastLanguageModel...")

            self.model, self.tokenizer = self.FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=model_config['max_seq_length'],
                dtype=model_config['dtype'],
                load_in_4bit=model_config['load_in_4bit'],
            )

            # 应用LoRA
            self.logger.info("Applying LoRA with Unsloth...")
            self.model = self.FastLanguageModel.get_peft_model(
                self.model,
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                lora_dropout=self.config['lora']['lora_dropout'],
                target_modules=self.config['lora']['target_modules'],
                bias=self.config['lora']['bias'],
                use_gradient_checkpointing=self.config['lora']['use_gradient_checkpointing'],
                random_state=self.config['lora']['random_state'],
                use_rslora=self.config['lora']['use_rslora'],
            )

        else:
            # 使用标准Transformers加载
            self.logger.info("Using standard Transformers...")

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            load_in_4bit = model_config.get('load_in_4bit', False)

            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )

                # 准备模型用于kbit训练
                self.model = prepare_model_for_kbit_training(self.model)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )

            # 应用LoRA
            self.logger.info("Applying LoRA with PEFT...")
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                lora_dropout=self.config['lora']['lora_dropout'],
                target_modules=self.config['lora']['target_modules'],
                bias=self.config['lora']['bias'],
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        self.model.print_trainable_parameters()

        self.logger.info("✓ Model and tokenizer loaded successfully")

    def prepare_datasets(self):
        """准备训练和验证数据集"""
        self.logger.info("Preparing datasets...")

        # 加载数据集
        dataset_loader = RouteDecisionDataset(training_config_path="configs/training_config.yaml")
        train_dataset, val_dataset = dataset_loader.load_train_val_datasets()

        # Tokenize函数
        def tokenize_function(examples):
            # Tokenize文本
            result = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.config['model']['max_seq_length'],
                padding=False,  # 将在collator中动态padding
            )

            # 复制input_ids到labels
            result['labels'] = result['input_ids'].copy()

            return result

        # 应用tokenization
        self.logger.info("Tokenizing train dataset...")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train"
        )

        self.logger.info("Tokenizing validation dataset...")
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val"
        )

        self.logger.info(f"✓ Datasets prepared: Train={len(train_dataset)}, Val={len(val_dataset)}")

        return train_dataset, val_dataset

    def setup_trainer(self, train_dataset, val_dataset):
        """设置Trainer"""
        training_config = self.config['training']

        # 创建输出目录
        ensure_dir(training_config['output_dir'])

        # 训练参数
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],

            # 优化器
            optim=training_config['optim'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            max_grad_norm=training_config['max_grad_norm'],

            # 学习率调度
            lr_scheduler_type=training_config['lr_scheduler_type'],

            # 日志和保存
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            evaluation_strategy=training_config['evaluation_strategy'],
            eval_steps=training_config['eval_steps'],

            # 精度
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],

            # 其他
            report_to=training_config['report_to'],
            seed=training_config['seed'],
            dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
            remove_unused_columns=training_config['remove_unused_columns'],
        )

        # Data collator
        from transformers import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM，不是MLM
        )

        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        self.logger.info("✓ Trainer setup complete")

    def train(self):
        """开始训练"""
        self.logger.info("=" * 80)
        self.logger.info("Starting training...")
        self.logger.info("=" * 80)

        # 训练
        train_result = self.trainer.train()

        # 保存模型
        self.logger.info("Saving final model...")
        self.trainer.save_model()

        # 保存分词器
        self.tokenizer.save_pretrained(self.config['training']['output_dir'])

        # 保存训练指标
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        self.logger.info("=" * 80)
        self.logger.info("✓ Training complete!")
        self.logger.info("=" * 80)

        return train_result

    def evaluate(self):
        """评估模型"""
        self.logger.info("Evaluating model...")

        metrics = self.trainer.evaluate()

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        self.logger.info(f"Eval metrics: {metrics}")

        return metrics

    def run(self):
        """完整的训练流程"""
        # 1. 加载模型
        self.load_model_and_tokenizer()

        # 2. 准备数据集
        train_dataset, val_dataset = self.prepare_datasets()

        # 3. 设置Trainer
        self.setup_trainer(train_dataset, val_dataset)

        # 4. 训练
        train_result = self.train()

        # 5. 评估
        eval_metrics = self.evaluate()

        return train_result, eval_metrics


def main():
    """主函数"""

    print("=" * 80)
    print("阶段3：大模型微调")
    print("=" * 80)

    # 初始化训练器
    print("\n[1/5] 初始化训练器...")
    trainer = SFTTrainer(use_unsloth=True)

    # 运行完整流程
    print("\n[2/5] 运行训练流程...")
    train_result, eval_metrics = trainer.run()

    print("\n" + "=" * 80)
    print("✓ 训练完成！")
    print("=" * 80)

    print(f"\n训练指标:")
    print(f"  - Loss: {train_result.training_loss:.4f}")

    print(f"\n评估指标:")
    for key, value in eval_metrics.items():
        print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")

    print(f"\n模型保存位置: {trainer.config['training']['output_dir']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

"""
模型评估脚本
评估指标：
1. 决策准确率 (Decision Accuracy)
2. 改道决策F1分数 (Reroute F1 Score)
3. 路径选择准确率 (Route Selection Accuracy)
4. Perplexity
5. 偏好一致性 (Preference Consistency)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from src.utils.common import load_config, setup_logger, load_json


class ModelEvaluator:
    """模型评估器"""

    def __init__(
        self,
        model_path: str = "checkpoints/sft_model",
        training_config_path: str = "configs/training_config.yaml"
    ):
        self.model_path = model_path
        self.config = load_config(training_config_path)
        self.logger = setup_logger(
            'ModelEvaluator',
            f"{self.config['paths']['logs_dir']}/evaluator.log"
        )

    def load_model(self):
        """加载微调后的模型"""
        self.logger.info(f"Loading model from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.model.eval()
        self.logger.info("✓ Model loaded")

    def load_ground_truth(self) -> List[Dict]:
        """加载ground truth数据"""
        val_file = self.config['data']['val_file']

        self.logger.info(f"Loading ground truth from {val_file}...")

        samples = []
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)

        self.logger.info(f"Loaded {len(samples)} validation samples")
        return samples

    def parse_model_output(self, output_text: str) -> Dict:
        """
        解析模型输出的JSON

        Returns:
            {"thinking": "...", "reflection": "...", "plan": "...", "concepts": []}
            如果解析失败返回None
        """
        try:
            # 查找JSON部分
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                return None

            json_str = output_text[start_idx:end_idx]
            result = json.loads(json_str)

            return result

        except Exception as e:
            self.logger.warning(f"Failed to parse output: {e}")
            return None

    def predict_single(self, prompt: str) -> Dict:
        """
        对单个prompt进行预测

        Args:
            prompt: 输入prompt

        Returns:
            预测的response dict
        """
        # 构造输入
        messages = [{"role": "user", "content": prompt}]

        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config['inference']['max_new_tokens'],
                temperature=self.config['inference']['temperature'],
                top_p=self.config['inference']['top_p'],
                do_sample=self.config['inference']['do_sample'],
            )

        # 解码
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取assistant部分
        if "assistant" in generated_text:
            assistant_response = generated_text.split("assistant")[-1].strip()
        else:
            assistant_response = generated_text

        # 解析JSON
        parsed = self.parse_model_output(assistant_response)

        return parsed

    def extract_route_choice(self, plan: str) -> str:
        """
        从plan字段提取路径选择

        Args:
            plan: "none" 或 "update path: Route B" 等

        Returns:
            "A" 或 "B" 或 "unknown"
        """
        if plan == "none":
            return "A"  # 保持默认路径A

        if "update path" in plan.lower():
            # 尝试提取路径
            if "route b" in plan.lower() or "ave_2" in plan.lower() or "side_st" in plan.lower():
                return "B"
            elif "route a" in plan.lower() or "ave_1" in plan.lower() or "main_ave" in plan.lower():
                return "A"

        return "unknown"

    def evaluate_decision_accuracy(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """
        评估决策准确率

        指标：
        1. 路径选择准确率
        2. 改道决策F1
        """
        correct_route = 0
        correct_reroute = 0

        # 改道决策的TP/FP/FN
        reroute_tp = 0
        reroute_fp = 0
        reroute_fn = 0

        total = len(predictions)

        for pred, gt in zip(predictions, ground_truths):
            if pred is None:
                continue

            # Ground truth
            gt_response = gt['response']
            gt_plan = gt_response['plan']
            gt_route = self.extract_route_choice(gt_plan)
            gt_reroute = (gt_route != "A")

            # Prediction
            pred_plan = pred.get('plan', '')
            pred_route = self.extract_route_choice(pred_plan)
            pred_reroute = (pred_route != "A")

            # 路径选择准确率
            if pred_route == gt_route and pred_route != "unknown":
                correct_route += 1

            # 改道决策
            if gt_reroute and pred_reroute:
                reroute_tp += 1
            elif pred_reroute and not gt_reroute:
                reroute_fp += 1
            elif gt_reroute and not pred_reroute:
                reroute_fn += 1
            elif not gt_reroute and not pred_reroute:
                correct_reroute += 1

        # 计算指标
        route_accuracy = correct_route / total if total > 0 else 0

        # 改道F1
        reroute_precision = reroute_tp / (reroute_tp + reroute_fp) if (reroute_tp + reroute_fp) > 0 else 0
        reroute_recall = reroute_tp / (reroute_tp + reroute_fn) if (reroute_tp + reroute_fn) > 0 else 0
        reroute_f1 = 2 * (reroute_precision * reroute_recall) / (reroute_precision + reroute_recall) if (reroute_precision + reroute_recall) > 0 else 0

        return {
            'route_selection_accuracy': route_accuracy,
            'reroute_precision': reroute_precision,
            'reroute_recall': reroute_recall,
            'reroute_f1': reroute_f1,
            'total_samples': total,
            'correct_route': correct_route,
            'reroute_tp': reroute_tp,
            'reroute_fp': reroute_fp,
            'reroute_fn': reroute_fn,
        }

    def calculate_perplexity(self, samples: List[Dict]) -> float:
        """
        计算模型perplexity

        Args:
            samples: 验证样本

        Returns:
            perplexity值
        """
        self.logger.info("Calculating perplexity...")

        total_loss = 0
        total_tokens = 0

        for sample in tqdm(samples, desc="Computing perplexity"):
            # 构造完整文本（包含response）
            prompt = sample['prompt']
            response = json.dumps(sample['response'], ensure_ascii=False)

            full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

            # Tokenize
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

            # 计算loss
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        self.logger.info(f"Perplexity: {perplexity:.2f}")

        return perplexity

    def run_evaluation(self, num_samples: int = None):
        """
        运行完整评估

        Args:
            num_samples: 评估样本数（None表示全部）
        """
        # 加载模型
        self.load_model()

        # 加载ground truth
        ground_truth_samples = self.load_ground_truth()

        if num_samples:
            ground_truth_samples = ground_truth_samples[:num_samples]

        # 1. 生成预测
        self.logger.info(f"Generating predictions for {len(ground_truth_samples)} samples...")

        predictions = []
        for sample in tqdm(ground_truth_samples, desc="Predicting"):
            pred = self.predict_single(sample['prompt'])
            predictions.append(pred)

        # 2. 计算决策准确率
        self.logger.info("Evaluating decision accuracy...")
        decision_metrics = self.evaluate_decision_accuracy(predictions, ground_truth_samples)

        # 3. 计算perplexity
        perplexity = self.calculate_perplexity(ground_truth_samples)

        # 汇总结果
        results = {
            'decision_metrics': decision_metrics,
            'perplexity': perplexity,
            'num_samples': len(ground_truth_samples),
        }

        # 保存结果
        output_path = "outputs/evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved evaluation results to {output_path}")

        return results

    def print_results(self, results: Dict):
        """打印评估结果"""
        print("\n" + "=" * 80)
        print("模型评估结果")
        print("=" * 80)

        decision_metrics = results['decision_metrics']

        print(f"\n决策准确率指标:")
        print(f"  - 路径选择准确率: {decision_metrics['route_selection_accuracy']:.2%}")
        print(f"  - 改道决策F1分数: {decision_metrics['reroute_f1']:.4f}")
        print(f"    - Precision: {decision_metrics['reroute_precision']:.4f}")
        print(f"    - Recall: {decision_metrics['reroute_recall']:.4f}")

        print(f"\n语言模型指标:")
        print(f"  - Perplexity: {results['perplexity']:.2f}")

        print(f"\n详细统计:")
        print(f"  - 总样本数: {decision_metrics['total_samples']}")
        print(f"  - 正确路径选择: {decision_metrics['correct_route']}")
        print(f"  - 改道TP: {decision_metrics['reroute_tp']}")
        print(f"  - 改道FP: {decision_metrics['reroute_fp']}")
        print(f"  - 改道FN: {decision_metrics['reroute_fn']}")

        print("\n" + "=" * 80)


def main():
    """主函数"""

    print("=" * 80)
    print("模型评估")
    print("=" * 80)

    # 初始化评估器
    evaluator = ModelEvaluator()

    # 运行评估
    results = evaluator.run_evaluation(num_samples=100)  # 先评估100个样本

    # 打印结果
    evaluator.print_results(results)


if __name__ == "__main__":
    main()

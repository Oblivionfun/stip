#!/usr/bin/env python3
"""
查看模型预测样例
直接加载验证数据并用指定模型进行预测，展示详细输出
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import argparse


def load_model(model_path):
    """加载模型"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded\n")
    return tokenizer, model


def predict_single(prompt, tokenizer, model):
    """预测单个样本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取response部分
    if prompt in generated:
        response = generated[len(prompt):].strip()
    else:
        response = generated

    return response


def main():
    parser = argparse.ArgumentParser(description="查看模型预测样例")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/models",
        help="模型路径 (默认: model/models)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="显示样本数量 (默认: 3)"
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="指定样本ID (可选)"
    )

    args = parser.parse_args()

    # 加载验证数据
    val_file = "outputs/validation_samples.jsonl"
    print(f"Loading validation samples from {val_file}...")
    samples = []
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"✓ Loaded {len(samples)} samples\n")

    # 如果指定了sample_id
    if args.sample_id:
        samples = [s for s in samples if s['id'] == args.sample_id]
        if not samples:
            print(f"❌ 未找到ID为 {args.sample_id} 的样本")
            return
    else:
        samples = samples[:args.num_samples]

    # 加载模型
    tokenizer, model = load_model(args.model_path)

    # 预测并显示
    print("=" * 80)
    print("预测结果")
    print("=" * 80)

    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"样本 {i}/{len(samples)}: {sample['id']}")
        print(f"{'='*80}")

        # 显示prompt (截断)
        prompt_preview = sample['prompt'][:300].replace('\n', ' ')
        print(f"\n📝 Prompt (前300字符):")
        print(f"   {prompt_preview}...")

        # Ground Truth
        print(f"\n✅ Ground Truth:")
        gt_response = sample['response']
        print(f"   Thinking: {gt_response.get('thinking', 'N/A')[:100]}...")
        print(f"   Reflection: {gt_response.get('reflection', 'N/A')[:100]}...")
        print(f"   Plan: {gt_response.get('plan', 'N/A')}")

        # 模型预测
        print(f"\n🤖 模型预测:")
        print(f"   正在生成...")
        prediction = predict_single(sample['prompt'], tokenizer, model)
        print(f"\n   原始输出:")
        print(f"   {prediction[:500]}")

        # 尝试解析JSON
        try:
            # 查找JSON部分
            if '{' in prediction and '}' in prediction:
                json_start = prediction.find('{')
                json_end = prediction.rfind('}') + 1
                json_str = prediction[json_start:json_end]
                pred_json = json.loads(json_str)

                print(f"\n   解析后的JSON:")
                print(f"   Thinking: {pred_json.get('thinking', 'N/A')[:100]}...")
                print(f"   Reflection: {pred_json.get('reflection', 'N/A')[:100]}...")
                print(f"   Plan: {pred_json.get('plan', 'N/A')}")

                # 对比
                gt_plan = gt_response.get('plan', '')
                pred_plan = pred_json.get('plan', '')
                match = '✅ 匹配' if gt_plan == pred_plan else '❌ 不匹配'
                print(f"\n   决策对比: {match}")
                print(f"     Ground Truth Plan: {gt_plan}")
                print(f"     Predicted Plan:    {pred_plan}")
            else:
                print(f"   ⚠️  输出不包含有效的JSON格式")
        except Exception as e:
            print(f"   ❌ JSON解析失败: {e}")

        print(f"\n{'='*80}\n")

    print("\n✓ 预测完成")


if __name__ == "__main__":
    main()

"""
阶段2.3：LLM样本构造器
功能：
1. 将场景+决策转换为GATSim格式的LLM训练样本
2. 生成prompt（场景描述）和response（reflection + plan）
3. 输出JSONL格式训练数据
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.common import (
    setup_logger,
    load_config,
    load_json,
    save_json,
    ensure_dir
)


class SampleBuilder:
    """LLM训练样本构造器"""

    def __init__(
        self,
        preference_config_path: str = "configs/preference_config.yaml",
        scenario_config_path: str = "configs/scenario_config.yaml"
    ):
        self.pref_config = load_config(preference_config_path)
        self.scen_config = load_config(scenario_config_path)
        self.logger = setup_logger(
            'SampleBuilder',
            f"{self.pref_config['data']['output_dir']}/logs/sample_builder.log"
        )
        self.rng = np.random.RandomState(42)

    def load_data(self) -> tuple:
        """加载所有数据"""
        output_dir = self.pref_config['data']['output_dir']

        personas = load_json(f"{output_dir}/personas.json")
        scenarios = load_json(f"{output_dir}/scenarios.json")
        decisions = load_json(f"{output_dir}/decisions.json")

        # 创建决策索引
        decision_dict = {d['scenario_id']: d for d in decisions}

        self.logger.info(f"Loaded {len(personas)} personas, {len(scenarios)} scenarios, {len(decisions)} decisions")

        return personas, scenarios, decision_dict

    def generate_prompt(self, persona: Dict, scenario: Dict) -> str:
        """
        生成GATSim格式的prompt

        Args:
            persona: Persona对象
            scenario: 场景对象

        Returns:
            prompt字符串
        """
        lines = []

        # 1. 角色描述
        lines.append("You play the role of the person:")
        lines.append(f"Name: {persona['name']} | Age: {persona['age']} | Gender: {persona['gender']} | Country: {persona['country']}")
        lines.append(f"Occupation: {persona.get('occupation', 'professional')} | Education: {persona.get('education', 'bachelor')}")
        lines.append("")
        lines.append(f"Transportation preferences: {persona['preferences_in_transportation']}")
        lines.append(f"Personality traits: {persona['innate']}")
        lines.append("")

        # 2. 当前情境
        lines.append("Current situation:")
        lines.append(f"Time period: {scenario['time_period']}")  # 使用抽象时段
        lines.append(f"Location: {scenario['origin']}")
        lines.append(f"Destination: {scenario['destination']}")
        lines.append(f"Purpose: {scenario['trip_purpose'].replace('_', ' ')}")

        # 重要会议提示
        if scenario['context'].get('has_important_meeting'):
            lines.append("⚠️ Important: You have an urgent meeting and cannot be late.")

        lines.append("")

        # 3. 可用路径
        lines.append("Available routes:")
        for route in scenario['routes']:
            lines.append(f"Route {route['id']} ({route['name']}):")
            lines.append(f"- Description: {route['description']}")
            lines.append(f"- Normal travel time: {route['normal_travel_time']} minutes")
            lines.append(f"- Current travel time: {route['current_travel_time']} minutes (delay: {route['current_delay']} min)")

            # 熟悉度描述
            if route['familiarity'] > 0.8:
                fam_desc = "High - your usual route"
            elif route['familiarity'] > 0.5:
                fam_desc = "Medium"
            else:
                fam_desc = "Low - unfamiliar"
            lines.append(f"- Familiarity: {fam_desc}")
            lines.append(f"- Uncertainty: {route['uncertainty_level']}")
            lines.append("")

        # 4. 交通状况
        if scenario['traffic_events']:
            lines.append("Current traffic conditions:")
            for event in scenario['traffic_events']:
                lines.append(f"- {event['type'].capitalize()} at {event['location']}: {event['severity']} severity, expected {event['expected_delay']}-minute delay")
            lines.append("")

        # 5. 天气
        if scenario['context']['weather'] != 'clear':
            lines.append(f"Weather: {scenario['context']['weather']}")
            lines.append("")

        # 6. 指令
        lines.append("What route would you choose and why? Respond in JSON format:")
        lines.append("{")
        lines.append('    "thinking": "your step-by-step analysis of the situation",')
        lines.append('    "reflection": "your final reasoning and decision-making process",')
        lines.append('    "plan": "your route choice (use \'none\' to keep Route A, or \'update path: <route_description>\' to switch)",')
        lines.append('    "concepts": []')
        lines.append("}")

        return "\n".join(lines)

    def generate_thinking(
        self,
        persona: Dict,
        scenario: Dict,
        decision: Dict
    ) -> str:
        """
        生成thinking过程（思考过程）

        Args:
            persona: Persona对象
            scenario: 场景对象
            decision: 决策对象

        Returns:
            thinking字符串
        """
        thinking_parts = []

        # 1. 分析当前情况
        thinking_parts.append("Let me analyze the current situation:")

        # 2. 路况分析
        route_a = scenario['routes'][0]
        route_b = scenario['routes'][1] if len(scenario['routes']) > 1 else None

        thinking_parts.append(f"- Route A (usual): {route_a['current_travel_time']}min (delay: {route_a['current_delay']}min, uncertainty: {route_a['uncertainty_level']})")
        if route_b:
            thinking_parts.append(f"- Route B (alternative): {route_b['current_travel_time']}min (delay: {route_b['current_delay']}min, uncertainty: {route_b['uncertainty_level']})")

        # 3. 个人偏好考虑
        factors = persona.get('preference_factors', {})
        thinking_parts.append("\nConsidering my preferences:")

        # 提取主要偏好
        if factors.get('time_sensitivity', 0) > 0.3:
            thinking_parts.append("- I prioritize time efficiency")
        if factors.get('risk_aversion', 0) > 0.3:
            thinking_parts.append("- I prefer predictable routes over uncertain ones")
        if factors.get('familiar_route_preference', 0) > 0.3:
            thinking_parts.append("- I prefer sticking with familiar routes")
        if factors.get('rerouting_proactiveness', 0) > 0.3:
            thinking_parts.append("- I tend to actively seek alternative routes")

        # 4. 情境因素
        if scenario['context'].get('has_important_meeting'):
            thinking_parts.append("- I have an urgent meeting, cannot afford to be late")

        # 5. 延误阈值判断
        delay_threshold = persona.get('delay_tolerance_planned', 10)
        if route_a['current_delay'] > delay_threshold:
            thinking_parts.append(f"\nThe delay ({route_a['current_delay']}min) exceeds my tolerance threshold ({delay_threshold}min).")
        else:
            thinking_parts.append(f"\nThe delay ({route_a['current_delay']}min) is within my tolerance threshold ({delay_threshold}min).")

        return "\n".join(thinking_parts)

    def generate_reflection(
        self,
        persona: Dict,
        scenario: Dict,
        decision: Dict
    ) -> str:
        """
        生成reflection文本（决策推理）

        Args:
            persona: Persona对象
            scenario: 场景对象
            decision: 决策对象

        Returns:
            reflection字符串
        """
        parts = []

        # 1. 当前情境（移除具体时间）
        parts.append(f"I am currently at {scenario['origin']}, planning to travel to {scenario['destination']} during {scenario['time_period']}.")

        # 2. 路况分析
        route_a = scenario['routes'][0]
        route_b = scenario['routes'][1] if len(scenario['routes']) > 1 else None

        if route_a['current_delay'] > 0:
            # 有延误
            delay_desc = "a significant" if route_a['current_delay'] > 15 else "a moderate" if route_a['current_delay'] > 5 else "a minor"
            if scenario['traffic_events']:
                event = scenario['traffic_events'][0]
                parts.append(f"My usual route (Route A) shows {delay_desc} {route_a['current_delay']}-minute delay due to {event['type']} at {event['location']}.")
            else:
                parts.append(f"My usual route (Route A) shows {delay_desc} {route_a['current_delay']}-minute delay.")
        else:
            # 无延误
            parts.append(f"My usual route (Route A) is currently clear with normal travel time of {route_a['normal_travel_time']} minutes.")

        # 3. 偏好考量
        factors = persona.get('preference_factors', {})
        dominant_factor = self.get_dominant_factor(factors)

        if dominant_factor == 'risk_aversion' and factors['risk_aversion'] > 0.3:
            parts.append("As someone who values travel time reliability and predictability, I prefer routes with less uncertainty even if they take slightly longer.")
        elif dominant_factor == 'time_sensitivity' and factors['time_sensitivity'] > 0.3:
            parts.append("Time efficiency is important to me, so I tend to choose the fastest option available.")
        elif dominant_factor == 'familiar_route_preference' and factors['familiar_route_preference'] > 0.3:
            parts.append("I generally prefer sticking with my familiar route unless there's a compelling reason to change.")
        elif dominant_factor == 'rerouting_proactiveness' and factors['rerouting_proactiveness'] > 0.3:
            parts.append("I am proactive about finding alternative routes to avoid delays and uncertainties.")

        # 4. 情境因素
        if scenario['context'].get('has_important_meeting'):
            parts.append("Given that I have an important meeting and cannot afford to be late, time reliability is especially critical.")

        # 5. 决策结论
        if decision['reroute'] and route_b:
            time_diff = route_a['current_travel_time'] - route_b['current_travel_time']
            if time_diff > 0:
                parts.append(f"Therefore, I will switch to Route {decision['chosen_route']}, which saves {time_diff} minutes and avoids the congested area.")
            else:
                parts.append(f"Therefore, I will switch to Route {decision['chosen_route']} for better reliability, even though it may take slightly longer.")
        else:
            # 保持原路径
            if route_a['current_delay'] > 0:
                parts.append(f"However, the delay is still within my tolerance (threshold: {persona.get('delay_tolerance_planned', 10)} minutes), so I will stick with my usual route.")
            else:
                parts.append("Therefore, I will continue with my usual Route A.")

        return " ".join(parts)

    def get_dominant_factor(self, factors: Dict[str, float]) -> str:
        """获取主导偏好因子"""
        if not factors:
            return 'time_sensitivity'

        # 只考虑绝对值较大的因子
        abs_factors = {k: abs(v) for k, v in factors.items()}
        if not abs_factors:
            return 'time_sensitivity'

        return max(abs_factors, key=abs_factors.get)

    def generate_plan(self, scenario: Dict, decision: Dict) -> str:
        """
        生成plan字段

        Args:
            scenario: 场景对象
            decision: 决策对象

        Returns:
            plan字符串
        """
        if not decision['reroute']:
            return "none"

        # 找到选择的路径
        chosen_route = next(
            (r for r in scenario['routes'] if r['id'] == decision['chosen_route']),
            None
        )

        if not chosen_route:
            return "none"

        # 提取路径描述
        # "via Main_Ave and Highway_1" -> "Main_Ave, Highway_1"
        description = chosen_route['description'].replace('via ', '').replace(' and ', ', ')

        return f"update path: {description}"

    def build_single_sample(
        self,
        persona: Dict,
        scenario: Dict,
        decision: Dict
    ) -> Dict:
        """
        构建单个训练样本

        Args:
            persona: Persona对象
            scenario: 场景对象
            decision: 决策对象

        Returns:
            训练样本字典
        """
        # 生成prompt
        prompt = self.generate_prompt(persona, scenario)

        # 生成response（包含思考过程）
        thinking = self.generate_thinking(persona, scenario, decision)
        reflection = self.generate_reflection(persona, scenario, decision)
        plan = self.generate_plan(scenario, decision)

        response = {
            "thinking": thinking,      # 思考过程
            "reflection": reflection,  # 决策推理
            "plan": plan,              # 行动计划
            "concepts": []
        }

        # 构造样本
        sample = {
            "id": scenario['scenario_id'],
            "persona_id": scenario['persona_id'],
            "scenario_id": scenario['scenario_id'],
            "prompt": prompt,
            "response": response
        }

        return sample

    def build_all_samples(
        self,
        personas: Dict[str, Dict],
        scenarios: List[Dict],
        decision_dict: Dict[str, Dict]
    ) -> List[Dict]:
        """构建所有训练样本"""
        self.logger.info(f"Building training samples for {len(scenarios)} scenarios...")

        samples = []
        for idx, scenario in enumerate(scenarios, 1):
            try:
                persona_id = scenario['persona_id']
                scenario_id = scenario['scenario_id']

                persona = personas.get(persona_id)
                decision = decision_dict.get(scenario_id)

                if not persona:
                    self.logger.warning(f"Persona {persona_id} not found, skipping...")
                    continue

                if not decision:
                    self.logger.warning(f"Decision for {scenario_id} not found, skipping...")
                    continue

                sample = self.build_single_sample(persona, scenario, decision)
                samples.append(sample)

                if idx % 1000 == 0:
                    self.logger.info(f"Progress: {idx}/{len(scenarios)} samples")

            except Exception as e:
                self.logger.error(f"Failed to build sample for {scenario['scenario_id']}: {e}")

        self.logger.info(f"Built {len(samples)} training samples")

        return samples

    def split_train_val(self, samples: List[Dict]) -> tuple:
        """划分训练集和验证集"""
        split_config = self.scen_config['data_split']

        # 打乱
        if split_config['shuffle']:
            self.rng.shuffle(samples)

        # 划分
        train_ratio = split_config['train_ratio']
        split_idx = int(len(samples) * train_ratio)

        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        self.logger.info(f"Split into {len(train_samples)} train and {len(val_samples)} val samples")

        return train_samples, val_samples

    def save_samples_jsonl(self, samples: List[Dict], filename: str):
        """保存样本为JSONL格式"""
        output_dir = self.pref_config['data']['output_dir']
        output_path = f"{output_dir}/{filename}"

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        self.logger.info(f"Saved {len(samples)} samples to {output_path}")

        # 计算文件大小
        import os
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        self.logger.info(f"File size: {file_size:.2f} MB")

    def compute_statistics(self, samples: List[Dict]) -> Dict:
        """计算样本统计信息"""
        stats = {
            'total_samples': len(samples),
            'avg_prompt_length': 0,
            'avg_reflection_length': 0,
            'plan_distribution': {}
        }

        prompt_lengths = []
        reflection_lengths = []

        for sample in samples:
            # Prompt长度
            prompt_lengths.append(len(sample['prompt'].split()))

            # Reflection长度
            reflection_lengths.append(len(sample['response']['reflection'].split()))

            # Plan分布
            plan = sample['response']['plan']
            if plan.startswith('update path'):
                plan_type = 'update_path'
            elif plan == 'none':
                plan_type = 'none'
            else:
                plan_type = 'other'

            stats['plan_distribution'][plan_type] = stats['plan_distribution'].get(plan_type, 0) + 1

        stats['avg_prompt_length'] = round(np.mean(prompt_lengths), 2)
        stats['avg_reflection_length'] = round(np.mean(reflection_lengths), 2)

        return stats

    def run(self) -> tuple:
        """运行完整的样本构造流程"""
        # 加载数据
        personas, scenarios, decision_dict = self.load_data()

        # 构建样本
        samples = self.build_all_samples(personas, scenarios, decision_dict)

        # 划分训练/验证集
        train_samples, val_samples = self.split_train_val(samples)

        # 计算统计
        stats = self.compute_statistics(samples)

        # 保存
        self.save_samples_jsonl(train_samples, "train_samples.jsonl")
        self.save_samples_jsonl(val_samples, "validation_samples.jsonl")

        # 保存统计
        output_dir = self.pref_config['data']['output_dir']
        save_json(stats, f"{output_dir}/sample_statistics.json", indent=2)
        self.logger.info(f"Saved statistics")

        return train_samples, val_samples


def main():
    """主函数"""

    print("=" * 80)
    print("阶段2.3：LLM样本构造器")
    print("=" * 80)

    # 初始化构造器
    builder = SampleBuilder()

    # 运行
    print("\n[1/1] 构建LLM训练样本...")
    train_samples, val_samples = builder.run()

    print(f"\n✓ 样本构造完成！")
    print(f"  - 训练样本: {len(train_samples)}")
    print(f"  - 验证样本: {len(val_samples)}")

    # 读取统计信息
    output_dir = builder.pref_config['data']['output_dir']
    stats = load_json(f"{output_dir}/sample_statistics.json")

    print(f"  - 平均prompt长度: {stats['avg_prompt_length']} 词")
    print(f"  - 平均reflection长度: {stats['avg_reflection_length']} 词")
    print(f"  - plan分布: {stats['plan_distribution']}")

    # 展示示例
    print("\n" + "=" * 80)
    print("样本示例（第1个）")
    print("=" * 80)

    sample = train_samples[0]
    print(f"\n[ID] {sample['id']}")
    print(f"\n[PROMPT]")
    print(sample['prompt'][:500] + "..." if len(sample['prompt']) > 500 else sample['prompt'])
    print(f"\n[RESPONSE]")
    print(json.dumps(sample['response'], indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("✓ 阶段2.3 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

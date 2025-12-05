"""
阶段2.2：决策模拟器
功能：
1. 根据Persona偏好因子和场景属性模拟路径选择决策
2. 使用简化的效用函数 + Gumbel噪声
3. 确保决策分布合理性
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


class DecisionSimulator:
    """路径选择决策模拟器"""

    def __init__(
        self,
        preference_config_path: str = "configs/preference_config.yaml",
        scenario_config_path: str = "configs/scenario_config.yaml"
    ):
        self.pref_config = load_config(preference_config_path)
        self.scen_config = load_config(scenario_config_path)
        self.logger = setup_logger(
            'DecisionSimulator',
            f"{self.pref_config['data']['output_dir']}/logs/decision_simulator.log"
        )
        self.sim_config = self.scen_config['decision_simulation']
        self.rng = np.random.RandomState(42)

    def load_data(self) -> tuple:
        """加载Personas和Scenarios"""
        output_dir = self.pref_config['data']['output_dir']

        personas = load_json(f"{output_dir}/personas.json")
        scenarios = load_json(f"{output_dir}/scenarios.json")

        self.logger.info(f"Loaded {len(personas)} personas and {len(scenarios)} scenarios")

        return personas, scenarios

    def calculate_utility(
        self,
        persona: Dict,
        route: Dict,
        scenario: Dict
    ) -> float:
        """
        计算路径效用

        Args:
            persona: Persona对象
            route: 路径对象
            scenario: 场景对象

        Returns:
            效用值（越高越好）
        """
        weights = self.sim_config['utility_weights']
        factors = persona.get('preference_factors', {})

        # 1. 基础时间成本
        time_cost = weights['time_cost'] * route['current_travel_time']

        # 2. 延误惩罚（基于time_sensitivity因子）
        time_sensitivity = factors.get('time_sensitivity', 0.0)
        delay_penalty = weights['delay_penalty'] * time_sensitivity * route['current_delay']

        # 3. 不确定性惩罚（基于risk_aversion因子）
        risk_aversion = factors.get('risk_aversion', 0.0)
        uncertainty_map = self.sim_config['uncertainty_cost_map']
        uncertainty_cost = uncertainty_map.get(route['uncertainty_level'], 0)
        uncertainty_penalty = weights['uncertainty_penalty'] * risk_aversion * uncertainty_cost

        # 4. 熟悉路线奖励（基于familiar_route_preference因子）
        familiar_preference = factors.get('familiar_route_preference', 0.0)
        familiarity_bonus = weights['familiarity_bonus'] * familiar_preference * route['familiarity']

        # 5. 情境调整
        context_adjustment = 0.0
        if scenario['context'].get('has_important_meeting', False):
            # 有重要会议时，时间成本和延误惩罚加倍
            boost = self.sim_config['threshold_rules']['important_meeting_boost']
            time_cost *= boost
            delay_penalty *= boost

        # 总效用
        utility = time_cost + delay_penalty + uncertainty_penalty + familiarity_bonus + context_adjustment

        return utility

    def apply_threshold_rules(
        self,
        persona: Dict,
        scenario: Dict,
        chosen_route: str,
        utilities: Dict[str, float]
    ) -> str:
        """
        应用阈值规则，可能覆盖效用计算的结果

        Args:
            persona: Persona对象
            scenario: 场景对象
            chosen_route: 基于效用选择的路径
            utilities: 各路径的效用值

        Returns:
            最终选择的路径
        """
        threshold_config = self.sim_config['threshold_rules']

        if not threshold_config['force_reroute_if_exceed_threshold']['enabled']:
            return chosen_route

        # 获取路径A的延误
        route_a = scenario['routes'][0]
        delay = route_a['current_delay']

        # 获取个人延误容忍阈值
        delay_threshold = persona.get('delay_tolerance_planned', 10)

        # 如果延误超过阈值，强制改道（高概率）
        if delay > delay_threshold:
            force_prob = threshold_config['force_reroute_if_exceed_threshold']['probability']
            if self.rng.rand() < force_prob:
                # 强制选择延误最小的替代路径
                route_delays = {r['id']: r['current_delay'] for r in scenario['routes']}
                min_delay_route = min(route_delays, key=route_delays.get)
                if min_delay_route != 'A':
                    return min_delay_route

        return chosen_route

    def simulate_single_decision(
        self,
        persona: Dict,
        scenario: Dict
    ) -> Dict:
        """
        模拟单个场景的决策

        Args:
            persona: Persona对象
            scenario: 场景对象

        Returns:
            决策结果字典
        """
        routes = scenario['routes']

        # 计算各路径效用
        utilities = {}
        for route in routes:
            utility = self.calculate_utility(persona, route, scenario)

            # 添加Gumbel噪声（模拟随机性）
            if self.sim_config['gumbel_noise']['enabled']:
                noise_scale = self.sim_config['gumbel_noise']['scale']
                noise = self.rng.gumbel(0, noise_scale)
                utility += noise

            utilities[route['id']] = utility

        # 选择效用最高的路径
        chosen = max(utilities, key=utilities.get)

        # 应用阈值规则
        chosen = self.apply_threshold_rules(persona, scenario, chosen, utilities)

        # 生成推理原因
        reasoning = self.generate_reasoning(persona, scenario, chosen, utilities)

        return {
            'scenario_id': scenario['scenario_id'],
            'persona_id': scenario['persona_id'],
            'chosen_route': chosen,
            'reroute': chosen != 'A',  # A是默认路径
            'reasoning': reasoning,
            'utility_scores': {k: round(v, 3) for k, v in utilities.items()}
        }

    def generate_reasoning(
        self,
        persona: Dict,
        scenario: Dict,
        chosen: str,
        utilities: Dict[str, float]
    ) -> str:
        """生成决策推理原因"""
        route_a = scenario['routes'][0]
        chosen_route = next(r for r in scenario['routes'] if r['id'] == chosen)

        reasons = []

        # 主要考虑因素
        factors = persona.get('preference_factors', {})

        # 1. 延误情况
        if route_a['current_delay'] > persona.get('delay_tolerance_planned', 10):
            reasons.append("delay exceeds tolerance threshold")
        elif route_a['current_delay'] > 5:
            reasons.append("moderate delay on usual route")

        # 2. 关键偏好因子
        if factors.get('time_sensitivity', 0) > 0.5:
            reasons.append("time-sensitive")
        if factors.get('risk_aversion', 0) > 0.5:
            reasons.append("risk-averse (prefer predictability)")
        if factors.get('familiar_route_preference', 0) > 0.5:
            reasons.append("prefer familiar routes")

        # 3. 情境因素
        if scenario['context'].get('has_important_meeting'):
            reasons.append("urgent meeting")

        # 4. 决策结果
        if chosen != 'A':
            time_saved = route_a['current_travel_time'] - chosen_route['current_travel_time']
            if time_saved > 0:
                reasons.append(f"saves {time_saved} minutes")
            else:
                reasons.append("avoids uncertainty")
        else:
            reasons.append("within tolerance")

        return "; ".join(reasons) if reasons else "balanced choice"

    def simulate_all_decisions(
        self,
        personas: Dict[str, Dict],
        scenarios: List[Dict]
    ) -> List[Dict]:
        """为所有场景模拟决策"""
        self.logger.info(f"Simulating decisions for {len(scenarios)} scenarios...")

        decisions = []
        persona_dict = {p_id: p for p_id, p in personas.items()}

        for idx, scenario in enumerate(scenarios, 1):
            try:
                persona_id = scenario['persona_id']
                persona = persona_dict.get(persona_id)

                if not persona:
                    self.logger.warning(f"Persona {persona_id} not found, skipping...")
                    continue

                decision = self.simulate_single_decision(persona, scenario)
                decisions.append(decision)

                if idx % 1000 == 0:
                    self.logger.info(f"Progress: {idx}/{len(scenarios)} scenarios")

            except Exception as e:
                self.logger.error(f"Failed to simulate decision for {scenario['scenario_id']}: {e}")

        self.logger.info(f"Generated {len(decisions)} decisions")

        return decisions

    def compute_statistics(self, decisions: List[Dict]) -> Dict:
        """计算决策统计信息"""
        total = len(decisions)
        reroute_count = sum(1 for d in decisions if d['reroute'])
        reroute_rate = reroute_count / total if total > 0 else 0

        stats = {
            'total_decisions': total,
            'reroute_count': reroute_count,
            'keep_route_count': total - reroute_count,
            'reroute_rate': round(reroute_rate * 100, 2),

            # 按路径统计
            'route_distribution': {},

            # 推理原因统计
            'reasoning_keywords': {}
        }

        # 统计路径分布
        for decision in decisions:
            route = decision['chosen_route']
            stats['route_distribution'][route] = stats['route_distribution'].get(route, 0) + 1

        # 统计推理关键词
        keywords = ['time-sensitive', 'risk-averse', 'familiar', 'delay', 'urgent', 'saves']
        for keyword in keywords:
            count = sum(1 for d in decisions if keyword in d['reasoning'])
            stats['reasoning_keywords'][keyword] = {
                'count': count,
                'percentage': round(count / total * 100, 2)
            }

        return stats

    def validate_quality(self, stats: Dict) -> bool:
        """验证决策质量"""
        target_range = self.scen_config['quality_control']['target_reroute_rate']
        reroute_rate = stats['reroute_rate'] / 100

        if target_range[0] <= reroute_rate <= target_range[1]:
            self.logger.info(f"✓ Reroute rate {stats['reroute_rate']}% is within target range [{target_range[0]*100}%, {target_range[1]*100}%]")
            return True
        else:
            self.logger.warning(f"⚠ Reroute rate {stats['reroute_rate']}% is outside target range [{target_range[0]*100}%, {target_range[1]*100}%]")
            return False

    def save_decisions(self, decisions: List[Dict], stats: Dict):
        """保存决策数据"""
        output_dir = self.pref_config['data']['output_dir']

        # 保存决策
        decisions_path = f"{output_dir}/decisions.json"
        save_json(decisions, decisions_path, indent=2)
        self.logger.info(f"Saved {len(decisions)} decisions to {decisions_path}")

        # 保存统计
        stats_path = f"{output_dir}/decision_statistics.json"
        save_json(stats, stats_path, indent=2)
        self.logger.info(f"Saved statistics to {stats_path}")

    def run(self) -> List[Dict]:
        """运行完整的决策模拟流程"""
        # 加载数据
        personas, scenarios = self.load_data()

        # 模拟决策
        decisions = self.simulate_all_decisions(personas, scenarios)

        # 计算统计
        stats = self.compute_statistics(decisions)

        # 质量验证
        self.validate_quality(stats)

        # 保存
        self.save_decisions(decisions, stats)

        return decisions


def main():
    """主函数"""

    print("=" * 80)
    print("阶段2.2：决策模拟器")
    print("=" * 80)

    # 初始化模拟器
    simulator = DecisionSimulator()

    # 运行
    print("\n[1/1] 模拟路径选择决策...")
    decisions = simulator.run()

    print(f"\n✓ 决策模拟完成！")
    print(f"  - 总决策数: {len(decisions)}")

    # 读取统计信息
    output_dir = simulator.pref_config['data']['output_dir']
    stats = load_json(f"{output_dir}/decision_statistics.json")

    print(f"  - 改道率: {stats['reroute_rate']}%")
    print(f"  - 改道决策: {stats['reroute_count']}")
    print(f"  - 保持原路径: {stats['keep_route_count']}")

    # 展示示例
    print("\n" + "=" * 80)
    print("决策示例（前3个）")
    print("=" * 80)

    for i, decision in enumerate(decisions[:3], 1):
        print(f"\n[{i}] {decision['scenario_id']}:")
        print(f"  选择: Route {decision['chosen_route']} ({'改道' if decision['reroute'] else '保持'})")
        print(f"  推理: {decision['reasoning']}")
        print(f"  效用: {decision['utility_scores']}")

    print("\n" + "=" * 80)
    print("✓ 阶段2.2 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

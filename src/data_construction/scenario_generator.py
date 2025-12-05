"""
阶段2.1：场景生成器
功能：
1. 为每个Persona生成多个路径选择场景
2. 确保场景多样性（延误程度、事件类型、时段等）
3. 根据Persona属性定制场景参数
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.common import (
    setup_logger,
    load_config,
    load_json,
    save_json,
    ensure_dir
)


class ScenarioGenerator:
    """路径选择场景生成器"""

    def __init__(
        self,
        preference_config_path: str = "configs/preference_config.yaml",
        scenario_config_path: str = "configs/scenario_config.yaml"
    ):
        self.pref_config = load_config(preference_config_path)
        self.scen_config = load_config(scenario_config_path)
        self.logger = setup_logger(
            'ScenarioGenerator',
            f"{self.pref_config['data']['output_dir']}/logs/scenario_generator.log"
        )
        self.gen_config = self.scen_config['scenario_generation']
        self.rng = np.random.RandomState(42)
        random.seed(42)

    def load_personas(self) -> Dict[str, Dict]:
        """加载Persona数据"""
        output_dir = self.pref_config['data']['output_dir']
        personas_path = f"{output_dir}/personas.json"

        personas = load_json(personas_path)
        self.logger.info(f"Loaded {len(personas)} personas")

        return personas

    def select_template(self) -> Dict:
        """根据权重随机选择场景模板"""
        templates = self.gen_config['templates']
        weights = [t['weight'] for t in templates]

        chosen = self.rng.choice(templates, p=weights)
        return chosen.copy()

    def select_delay_scenario(self) -> Dict:
        """选择延误场景"""
        delay_scenarios = self.gen_config['delay_scenarios']
        weights = [d['weight'] for d in delay_scenarios]

        chosen = self.rng.choice(delay_scenarios, p=weights)
        return chosen.copy()

    def select_event_type(self) -> str:
        """选择事件类型"""
        event_types = self.gen_config['event_types']
        types = list(event_types.keys())
        probs = list(event_types.values())

        return self.rng.choice(types, p=probs)

    def select_event_severity(self) -> str:
        """选择事件严重程度"""
        severity_config = self.gen_config['event_severity']
        severities = list(severity_config.keys())
        probs = list(severity_config.values())

        return self.rng.choice(severities, p=probs)

    def generate_time_period(self, template: Dict) -> str:
        """
        生成时段描述（抽象化，避免过拟合）
        不再使用具体时间，改用时段类型
        """
        # 根据模板类型返回时段
        if template['name'] == 'commute_morning':
            return "morning rush hour"
        elif template['name'] == 'commute_evening':
            return "evening rush hour"
        else:
            return "off-peak hours"

    def generate_route(
        self,
        route_id: str,
        is_usual: bool,
        persona: Dict,
        delay: int = 0
    ) -> Dict:
        """生成单个路径"""
        # 基准旅行时间（基于Persona的commute_time）
        base_commute = persona.get('commute_time_morning', 30)

        # 根据配置添加随机变化
        time_ratio_range = self.gen_config['routes']['travel_time_base_ratio']
        time_ratio = self.rng.uniform(time_ratio_range[0], time_ratio_range[1])
        normal_time = int(base_commute * time_ratio)

        # 替代路线通常稍长
        if not is_usual:
            time_diff_range = self.gen_config['routes']['alternative_time_difference']
            time_diff = self.rng.randint(time_diff_range[0], time_diff_range[1])
            normal_time += time_diff

        # 熟悉度
        if is_usual:
            fam_range = self.gen_config['routes']['familiarity']['usual_route']
        else:
            fam_range = self.gen_config['routes']['familiarity']['alternative_route']
        familiarity = self.rng.uniform(fam_range[0], fam_range[1])

        # 不确定性等级
        if delay > 15:
            uncertainty = "high"
        elif delay > 5:
            uncertainty = "moderate"
        else:
            uncertainty = "low"

        # 当前旅行时间
        current_time = normal_time + delay

        # 路线描述
        if route_id == "A":
            description = "via Main_Ave and Highway_1"
            name = "Usual Route"
        else:
            description = "via Side_St and Local_Road_2"
            name = "Alternative Route"

        return {
            "id": route_id,
            "name": name,
            "description": description,
            "normal_travel_time": normal_time,
            "current_delay": delay,
            "current_travel_time": current_time,
            "uncertainty_level": uncertainty,
            "familiarity": round(familiarity, 2)
        }

    def generate_traffic_event(
        self,
        event_type: str,
        severity: str,
        delay: int
    ) -> Dict:
        """生成交通事件"""
        # 事件位置（简化版）
        locations = ["Main_Ave_Link_1", "Highway_1_Segment_2", "Intersection_A"]
        location = random.choice(locations)

        return {
            "location": location,
            "type": event_type,
            "severity": severity,
            "expected_delay": delay
        }

    def generate_context(self, template: Dict) -> Dict:
        """生成情境因素"""
        weather_config = self.gen_config['context']['weather_conditions']
        weathers = list(weather_config.keys())
        weather_probs = list(weather_config.values())
        weather = self.rng.choice(weathers, p=weather_probs)

        # 是否有重要会议
        important_meeting_prob = self.gen_config['context']['important_meeting_prob']
        has_important_meeting = self.rng.rand() < important_meeting_prob

        # 时间裕量
        buffer_range = self.gen_config['context']['time_buffer_range']
        time_buffer = self.rng.randint(buffer_range[0], buffer_range[1])

        # 是否高峰期
        is_rush_hour = template['name'] in ['commute_morning', 'commute_evening']

        return {
            "weather": weather,
            "is_rush_hour": is_rush_hour,
            "has_important_meeting": has_important_meeting,
            "time_buffer": time_buffer
        }

    def generate_single_scenario(
        self,
        persona_id: str,
        persona: Dict,
        scenario_idx: int
    ) -> Dict:
        """为单个Persona生成一个场景"""
        # 选择模板
        template = self.select_template()

        # 选择延误情况
        delay_scenario = self.select_delay_scenario()
        delay_range = delay_scenario['delay_range']
        delay = self.rng.randint(delay_range[0], delay_range[1] + 1)

        # 选择事件类型和严重程度
        event_type = self.select_event_type()
        severity = self.select_event_severity()

        # 生成时段（抽象化，避免过拟合）
        time_period = self.generate_time_period(template)

        # 生成路径
        # 路径A（常用路径）有延误
        route_a = self.generate_route("A", is_usual=True, persona=persona, delay=delay)

        # 路径B（替代路径）通常无延误或延误较小
        route_b_delay = 0 if delay > 5 else self.rng.randint(0, 3)
        route_b = self.generate_route("B", is_usual=False, persona=persona, delay=route_b_delay)

        routes = [route_a, route_b]

        # 生成交通事件（如果有延误）
        traffic_events = []
        if delay > 0:
            traffic_events.append(
                self.generate_traffic_event(event_type, severity, delay)
            )

        # 生成情境
        context = self.generate_context(template)

        # 构造场景对象
        scenario = {
            "scenario_id": f"{persona_id}_S{scenario_idx:02d}",
            "persona_id": persona_id,

            # 出行信息
            "origin": template['origin'],
            "destination": template['destination'],
            "time_period": time_period,  # 使用抽象时段而非具体时间
            "trip_purpose": template['trip_purpose'],

            # 路径选项
            "routes": routes,

            # 交通事件
            "traffic_events": traffic_events,

            # 情境因素
            "context": context
        }

        return scenario

    def validate_scenario(self, scenario: Dict) -> bool:
        """验证场景有效性"""
        validation = self.scen_config['quality_control']['validation']

        for route in scenario['routes']:
            # 检查旅行时间范围
            if route['current_travel_time'] > validation['max_travel_time']:
                return False
            if route['current_travel_time'] < validation['min_travel_time']:
                return False

            # 检查延误比例
            if route['normal_travel_time'] > 0:
                delay_ratio = route['current_delay'] / route['normal_travel_time']
                if delay_ratio > validation['max_delay_ratio']:
                    return False

        return True

    def generate_scenarios_for_persona(
        self,
        persona_id: str,
        persona: Dict
    ) -> List[Dict]:
        """为单个Persona生成多个场景"""
        num_scenarios = self.gen_config['scenarios_per_persona']
        scenarios = []

        for idx in range(1, num_scenarios + 1):
            scenario = self.generate_single_scenario(persona_id, persona, idx)

            # 验证
            if self.validate_scenario(scenario):
                scenarios.append(scenario)
            else:
                self.logger.warning(f"Invalid scenario {scenario['scenario_id']}, regenerating...")
                # 重新生成
                scenario = self.generate_single_scenario(persona_id, persona, idx)
                scenarios.append(scenario)

        return scenarios

    def generate_all_scenarios(self, personas: Dict[str, Dict]) -> List[Dict]:
        """为所有Persona生成场景"""
        self.logger.info(f"Generating scenarios for {len(personas)} personas...")

        all_scenarios = []
        for idx, (persona_id, persona) in enumerate(personas.items(), 1):
            try:
                scenarios = self.generate_scenarios_for_persona(persona_id, persona)
                all_scenarios.extend(scenarios)

                if idx % 100 == 0:
                    self.logger.info(f"Progress: {idx}/{len(personas)} personas ({len(all_scenarios)} scenarios)")

            except Exception as e:
                self.logger.error(f"Failed to generate scenarios for {persona_id}: {e}")

        self.logger.info(f"Generated {len(all_scenarios)} scenarios in total")

        return all_scenarios

    def compute_statistics(self, scenarios: List[Dict]) -> Dict:
        """计算场景统计信息"""
        stats = {
            'total_scenarios': len(scenarios),
            'delay_levels': {},
            'event_types': {},
            'trip_purposes': {},
            'time_periods': {}
        }

        for scenario in scenarios:
            # 延误等级
            route_a = scenario['routes'][0]
            delay = route_a['current_delay']

            if delay == 0:
                level = 'none'
            elif delay < 8:
                level = 'minor'
            elif delay < 16:
                level = 'moderate'
            else:
                level = 'severe'

            stats['delay_levels'][level] = stats['delay_levels'].get(level, 0) + 1

            # 事件类型
            if scenario['traffic_events']:
                event_type = scenario['traffic_events'][0]['type']
                stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
            else:
                stats['event_types']['none'] = stats['event_types'].get('none', 0) + 1

            # 出行目的
            purpose = scenario['trip_purpose']
            stats['trip_purposes'][purpose] = stats['trip_purposes'].get(purpose, 0) + 1

            # 时段
            hour = int(scenario['departure_time'].split(':')[0])
            if 7 <= hour < 10:
                period = 'morning_rush'
            elif 17 <= hour < 20:
                period = 'evening_rush'
            else:
                period = 'off_peak'

            stats['time_periods'][period] = stats['time_periods'].get(period, 0) + 1

        # 计算比例
        total = len(scenarios)
        for category in ['delay_levels', 'event_types', 'trip_purposes', 'time_periods']:
            for key in stats[category]:
                count = stats[category][key]
                stats[category][key] = {
                    'count': count,
                    'percentage': round(count / total * 100, 2)
                }

        return stats

    def save_scenarios(self, scenarios: List[Dict], stats: Dict):
        """保存场景数据"""
        output_dir = self.pref_config['data']['output_dir']

        # 保存场景
        scenarios_path = f"{output_dir}/scenarios.json"
        save_json(scenarios, scenarios_path, indent=2)
        self.logger.info(f"Saved {len(scenarios)} scenarios to {scenarios_path}")

        # 保存统计信息
        stats_path = f"{output_dir}/scenario_statistics.json"
        save_json(stats, stats_path, indent=2)
        self.logger.info(f"Saved statistics to {stats_path}")

    def run(self) -> List[Dict]:
        """运行完整的场景生成流程"""
        # 加载Personas
        personas = self.load_personas()

        # 生成场景
        scenarios = self.generate_all_scenarios(personas)

        # 计算统计
        stats = self.compute_statistics(scenarios)

        # 保存
        self.save_scenarios(scenarios, stats)

        return scenarios


def main():
    """主函数"""

    print("=" * 80)
    print("阶段2.1：场景生成器")
    print("=" * 80)

    # 初始化生成器
    generator = ScenarioGenerator()

    # 运行
    print("\n[1/1] 生成路径选择场景...")
    scenarios = generator.run()

    print(f"\n✓ 场景生成完成！")
    print(f"  - 总场景数: {len(scenarios)}")

    # 展示示例
    print("\n" + "=" * 80)
    print("场景示例（前2个）")
    print("=" * 80)

    for i, scenario in enumerate(scenarios[:2], 1):
        print(f"\n[{i}] {scenario['scenario_id']}:")
        print(f"  出行: {scenario['origin']} → {scenario['destination']} ({scenario['trip_purpose']})")
        print(f"  时间: {scenario['departure_time']} ({scenario['day_of_week']})")
        print(f"  路径A: {scenario['routes'][0]['current_travel_time']}分钟 (延误{scenario['routes'][0]['current_delay']}分钟)")
        print(f"  路径B: {scenario['routes'][1]['current_travel_time']}分钟 (延误{scenario['routes'][1]['current_delay']}分钟)")
        if scenario['traffic_events']:
            event = scenario['traffic_events'][0]
            print(f"  事件: {event['type']} at {event['location']} ({event['severity']})")

    print("\n" + "=" * 80)
    print("✓ 阶段2.1 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

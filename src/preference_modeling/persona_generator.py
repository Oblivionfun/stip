"""
阶段1.3：Persona生成器
功能：
1. 整合问卷数据和因子得分
2. 生成GATSim格式的Persona对象
3. 生成自然语言偏好描述
4. 提取决策关键参数
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import random
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.common import (
    setup_logger,
    load_config,
    ensure_dir,
    save_json,
    decode_value,
    calculate_income_level
)


class PersonaGenerator:
    """GATSim格式Persona生成器"""

    def __init__(self, config_path: str = "configs/preference_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(
            'PersonaGenerator',
            f"{self.config['data']['output_dir']}/logs/persona_generator.log"
        )
        self.encodings = self.config['encodings']

    def load_data(self) -> tuple:
        """加载清洗数据和因子得分"""
        output_dir = self.config['data']['output_dir']

        self.logger.info("Loading cleaned data and factor scores...")

        # 加载清洗后的数据
        cleaned_df = pd.read_csv(f"{output_dir}/cleaned_survey_data.csv")

        # 加载因子得分
        factors_df = pd.read_csv(f"{output_dir}/preference_factors.csv")

        # 合并
        merged_df = cleaned_df.merge(factors_df, on=['persona_id', 'country'], how='inner')

        self.logger.info(f"Loaded and merged {len(merged_df)} samples")

        return merged_df

    def generate_name(self, persona_id: str, gender: int, country: str) -> str:
        """生成Persona名字"""
        name_config = self.config['name_generation']

        if not name_config['enabled'] or not name_config['use_real_names']:
            # 生成格式化ID作为名字
            return f"User_{persona_id.split('_')[1]}"

        # 从名字池中随机选择
        gender_key = 'male' if gender == 1 else 'female'
        name_pool = name_config['name_pool'].get(country, {}).get(gender_key, [])

        if name_pool:
            return random.choice(name_pool)
        else:
            return f"User_{persona_id.split('_')[1]}"

    def generate_preference_description(self, row: pd.Series) -> str:
        """
        生成自然语言偏好描述

        Args:
            row: 包含因子得分和原始问题的行

        Returns:
            偏好描述字符串
        """
        parts = []

        # 1. 熟悉路线偏好
        if 'Q22' in row and row['Q22'] >= 4:
            parts.append("prefer familiar routes")

        # 2. 风险厌恶度
        if 'risk_aversion' in row:
            if row['risk_aversion'] > 0.5:
                parts.append("high risk aversion")
            elif row['risk_aversion'] < -0.5:
                parts.append("risk-tolerant")

        # 3. 信息依赖度
        if 'information_dependency' in row:
            if row['information_dependency'] > 0.5:
                parts.append("rely heavily on navigation apps")
            elif row['information_dependency'] < -0.5:
                parts.append("prefer self-judgment over navigation")

        # 4. 时间可靠性偏好
        if 'Q29' in row and row['Q29'] >= 4:
            parts.append("prefer smooth routes over fast but uncertain ones")

        # 5. 改道阈值
        if 'Q21' in row:
            delay_threshold = int(row['Q21'])
            if delay_threshold <= 10:
                parts.append(f"willing to reroute if delay >{delay_threshold}min")
            else:
                parts.append(f"patient with delays up to {delay_threshold}min")

        # 6. 时间敏感度
        if 'Q30' in row and row['Q30'] >= 4:
            parts.append("immediately reroute to save time")

        # 7. 信息板依赖
        if 'use_nav_app' in row and row['use_nav_app']:
            parts.append("use navigation apps regularly")

        return "; ".join(parts) if parts else "balanced decision-maker"

    def generate_personality_description(self, row: pd.Series) -> str:
        """生成个性特质描述"""
        traits = []

        # 基于因子得分
        if 'risk_aversion' in row:
            if row['risk_aversion'] > 0.5:
                traits.append("cautious")
            elif row['risk_aversion'] < -0.5:
                traits.append("bold")

        if 'time_sensitivity' in row and row['time_sensitivity'] > 0.5:
            traits.append("time-sensitive")

        if 'information_dependency' in row and row['information_dependency'] > 0.5:
            traits.append("information-dependent")

        if 'route_flexibility' in row and row['route_flexibility'] > 0.5:
            traits.append("flexible")

        if 'rerouting_proactiveness' in row and row['rerouting_proactiveness'] > 0.5:
            traits.append("proactive")

        # 默认
        if not traits:
            traits = ["rational", "balanced"]

        return ", ".join(traits)

    def create_persona(self, row: pd.Series) -> Dict:
        """
        从数据行创建单个Persona对象

        Args:
            row: 包含所有数据的行

        Returns:
            GATSim格式的Persona字典
        """
        persona_id = row['persona_id']
        country = row['country']

        # 基本属性
        persona = {
            # 基本信息
            'persona_id': persona_id,
            'name': self.generate_name(persona_id, int(row.get('Q49', 1)), country),
            'age': int(row['Q50']) if 'Q50' in row and not pd.isna(row['Q50']) else 30,
            'gender': decode_value(row.get('Q49'), self.encodings['gender'], 'male'),
            'country': country,

            # 教育和职业
            'education': decode_value(row.get('Q54'), self.encodings['education'], 'bachelor'),
            'occupation': decode_value(row.get('Q53'), self.encodings['occupation'], 'professional'),

            # 交通偏好（自然语言）
            'preferences_in_transportation': self.generate_preference_description(row),

            # 个性特质
            'innate': self.generate_personality_description(row),

            # 通勤属性
            'commute_time_morning': int(row['Q1']) if 'Q1' in row and not pd.isna(row['Q1']) else 30,
            'commute_time_afternoon': int(row['Q2']) if 'Q2' in row and not pd.isna(row['Q2']) else 30,
            'congestion_duration_morning': int(row['Q4']) if 'Q4' in row and not pd.isna(row['Q4']) else 10,
            'congestion_duration_afternoon': int(row['Q5']) if 'Q5' in row and not pd.isna(row['Q5']) else 10,
            'main_route_type': decode_value(row.get('Q3'), self.encodings['route_type'], 'arterial'),

            # 家庭属性
            'household_income': calculate_income_level(int(row.get('Q52', 3)), self.config),
            'home_location': decode_value(row.get('Q57'), self.encodings['location'], 'urban'),
            'work_location': decode_value(row.get('Q58'), self.encodings['location'], 'urban'),

            # 车辆与技术
            'has_navigation': decode_value(row.get('Q56'), self.encodings['has_navigation'], True),
            'licensed_driver': True,  # 问卷对象都是驾驶员

            # 决策关键参数（分钟）
            'delay_tolerance_planned': int(row['Q21']) if 'Q21' in row and not pd.isna(row['Q21']) else 10,
            'delay_tolerance_realtime': int(row['Q42']) if 'Q42' in row and not pd.isna(row['Q42']) else 15,

            # 情境改道阈值（1-5 Likert量表）
            'reroute_threshold_construction': int(row['Q16']) if 'Q16' in row and not pd.isna(row['Q16']) else 3,
            'reroute_threshold_special_event': int(row['Q17']) if 'Q17' in row and not pd.isna(row['Q17']) else 3,
            'reroute_threshold_weather': int(row['Q18']) if 'Q18' in row and not pd.isna(row['Q18']) else 3,
            'reroute_threshold_congestion': int(row['Q19']) if 'Q19' in row and not pd.isna(row['Q19']) else 4,
            'reroute_threshold_accident': int(row['Q20']) if 'Q20' in row and not pd.isna(row['Q20']) else 5,
        }

        # 偏好因子得分
        factor_names = self.config['factor_analysis']['factor_names']
        persona['preference_factors'] = {}
        for factor in factor_names:
            if factor in row and not pd.isna(row[factor]):
                persona['preference_factors'][factor] = float(row[factor])

        # 衍生特征
        if 'congestion_ratio_morning' in row and not pd.isna(row['congestion_ratio_morning']):
            persona['congestion_ratio_morning'] = float(row['congestion_ratio_morning'])
        if 'congestion_ratio_afternoon' in row and not pd.isna(row['congestion_ratio_afternoon']):
            persona['congestion_ratio_afternoon'] = float(row['congestion_ratio_afternoon'])
        if 'delay_tolerance_ratio' in row and not pd.isna(row['delay_tolerance_ratio']):
            persona['delay_tolerance_ratio'] = float(row['delay_tolerance_ratio'])

        return persona

    def generate_all_personas(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        生成所有Persona

        Args:
            df: 合并后的DataFrame

        Returns:
            persona_id -> Persona字典的映射
        """
        self.logger.info(f"Generating {len(df)} personas...")

        personas = {}
        for idx, row in df.iterrows():
            try:
                persona = self.create_persona(row)
                personas[persona['persona_id']] = persona
            except Exception as e:
                self.logger.error(f"Failed to create persona for {row.get('persona_id', idx)}: {e}")

        self.logger.info(f"Successfully generated {len(personas)} personas")

        return personas

    def save_personas(self, personas: Dict[str, Dict], filename: str = "personas.json"):
        """保存Personas到JSON文件"""
        output_dir = self.config['data']['output_dir']
        output_path = f"{output_dir}/{filename}"

        save_json(personas, output_path, indent=2)

        self.logger.info(f"Saved {len(personas)} personas to {output_path}")

        # 计算文件大小
        import os
        file_size = os.path.getsize(output_path) / 1024  # KB
        self.logger.info(f"File size: {file_size:.1f} KB")

    def run(self) -> Dict[str, Dict]:
        """运行完整的Persona生成流程"""
        # 加载数据
        df = self.load_data()

        # 生成Personas
        personas = self.generate_all_personas(df)

        # 保存
        self.save_personas(personas)

        return personas


def main():
    """主函数"""

    print("=" * 80)
    print("阶段1.3：Persona生成器")
    print("=" * 80)

    # 初始化生成器
    generator = PersonaGenerator()

    # 运行
    print("\n[1/1] 生成GATSim格式Personas...")
    personas = generator.run()

    print(f"\n✓ Persona生成完成！")
    print(f"  - 总数: {len(personas)}")

    # 展示示例
    print("\n" + "=" * 80)
    print("Persona示例（前3个）")
    print("=" * 80)

    for i, (persona_id, persona) in enumerate(list(personas.items())[:3]):
        print(f"\n[{i+1}] {persona_id}:")
        print(f"  Name: {persona['name']}")
        print(f"  Age: {persona['age']}, Gender: {persona['gender']}")
        print(f"  Preferences: {persona['preferences_in_transportation'][:80]}...")
        print(f"  Innate: {persona['innate']}")
        print(f"  Delay tolerance: {persona['delay_tolerance_planned']}min (planned), "
              f"{persona['delay_tolerance_realtime']}min (realtime)")

    print("\n" + "=" * 80)
    print("✓ 阶段1.3 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
阶段1.1：数据加载与预处理
功能：
1. 加载四国问卷数据
2. 数据质量检查与过滤
3. 反向题处理
4. 衍生特征计算
5. 数据标准化与合并
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.common import (
    setup_logger,
    load_config,
    ensure_dir,
    QuestionMapper,
    reverse_likert,
    parse_multi_choice,
    generate_unique_id
)


class SurveyDataLoader:
    """问卷数据加载器"""

    def __init__(self, config_path: str = "configs/preference_config.yaml"):
        """
        初始化数据加载器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            'DataLoader',
            f"{self.config['data']['output_dir']}/logs/data_loader.log"
        )

    def load_single_country(self, country: str) -> pd.DataFrame:
        """
        加载单个国家的数据

        Args:
            country: 国家代码 (CN, UK, US, AU)

        Returns:
            DataFrame
        """
        data_dir = self.config['data']['raw_data_dir']
        file_path = f"{data_dir}/{country}_dataset.xlsx"

        self.logger.info(f"Loading data from {file_path}")

        try:
            df = pd.read_excel(file_path)
            self.logger.info(f"Loaded {len(df)} samples from {country}")

            # 检测数据格式：CN是中文问卷格式，UK/US/AU是Qualtrics格式
            if '序号' in df.columns:
                # 中文问卷格式（CN）
                df = self._process_chinese_format(df, country)
            else:
                # Qualtrics格式（UK/US/AU）
                df = self._process_qualtrics_format(df, country)

            return df

        except Exception as e:
            self.logger.error(f"Failed to load {country} data: {e}")
            raise

    def _process_chinese_format(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        """处理中文问卷格式"""
        # 重命名列
        df = QuestionMapper.rename_columns(df)

        # 添加国家标签
        df['country'] = country

        # 生成唯一ID
        df['persona_id'] = df['序号'].apply(
            lambda x: generate_unique_id(country, x)
        )

        return df

    def _process_qualtrics_format(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        """
        处理Qualtrics格式数据（UK/US/AU）

        注意：Qualtrics的第一行是问题描述，第二行开始是数据
        """
        # 删除第一行（问题描述行）
        if len(df) > 0 and df.iloc[0, 0] == 'Start Date':
            df = df.iloc[1:].reset_index(drop=True)
            self.logger.info(f"Removed Qualtrics header row")

        # Qualtrics列名映射（需要根据实际列名调整）
        # TODO: 这里需要您提供Qualtrics格式的列名映射
        # 暂时跳过，因为我们只处理CN数据

        self.logger.warning(f"Qualtrics format for {country} not fully implemented yet. "
                           f"Currently only CN format is supported.")

        # 添加国家标签
        df['country'] = country

        # 使用ResponseId作为序号
        if 'ResponseId' in df.columns:
            df['persona_id'] = df.index.map(
                lambda x: generate_unique_id(country, x)
            )
        else:
            df['persona_id'] = df.index.map(
                lambda x: generate_unique_id(country, x)
            )

        return df

    def load_all_countries(self) -> pd.DataFrame:
        """
        加载所有国家数据并合并

        Returns:
            合并后的DataFrame
        """
        countries = self.config['data']['countries']
        dfs = []

        for country in countries:
            try:
                df = self.load_single_country(country)
                dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Skipping {country}: {e}")

        if not dfs:
            raise ValueError("No data loaded!")

        merged_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Total samples loaded: {len(merged_df)}")

        return merged_df

    def quality_check(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        数据质量检查

        Args:
            df: 原始DataFrame

        Returns:
            (清洗后的DataFrame, 质量报告字典)
        """
        qc_config = self.config['quality_control']
        initial_count = len(df)

        quality_report = {
            'initial_count': initial_count,
            'filters': {}
        }

        # 过滤1：填写时长过短
        if '所用+C1:D851时间' in df.columns:
            min_time = qc_config['min_completion_time']
            mask = df['所用+C1:D851时间'] >= min_time
            filtered_count = (~mask).sum()
            df = df[mask]

            quality_report['filters']['completion_time_too_short'] = filtered_count
            self.logger.info(f"Filtered {filtered_count} samples (completion time < {min_time}s)")

        # 过滤2：注意力检查题错误
        attn_q = qc_config['attention_check_question']
        attn_ans = qc_config['attention_check_answer']

        if attn_q in df.columns:
            mask = df[attn_q] == attn_ans
            filtered_count = (~mask).sum()
            df = df[mask]

            quality_report['filters']['attention_check_failed'] = filtered_count
            self.logger.info(f"Filtered {filtered_count} samples (attention check failed)")

        # 过滤3：通勤时间异常
        max_commute = qc_config['max_commute_time']
        if 'Q1' in df.columns and 'Q2' in df.columns:
            mask = (df['Q1'] <= max_commute) & (df['Q2'] <= max_commute)
            filtered_count = (~mask).sum()
            df = df[mask]

            quality_report['filters']['commute_time_abnormal'] = filtered_count
            self.logger.info(f"Filtered {filtered_count} samples (commute time > {max_commute}min)")

        # 过滤4：Q21和Q42逻辑一致性
        if 'Q21' in df.columns and 'Q42' in df.columns:
            # Q42应该 >= Q21（沉没成本效应）
            # 但允许一定偏差，这里只过滤极端不一致的
            mask = (df['Q42'] / df['Q21'].replace(0, 1)) < 5  # Q42不能超过Q21的5倍
            filtered_count = (~mask).sum()
            df = df[mask]

            quality_report['filters']['delay_threshold_inconsistent'] = filtered_count
            self.logger.info(f"Filtered {filtered_count} samples (Q21/Q42 inconsistent)")

        quality_report['final_count'] = len(df)
        quality_report['total_filtered'] = initial_count - len(df)
        quality_report['retention_rate'] = len(df) / initial_count

        self.logger.info(f"Quality check complete: {len(df)}/{initial_count} samples retained "
                        f"({quality_report['retention_rate']:.1%})")

        return df, quality_report

    def process_reverse_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理反向题

        Args:
            df: DataFrame

        Returns:
            处理后的DataFrame
        """
        # Q23：熟悉地区更愿改道（与Q22相反）
        if 'Q23' in df.columns:
            df['Q23_reversed'] = reverse_likert(df['Q23'], max_value=5)
            self.logger.info("Processed Q23 (reversed)")

        # Q40：我是激进驾驶员（与Q41相反）
        if 'Q40' in df.columns:
            df['Q40_reversed'] = reverse_likert(df['Q40'], max_value=5)
            self.logger.info("Processed Q40 (reversed)")

        return df

    def extract_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取衍生特征

        Args:
            df: DataFrame

        Returns:
            添加衍生特征后的DataFrame
        """
        # 拥堵占比
        if 'Q1' in df.columns and 'Q4' in df.columns:
            df['congestion_ratio_morning'] = df['Q4'] / df['Q1'].replace(0, 1)
            df['congestion_ratio_morning'] = df['congestion_ratio_morning'].clip(0, 1)

        if 'Q2' in df.columns and 'Q5' in df.columns:
            df['congestion_ratio_afternoon'] = df['Q5'] / df['Q2'].replace(0, 1)
            df['congestion_ratio_afternoon'] = df['congestion_ratio_afternoon'].clip(0, 1)

        # 通勤时间不对称性
        if 'Q1' in df.columns and 'Q2' in df.columns:
            df['commute_asymmetry'] = abs(df['Q1'] - df['Q2'])

        # 延误容忍度比率（沉没成本效应指标）
        if 'Q21' in df.columns and 'Q42' in df.columns:
            df['delay_tolerance_ratio'] = df['Q42'] / df['Q21'].replace(0, 1)
            df['delay_tolerance_ratio'] = df['delay_tolerance_ratio'].clip(0, 5)

        # Q10多选题处理：信息渠道
        if 'Q10' in df.columns:
            df['info_channels'] = df['Q10'].apply(parse_multi_choice)
            df['info_channel_count'] = df['info_channels'].apply(len)
            df['use_nav_app'] = df['Q10'].astype(str).str.contains('2', na=False)
            df['use_vms'] = df['Q10'].astype(str).str.contains('3', na=False)

        # 出行类型平均改道意愿
        trip_cols = ['Q43', 'Q44', 'Q45', 'Q46', 'Q47', 'Q48']
        if all(col in df.columns for col in trip_cols):
            df['avg_reroute_willingness'] = df[trip_cols].mean(axis=1)

        self.logger.info(f"Extracted {len([c for c in df.columns if c not in df.columns])} derived features")

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        完整预处理流程

        Args:
            df: 原始DataFrame

        Returns:
            预处理后的DataFrame
        """
        self.logger.info("Starting preprocessing...")

        # 质量检查
        df, quality_report = self.quality_check(df)

        # 反向题处理
        df = self.process_reverse_questions(df)

        # 衍生特征
        df = self.extract_derived_features(df)

        # 缺失值处理（保留，后续在因子分析时处理）
        missing_stats = df.isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]
        if len(missing_cols) > 0:
            self.logger.warning(f"Columns with missing values:\n{missing_cols}")

        self.logger.info("Preprocessing complete!")

        return df

    def save(self, df: pd.DataFrame, filename: str = "cleaned_survey_data.csv"):
        """
        保存清洗后的数据

        Args:
            df: DataFrame
            filename: 保存文件名
        """
        output_dir = self.config['data']['output_dir']
        ensure_dir(output_dir)

        output_path = f"{output_dir}/{filename}"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        self.logger.info(f"Saved cleaned data to {output_path}")
        self.logger.info(f"Shape: {df.shape}")

        return output_path


def main():
    """主函数：运行完整的数据加载与预处理流程"""

    print("=" * 80)
    print("阶段1.1：数据加载与预处理")
    print("=" * 80)

    # 初始化加载器
    loader = SurveyDataLoader()

    # 加载数据
    print("\n[1/3] 加载四国问卷数据...")
    df_raw = loader.load_all_countries()
    print(f"✓ 加载完成：{len(df_raw)} 条样本")

    # 预处理
    print("\n[2/3] 数据预处理（质量检查、反向题、衍生特征）...")
    df_clean = loader.preprocess(df_raw)
    print(f"✓ 预处理完成：保留 {len(df_clean)} 条样本")

    # 保存
    print("\n[3/3] 保存清洗后的数据...")
    output_path = loader.save(df_clean)
    print(f"✓ 保存成功：{output_path}")

    # 统计摘要
    print("\n" + "=" * 80)
    print("数据摘要")
    print("=" * 80)
    print(f"总样本数: {len(df_clean)}")
    print(f"国家分布:")
    print(df_clean['country'].value_counts())
    print(f"\n性别分布:")
    if 'Q49' in df_clean.columns:
        print(df_clean['Q49'].value_counts())
    print(f"\n年龄统计:")
    if 'Q50' in df_clean.columns:
        print(df_clean['Q50'].describe())

    print("\n" + "=" * 80)
    print("✓ 阶段1.1 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

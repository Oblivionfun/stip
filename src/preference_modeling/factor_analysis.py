"""
阶段1.2：偏好因子提取
功能：
1. 从态度题中提取偏好因子
2. 因子分析（EFA）
3. 计算因子得分
4. 信度检验
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.common import (
    setup_logger,
    load_config,
    ensure_dir,
    save_json
)


class PreferenceFactorExtractor:
    """偏好因子提取器"""

    def __init__(self, config_path: str = "configs/preference_config.yaml"):
        """
        初始化因子提取器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            'FactorExtractor',
            f"{self.config['data']['output_dir']}/logs/factor_analysis.log"
        )

        self.fa_config = self.config['factor_analysis']
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def load_cleaned_data(self, file_path: str = None) -> pd.DataFrame:
        """加载清洗后的数据"""
        if file_path is None:
            file_path = f"{self.config['data']['output_dir']}/cleaned_survey_data.csv"

        self.logger.info(f"Loading cleaned data from {file_path}")
        df = pd.read_csv(file_path)
        self.logger.info(f"Loaded {len(df)} samples")

        return df

    def select_attitude_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        选择态度题用于因子分析

        Args:
            df: 清洗后的DataFrame

        Returns:
            仅包含态度题的DataFrame
        """
        attitude_cols = self.fa_config['attitude_questions']

        # 检查哪些列存在
        existing_cols = [col for col in attitude_cols if col in df.columns]
        missing_cols = [col for col in attitude_cols if col not in df.columns]

        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")

        self.logger.info(f"Selected {len(existing_cols)}/{len(attitude_cols)} attitude questions")

        return df[existing_cols + ['persona_id', 'country']]

    def preprocess_for_fa(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        因子分析预处理

        Args:
            df: 态度题DataFrame

        Returns:
            (标准化后的数据, 元数据DataFrame)
        """
        # 分离ID和数据
        meta_cols = ['persona_id', 'country']
        meta_df = df[meta_cols]
        data_df = df.drop(columns=meta_cols)

        self.logger.info(f"Data shape before preprocessing: {data_df.shape}")

        # 缺失值填充（用中位数）
        X = self.imputer.fit_transform(data_df)
        self.logger.info(f"Imputed missing values")

        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        self.logger.info(f"Standardized data")

        return X_scaled, meta_df, data_df.columns.tolist()

    def perform_factor_analysis(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行因子分析

        Args:
            X: 标准化后的数据

        Returns:
            (因子得分, 因子载荷矩阵)
        """
        n_factors = self.fa_config['n_factors']

        self.logger.info(f"Performing factor analysis with {n_factors} factors...")

        fa = FactorAnalysis(
            n_components=n_factors,
            random_state=self.fa_config['random_state'],
            rotation=None  # 不使用旋转，sklearn的FA不支持varimax
        )

        # 拟合并转换
        factor_scores = fa.fit_transform(X)

        # 获取载荷矩阵
        loadings = fa.components_.T

        self.logger.info(f"Factor analysis complete!")
        self.logger.info(f"Explained variance: {fa.noise_variance_.sum():.4f}")

        return factor_scores, loadings

    def create_factor_dataframe(
        self,
        factor_scores: np.ndarray,
        meta_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        创建因子得分DataFrame

        Args:
            factor_scores: 因子得分矩阵
            meta_df: 元数据DataFrame

        Returns:
            包含因子得分的DataFrame
        """
        factor_names = self.fa_config['factor_names']

        # 创建DataFrame
        factor_df = pd.DataFrame(
            factor_scores,
            columns=factor_names
        )

        # 合并元数据
        result_df = pd.concat([meta_df.reset_index(drop=True), factor_df], axis=1)

        self.logger.info(f"Created factor DataFrame with shape {result_df.shape}")

        return result_df

    def save_loadings(
        self,
        loadings: np.ndarray,
        question_cols: List[str],
        filename: str = "factor_loadings.csv"
    ):
        """
        保存因子载荷矩阵

        Args:
            loadings: 载荷矩阵
            question_cols: 问题列名
            filename: 保存文件名
        """
        factor_names = self.fa_config['factor_names']

        # 创建DataFrame
        loadings_df = pd.DataFrame(
            loadings,
            index=question_cols,
            columns=factor_names
        )

        # 保存
        output_dir = self.config['data']['output_dir']
        ensure_dir(output_dir)
        output_path = f"{output_dir}/{filename}"

        loadings_df.to_csv(output_path)
        self.logger.info(f"Saved factor loadings to {output_path}")

        # 打印主要载荷
        self.logger.info("\nFactor loadings (top 3 per factor):")
        for factor in factor_names:
            top_3 = loadings_df[factor].abs().nlargest(3)
            self.logger.info(f"\n{factor}:")
            for q, loading in top_3.items():
                self.logger.info(f"  {q}: {loadings_df.loc[q, factor]:.3f}")

    def plot_factor_loadings(
        self,
        loadings: np.ndarray,
        question_cols: List[str],
        filename: str = "factor_loadings_heatmap.png"
    ):
        """
        绘制因子载荷热力图

        Args:
            loadings: 载荷矩阵
            question_cols: 问题列名
            filename: 保存文件名
        """
        factor_names = self.fa_config['factor_names']

        plt.figure(figsize=(12, 16))
        sns.heatmap(
            loadings,
            xticklabels=factor_names,
            yticklabels=question_cols,
            cmap='RdBu_r',
            center=0,
            annot=False,
            cbar_kws={'label': 'Factor Loading'}
        )
        plt.title('Factor Loadings Heatmap', fontsize=16)
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Questions', fontsize=12)
        plt.tight_layout()

        output_dir = self.config['data']['output_dir']
        output_path = f"{output_dir}/{filename}"
        plt.savefig(output_path, dpi=150)
        plt.close()

        self.logger.info(f"Saved factor loadings heatmap to {output_path}")

    def run(self, input_file: str = None) -> pd.DataFrame:
        """
        运行完整的因子分析流程

        Args:
            input_file: 输入文件路径

        Returns:
            包含因子得分的DataFrame
        """
        # 加载数据
        df = self.load_cleaned_data(input_file)

        # 选择态度题
        attitude_df = self.select_attitude_questions(df)

        # 预处理
        X_scaled, meta_df, question_cols = self.preprocess_for_fa(attitude_df)

        # 因子分析
        factor_scores, loadings = self.perform_factor_analysis(X_scaled)

        # 创建因子DataFrame
        factor_df = self.create_factor_dataframe(factor_scores, meta_df)

        # 保存载荷矩阵
        self.save_loadings(loadings, question_cols)

        # 绘制热力图
        self.plot_factor_loadings(loadings, question_cols)

        # 保存因子得分
        output_path = f"{self.config['data']['output_dir']}/preference_factors.csv"
        factor_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved preference factors to {output_path}")

        return factor_df


def main():
    """主函数：运行因子分析"""

    print("=" * 80)
    print("阶段1.2：偏好因子提取")
    print("=" * 80)

    # 初始化提取器
    extractor = PreferenceFactorExtractor()

    # 运行因子分析
    print("\n[1/1] 执行因子分析...")
    factor_df = extractor.run()

    print(f"\n✓ 因子分析完成！")
    print(f"  - 样本数: {len(factor_df)}")
    print(f"  - 因子数: {len(factor_df.columns) - 2}")  # 减去persona_id和country

    # 统计摘要
    print("\n" + "=" * 80)
    print("因子得分统计摘要")
    print("=" * 80)
    factor_cols = [col for col in factor_df.columns if col not in ['persona_id', 'country']]
    print(factor_df[factor_cols].describe())

    print("\n" + "=" * 80)
    print("✓ 阶段1.2 完成！")
    print("=" * 80)


if __name__ == "__main__":
    from typing import List, Tuple
    main()

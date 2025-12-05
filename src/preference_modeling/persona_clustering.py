"""
阶段1.4：Persona聚类
功能：
1. K-Means聚类（基于因子得分）
2. 分配类型标签
3. 生成可视化（PCA降维）
4. 统计分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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


class PersonaClusterer:
    """Persona聚类器"""

    def __init__(self, config_path: str = "configs/preference_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(
            'PersonaClusterer',
            f"{self.config['data']['output_dir']}/logs/persona_clustering.log"
        )
        self.cluster_config = self.config['clustering']

    def load_factors(self) -> pd.DataFrame:
        """加载因子得分"""
        output_dir = self.config['data']['output_dir']
        factors_df = pd.read_csv(f"{output_dir}/preference_factors.csv")

        self.logger.info(f"Loaded {len(factors_df)} factor scores")

        return factors_df

    def perform_clustering(self, X: np.ndarray) -> tuple:
        """
        执行K-Means聚类

        Args:
            X: 因子得分矩阵

        Returns:
            (聚类标签, KMeans模型, 轮廓系数)
        """
        n_clusters = self.cluster_config['n_clusters']

        self.logger.info(f"Performing K-Means clustering with {n_clusters} clusters...")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.cluster_config['random_state'],
            n_init=10
        )

        labels = kmeans.fit_predict(X)

        # 计算轮廓系数
        silhouette = silhouette_score(X, labels)

        self.logger.info(f"Clustering complete!")
        self.logger.info(f"Silhouette score: {silhouette:.4f}")

        return labels, kmeans, silhouette

    def assign_type_labels(self, labels: np.ndarray, factor_df: pd.DataFrame) -> Dict[int, str]:
        """
        为聚类分配类型标签（基于cluster的因子特征）

        Args:
            labels: 聚类标签
            factor_df: 因子DataFrame

        Returns:
            cluster_id -> 类型标签的映射
        """
        factor_names = self.config['factor_analysis']['factor_names']
        factor_cols = [col for col in factor_df.columns if col in factor_names]

        # 计算每个cluster的因子均值
        cluster_profiles = {}
        for cluster_id in range(self.cluster_config['n_clusters']):
            mask = labels == cluster_id
            cluster_data = factor_df[mask][factor_cols]
            cluster_mean = cluster_data.mean()

            cluster_profiles[cluster_id] = {
                'size': mask.sum(),
                'mean_factors': cluster_mean.to_dict(),
                'dominant_factors': cluster_mean.abs().nlargest(3).to_dict()
            }

        # 打印cluster特征
        self.logger.info("\nCluster Profiles:")
        for cluster_id, profile in cluster_profiles.items():
            self.logger.info(f"\nCluster {cluster_id} (n={profile['size']}):")
            self.logger.info("  Top 3 dominant factors:")
            for factor, value in profile['dominant_factors'].items():
                self.logger.info(f"    {factor}: {value:.3f}")

        # 使用配置中的类型标签
        persona_types = self.cluster_config['persona_types']

        return persona_types

    def create_type_mapping(
        self,
        labels: np.ndarray,
        persona_ids: List[str],
        type_labels: Dict[int, str]
    ) -> Dict[str, Dict]:
        """
        创建persona_id到类型标签的映射

        Args:
            labels: 聚类标签数组
            persona_ids: persona_id列表
            type_labels: cluster_id到类型标签的映射

        Returns:
            persona_id -> {cluster, type_label}的映射
        """
        mapping = {}
        for persona_id, label in zip(persona_ids, labels):
            mapping[persona_id] = {
                'cluster': int(label),
                'type_label': type_labels[label]
            }

        return mapping

    def visualize_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        filename: str = "persona_clustering.png"
    ):
        """
        使用PCA降维可视化聚类结果

        Args:
            X: 因子得分矩阵
            labels: 聚类标签
            filename: 保存文件名
        """
        # PCA降到2维
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        # 绘图
        plt.figure(figsize=(12, 8))

        scatter = plt.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=50
        )

        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Persona Clustering Visualization (PCA)')
        plt.grid(True, alpha=0.3)

        # 标注cluster中心
        for cluster_id in range(self.cluster_config['n_clusters']):
            mask = labels == cluster_id
            center = X_2d[mask].mean(axis=0)
            plt.scatter(center[0], center[1], c='red', s=200, marker='*', edgecolors='black', linewidths=2)
            plt.text(center[0], center[1], f'C{cluster_id}', fontsize=12, ha='center', va='center')

        plt.tight_layout()

        output_dir = self.config['data']['output_dir']
        output_path = f"{output_dir}/{filename}"
        plt.savefig(output_path, dpi=150)
        plt.close()

        self.logger.info(f"Saved clustering visualization to {output_path}")

    def plot_cluster_distributions(
        self,
        factor_df: pd.DataFrame,
        labels: np.ndarray,
        filename: str = "cluster_distributions.png"
    ):
        """绘制每个cluster的因子分布"""
        factor_names = self.config['factor_analysis']['factor_names']
        factor_cols = [col for col in factor_df.columns if col in factor_names]

        n_factors = len(factor_cols)
        n_cols = 3
        n_rows = (n_factors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()

        for idx, factor in enumerate(factor_cols):
            ax = axes[idx]

            # 为每个cluster绘制分布
            for cluster_id in range(self.cluster_config['n_clusters']):
                mask = labels == cluster_id
                data = factor_df[mask][factor]
                ax.hist(data, alpha=0.5, label=f'Cluster {cluster_id}', bins=20)

            ax.set_xlabel(factor)
            ax.set_ylabel('Count')
            ax.set_title(f'{factor} Distribution by Cluster')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 移除多余的子图
        for idx in range(n_factors, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        output_dir = self.config['data']['output_dir']
        output_path = f"{output_dir}/{filename}"
        plt.savefig(output_path, dpi=150)
        plt.close()

        self.logger.info(f"Saved cluster distributions to {output_path}")

    def run(self) -> Dict:
        """运行完整的聚类流程"""
        # 加载因子得分
        factor_df = self.load_factors()

        # 提取因子矩阵
        factor_names = self.config['factor_analysis']['factor_names']
        factor_cols = [col for col in factor_df.columns if col in factor_names]
        X = factor_df[factor_cols].values

        # 聚类
        labels, kmeans, silhouette = self.perform_clustering(X)

        # 分配类型标签
        type_labels = self.assign_type_labels(labels, factor_df)

        # 创建映射
        persona_ids = factor_df['persona_id'].tolist()
        type_mapping = self.create_type_mapping(labels, persona_ids, type_labels)

        # 可视化
        self.visualize_clusters(X, labels)
        self.plot_cluster_distributions(factor_df, labels)

        # 保存
        output_dir = self.config['data']['output_dir']
        save_json(type_mapping, f"{output_dir}/persona_types.json", indent=2)
        self.logger.info(f"Saved persona type mapping")

        # 统计
        cluster_stats = {
            'n_clusters': self.cluster_config['n_clusters'],
            'silhouette_score': float(silhouette),
            'cluster_sizes': {}
        }

        for cluster_id in range(self.cluster_config['n_clusters']):
            count = (labels == cluster_id).sum()
            cluster_stats['cluster_sizes'][cluster_id] = int(count)

        save_json(cluster_stats, f"{output_dir}/cluster_statistics.json", indent=2)

        return type_mapping


def main():
    """主函数"""

    print("=" * 80)
    print("阶段1.4：Persona聚类")
    print("=" * 80)

    # 初始化聚类器
    clusterer = PersonaClusterer()

    # 运行
    print("\n[1/1] 执行K-Means聚类...")
    type_mapping = clusterer.run()

    print(f"\n✓ 聚类完成！")
    print(f"  - 总样本数: {len(type_mapping)}")
    print(f"  - 聚类数: {clusterer.cluster_config['n_clusters']}")

    # 统计每个cluster的数量
    print("\n" + "=" * 80)
    print("聚类分布")
    print("=" * 80)

    cluster_counts = {}
    for info in type_mapping.values():
        cluster_id = info['cluster']
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        pct = count / len(type_mapping) * 100
        type_label = clusterer.cluster_config['persona_types'][cluster_id]
        print(f"Cluster {cluster_id} ({type_label}): {count} ({pct:.1f}%)")

    print("\n" + "=" * 80)
    print("✓ 阶段1.4 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

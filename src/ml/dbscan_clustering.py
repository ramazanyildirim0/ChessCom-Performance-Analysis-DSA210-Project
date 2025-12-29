"""
DBSCAN Clustering for Chess Game Pattern Discovery
Density-based clustering to find game groups and identify outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, VISUALIZATIONS_DIR, USERNAME


class ChessDBSCAN:
    """DBSCAN clustering for chess game pattern discovery"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.viz_dir = os.path.join(VISUALIZATIONS_DIR, 'ml')
        os.makedirs(self.viz_dir, exist_ok=True)

    def prepare_features(self):
        """Prepare features for clustering"""
        df = self.df.copy()

        # Encode player color (White=1, Black=0)
        df['color_encoded'] = (df['player_color'] == 'White').astype(int)

        # Encode time_class
        time_map = {'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3}
        df['time_encoded'] = df['time_class'].map(time_map).fillna(1)

        # Select features for clustering
        feature_cols = ['rating_diff', 'move_count', 'hour', 'time_encoded',
                        'player_rating', 'opponent_rating']

        self.feature_names = feature_cols
        X = df[feature_cols].fillna(0).astype(float)

        return X, df

    def find_optimal_eps(self, X_scaled, k=5):
        """Find optimal epsilon using k-distance graph"""
        # Compute k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)

        # Get the k-th nearest neighbor distance for each point
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)

        # Find elbow point (where curve starts to flatten)
        # Using second derivative
        diffs = np.diff(k_distances)
        diffs2 = np.diff(diffs)

        # Find the point of maximum curvature
        elbow_idx = np.argmax(diffs2) + 2 if len(diffs2) > 0 else len(k_distances) // 2
        optimal_eps = k_distances[elbow_idx]

        return {
            'k_distances': k_distances,
            'optimal_eps': optimal_eps,
            'elbow_idx': elbow_idx
        }

    def train_model(self, X, eps=None, min_samples=5):
        """Train DBSCAN model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal epsilon if not provided
        eps_analysis = self.find_optimal_eps(X_scaled, k=min_samples)
        if eps is None:
            eps = eps_analysis['optimal_eps']
        print(f"  Using epsilon: {eps:.4f}")

        # Train DBSCAN
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.model.fit_predict(X_scaled)

        # Count clusters and noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Calculate silhouette score (excluding noise points)
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil_score = silhouette_score(X_scaled[mask], labels[mask])
            else:
                sil_score = np.nan
        else:
            sil_score = np.nan

        results = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_pct': n_noise / len(labels) * 100,
            'labels': labels,
            'silhouette_score': sil_score,
            'eps_analysis': eps_analysis,
            'X_scaled': X_scaled
        }

        return results

    def analyze_clusters(self, df, labels):
        """Analyze characteristics of each cluster including noise"""
        df = df.copy()
        df['cluster'] = labels

        cluster_stats = []
        unique_labels = sorted(set(labels))

        for cluster_id in unique_labels:
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_name = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'

            stats = {
                'cluster': cluster_name,
                'cluster_id': cluster_id,
                'size': len(cluster_df),
                'pct': len(cluster_df) / len(df) * 100,
                'avg_rating_diff': cluster_df['rating_diff'].mean(),
                'avg_move_count': cluster_df['move_count'].mean(),
                'avg_hour': cluster_df['hour'].mean(),
                'win_rate': (cluster_df['result'] == 'Win').mean() * 100,
                'dominant_time_class': cluster_df['time_class'].mode().iloc[0] if len(cluster_df) > 0 else 'N/A',
                'avg_player_rating': cluster_df['player_rating'].mean()
            }
            cluster_stats.append(stats)

        return pd.DataFrame(cluster_stats)

    def plot_eps_selection(self, results):
        """Plot k-distance graph for epsilon selection"""
        fig, ax = plt.subplots(figsize=(10, 6))

        eps_analysis = results['eps_analysis']
        k_distances = eps_analysis['k_distances']

        ax.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1)
        ax.axhline(y=results['eps'], color='r', linestyle='--',
                   label=f'Optimal ε = {results["eps"]:.4f}')
        ax.axvline(x=eps_analysis['elbow_idx'], color='g', linestyle=':',
                   alpha=0.7, label=f'Elbow point')

        ax.set_xlabel('Points (sorted by distance)', fontsize=12)
        ax.set_ylabel('k-Distance', fontsize=12)
        ax.set_title('DBSCAN - k-Distance Graph for Epsilon Selection',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'dbscan_eps_selection.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_cluster_visualization(self, results, df):
        """Plot DBSCAN clusters in 2D using PCA"""
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(results['X_scaled'])

        labels = results['labels']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Clusters with noise
        ax = axes[0]
        unique_labels = set(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(sorted(unique_labels), colors):
            if k == -1:
                # Noise points in black
                col = 'black'
                marker = 'x'
                label = f'Noise ({(labels == k).sum()} points)'
                alpha = 0.5
                s = 30
            else:
                marker = 'o'
                label = f'Cluster {k} ({(labels == k).sum()} points)'
                alpha = 0.6
                s = 20

            mask = labels == k
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[col], marker=marker,
                       s=s, alpha=alpha, label=label)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title(f'DBSCAN Clusters (ε={results["eps"]:.3f})',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

        # Color by result
        ax = axes[1]
        result_colors = {'Win': 'green', 'Draw': 'gray', 'Loss': 'red'}
        for result, color in result_colors.items():
            mask = df['result'] == result
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.5, s=20, label=result)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('Games Colored by Result', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'dbscan_clusters.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_cluster_comparison(self, cluster_stats):
        """Plot cluster characteristics comparison"""
        # Filter out noise for comparison if there are other clusters
        plot_stats = cluster_stats.copy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        n_clusters = len(plot_stats)
        colors = ['black' if cid == -1 else plt.cm.viridis(i / max(1, n_clusters - 1))
                  for i, cid in enumerate(plot_stats['cluster_id'])]

        # Cluster sizes
        ax = axes[0, 0]
        bars = ax.bar(range(len(plot_stats)), plot_stats['size'], color=colors)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Number of Games', fontsize=12)
        ax.set_title('Cluster/Noise Sizes', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(plot_stats)))
        ax.set_xticklabels(plot_stats['cluster'], rotation=45, ha='right')

        # Win rates
        ax = axes[0, 1]
        bars = ax.bar(range(len(plot_stats)), plot_stats['win_rate'], color=colors)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50%')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Win Rate by Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(plot_stats)))
        ax.set_xticklabels(plot_stats['cluster'], rotation=45, ha='right')
        ax.legend()

        # Average rating diff
        ax = axes[1, 0]
        bars = ax.bar(range(len(plot_stats)), plot_stats['avg_rating_diff'], color=colors)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Average Rating Difference', fontsize=12)
        ax.set_title('Average Rating Diff by Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(plot_stats)))
        ax.set_xticklabels(plot_stats['cluster'], rotation=45, ha='right')

        # Average move count
        ax = axes[1, 1]
        ax.bar(range(len(plot_stats)), plot_stats['avg_move_count'], color=colors)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Average Move Count', fontsize=12)
        ax.set_title('Average Game Length by Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(plot_stats)))
        ax.set_xticklabels(plot_stats['cluster'], rotation=45, ha='right')

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'dbscan_cluster_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def run_analysis(self):
        """Run complete DBSCAN analysis"""
        print("\n" + "="*60)
        print("DBSCAN CLUSTERING")
        print("="*60)

        # Prepare data
        X, df = self.prepare_features()
        print(f"\nDataset: {len(X)} games, {len(self.feature_names)} features")
        print(f"Features: {', '.join(self.feature_names)}")

        # Train model
        print("\nFinding optimal epsilon and training DBSCAN...")
        results = self.train_model(X, min_samples=10)

        print(f"\nDBSCAN Results:")
        print(f"  Epsilon: {results['eps']:.4f}")
        print(f"  Min Samples: {results['min_samples']}")
        print(f"  Number of Clusters: {results['n_clusters']}")
        print(f"  Noise Points: {results['n_noise']} ({results['noise_pct']:.1f}%)")
        if not np.isnan(results['silhouette_score']):
            print(f"  Silhouette Score: {results['silhouette_score']:.4f}")

        # Analyze clusters
        print("\nAnalyzing clusters...")
        cluster_stats = self.analyze_clusters(df, results['labels'])
        results['cluster_stats'] = cluster_stats

        print("\nCluster Summary:")
        print(cluster_stats.to_string(index=False))

        # Cluster interpretation
        print("\nCluster Interpretation:")
        for _, row in cluster_stats.iterrows():
            if row['cluster_id'] == -1:
                print(f"  Noise: {int(row['size'])} outlier games ({row['pct']:.1f}%), "
                      f"Win Rate: {row['win_rate']:.1f}%")
            else:
                rating_type = "higher" if row['avg_rating_diff'] > 0 else "lower"
                game_length = "long" if row['avg_move_count'] > 70 else "short"
                print(f"  {row['cluster']}: {int(row['size'])} games ({row['pct']:.1f}%), "
                      f"Win Rate: {row['win_rate']:.1f}%, "
                      f"Playing {rating_type}-rated opponents, {game_length} games")

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_eps_selection(results)
        self.plot_cluster_visualization(results, df)
        self.plot_cluster_comparison(cluster_stats)

        return results


def run_dbscan(df: pd.DataFrame = None) -> dict:
    """Main function to run DBSCAN analysis"""
    if df is None:
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        df = pd.read_csv(csv_path)

    dbscan = ChessDBSCAN(df)
    results = dbscan.run_analysis()
    return results


if __name__ == "__main__":
    results = run_dbscan()

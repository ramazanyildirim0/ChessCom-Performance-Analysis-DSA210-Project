"""
K-Means Clustering for Chess Game Pattern Discovery
Groups games into clusters to discover hidden playing patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, VISUALIZATIONS_DIR, USERNAME


class ChessKMeans:
    """K-Means clustering for chess game pattern discovery"""

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

        # Encode result for analysis (not used in clustering)
        result_map = {'Win': 1, 'Draw': 0.5, 'Loss': 0}
        df['result_encoded'] = df['result'].map(result_map)

        # Select features for clustering
        feature_cols = ['rating_diff', 'move_count', 'hour', 'time_encoded',
                        'player_rating', 'opponent_rating']

        self.feature_names = feature_cols
        X = df[feature_cols].fillna(0).astype(float)

        return X, df

    def find_optimal_k(self, X_scaled, k_range=range(2, 11)):
        """Find optimal K using elbow method and silhouette score"""
        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

        # Find elbow point using second derivative
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_k = list(k_range)[np.argmax(diffs2) + 2]

        # Also consider best silhouette score
        best_silhouette_k = list(k_range)[np.argmax(silhouette_scores)]

        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'elbow_k': elbow_k,
            'best_silhouette_k': best_silhouette_k
        }

    def train_model(self, X, n_clusters=4):
        """Train K-Means model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal K
        k_analysis = self.find_optimal_k(X_scaled)
        optimal_k = k_analysis['best_silhouette_k']
        print(f"  Optimal K (silhouette): {optimal_k}")
        print(f"  Elbow K: {k_analysis['elbow_k']}")

        # Train with optimal K
        self.model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = self.model.fit_predict(X_scaled)

        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, labels)

        results = {
            'n_clusters': optimal_k,
            'labels': labels,
            'silhouette_score': sil_score,
            'inertia': self.model.inertia_,
            'cluster_centers': self.model.cluster_centers_,
            'k_analysis': k_analysis,
            'X_scaled': X_scaled
        }

        return results

    def analyze_clusters(self, df, labels):
        """Analyze characteristics of each cluster"""
        df = df.copy()
        df['cluster'] = labels

        cluster_stats = []
        for cluster_id in range(len(np.unique(labels))):
            cluster_df = df[df['cluster'] == cluster_id]

            stats = {
                'cluster': cluster_id,
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

    def plot_elbow_silhouette(self, results):
        """Plot elbow method and silhouette scores"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        k_analysis = results['k_analysis']

        # Elbow plot
        ax = axes[0]
        ax.plot(k_analysis['k_range'], k_analysis['inertias'], 'b-o', linewidth=2, markersize=8)
        ax.axvline(x=k_analysis['elbow_k'], color='r', linestyle='--',
                   label=f'Elbow K = {k_analysis["elbow_k"]}')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Inertia (Within-cluster SSE)', fontsize=12)
        ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Silhouette plot
        ax = axes[1]
        ax.plot(k_analysis['k_range'], k_analysis['silhouette_scores'], 'g-o', linewidth=2, markersize=8)
        ax.axvline(x=k_analysis['best_silhouette_k'], color='r', linestyle='--',
                   label=f'Best K = {k_analysis["best_silhouette_k"]}')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Silhouette Score for Optimal K', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'kmeans_elbow_silhouette.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_cluster_visualization(self, results, df):
        """Plot clusters in 2D using PCA"""
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(results['X_scaled'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Clusters
        ax = axes[0]
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=results['labels'],
                             cmap='viridis', alpha=0.6, s=20)
        # Plot centroids
        centers_2d = pca.transform(results['cluster_centers'])
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X',
                   s=200, edgecolors='black', linewidths=2, label='Centroids')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('K-Means Clusters (PCA Projection)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.legend()

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
        filepath = os.path.join(self.viz_dir, 'kmeans_clusters.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_cluster_analysis(self, cluster_stats):
        """Plot cluster characteristics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        n_clusters = len(cluster_stats)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_clusters))

        # Cluster sizes
        ax = axes[0, 0]
        ax.bar(cluster_stats['cluster'], cluster_stats['size'], color=colors)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Number of Games', fontsize=12)
        ax.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
        for i, (cluster, size) in enumerate(zip(cluster_stats['cluster'], cluster_stats['size'])):
            ax.text(cluster, size + 20, f'{size}', ha='center', fontsize=10)

        # Win rates
        ax = axes[0, 1]
        bars = ax.bar(cluster_stats['cluster'], cluster_stats['win_rate'], color=colors)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50%')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Win Rate by Cluster', fontsize=14, fontweight='bold')
        ax.legend()
        for i, (cluster, wr) in enumerate(zip(cluster_stats['cluster'], cluster_stats['win_rate'])):
            ax.text(cluster, wr + 1, f'{wr:.1f}%', ha='center', fontsize=10)

        # Average rating diff
        ax = axes[1, 0]
        bars = ax.bar(cluster_stats['cluster'], cluster_stats['avg_rating_diff'], color=colors)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Average Rating Difference', fontsize=12)
        ax.set_title('Average Rating Diff by Cluster', fontsize=14, fontweight='bold')

        # Average move count
        ax = axes[1, 1]
        ax.bar(cluster_stats['cluster'], cluster_stats['avg_move_count'], color=colors)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Average Move Count', fontsize=12)
        ax.set_title('Average Game Length by Cluster', fontsize=14, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'kmeans_cluster_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def run_analysis(self):
        """Run complete K-Means analysis"""
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING")
        print("="*60)

        # Prepare data
        X, df = self.prepare_features()
        print(f"\nDataset: {len(X)} games, {len(self.feature_names)} features")
        print(f"Features: {', '.join(self.feature_names)}")

        # Train model
        print("\nFinding optimal K and training K-Means...")
        results = self.train_model(X)

        print(f"\nNumber of clusters: {results['n_clusters']}")
        print(f"Silhouette Score: {results['silhouette_score']:.4f}")
        print(f"Inertia: {results['inertia']:.2f}")

        # Analyze clusters
        print("\nAnalyzing clusters...")
        cluster_stats = self.analyze_clusters(df, results['labels'])
        results['cluster_stats'] = cluster_stats

        print("\nCluster Summary:")
        print(cluster_stats.to_string(index=False))

        # Cluster interpretation
        print("\nCluster Interpretation:")
        for _, row in cluster_stats.iterrows():
            rating_type = "higher" if row['avg_rating_diff'] > 0 else "lower"
            game_length = "long" if row['avg_move_count'] > 70 else "short"
            print(f"  Cluster {int(row['cluster'])}: {int(row['size'])} games ({row['pct']:.1f}%), "
                  f"Win Rate: {row['win_rate']:.1f}%, "
                  f"Playing {rating_type}-rated opponents, {game_length} games, "
                  f"Mostly {row['dominant_time_class']}")

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_elbow_silhouette(results)
        self.plot_cluster_visualization(results, df)
        self.plot_cluster_analysis(cluster_stats)

        return results


def run_kmeans(df: pd.DataFrame = None) -> dict:
    """Main function to run K-Means analysis"""
    if df is None:
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        df = pd.read_csv(csv_path)

    kmeans = ChessKMeans(df)
    results = kmeans.run_analysis()
    return results


if __name__ == "__main__":
    results = run_kmeans()

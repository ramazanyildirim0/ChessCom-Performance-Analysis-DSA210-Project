"""
Principal Component Analysis (PCA) for Chess Game Data
Dimensionality reduction and visualization of game patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, VISUALIZATIONS_DIR, USERNAME


class ChessPCA:
    """PCA analysis for chess game data dimensionality reduction"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.viz_dir = os.path.join(VISUALIZATIONS_DIR, 'ml')
        os.makedirs(self.viz_dir, exist_ok=True)

    def prepare_features(self):
        """Prepare features for PCA"""
        df = self.df.copy()

        # Encode player color (White=1, Black=0)
        df['color_encoded'] = (df['player_color'] == 'White').astype(int)

        # Encode day of week
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df['day_encoded'] = df['day_of_week'].map(day_map).fillna(0)

        # Encode time_class
        time_map = {'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3}
        df['time_encoded'] = df['time_class'].map(time_map).fillna(1)

        # Encode result
        result_map = {'Win': 1, 'Draw': 0.5, 'Loss': 0}
        df['result_encoded'] = df['result'].map(result_map)

        # Select all numeric features
        feature_cols = ['rating_diff', 'move_count', 'hour', 'day_encoded',
                        'time_encoded', 'player_rating', 'opponent_rating', 'color_encoded']

        self.feature_names = feature_cols
        X = df[feature_cols].fillna(0).astype(float)

        return X, df

    def perform_pca(self, X, n_components=None):
        """Perform PCA on the data"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform PCA with all components first to analyze variance
        if n_components is None:
            n_components = min(len(self.feature_names), len(X))

        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        # Calculate cumulative variance
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        results = {
            'X_pca': X_pca,
            'X_scaled': X_scaled,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'components': self.pca.components_,
            'n_components_95': n_components_95,
            'total_variance_explained': cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
        }

        return results

    def get_loadings(self):
        """Get component loadings (correlation between features and components)"""
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )
        return loadings

    def plot_explained_variance(self, results):
        """Plot explained variance ratio"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n_components = len(results['explained_variance_ratio'])

        # Individual variance
        ax = axes[0]
        bars = ax.bar(range(1, n_components + 1), results['explained_variance_ratio'] * 100,
                      color='steelblue', alpha=0.8)
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance (%)', fontsize=12)
        ax.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, n_components + 1))

        # Add percentage labels
        for bar, var in zip(bars, results['explained_variance_ratio']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{var*100:.1f}%', ha='center', va='bottom', fontsize=9)

        # Cumulative variance
        ax = axes[1]
        ax.plot(range(1, n_components + 1), results['cumulative_variance'] * 100,
                'b-o', linewidth=2, markersize=8)
        ax.axhline(y=95, color='r', linestyle='--', label='95% threshold')
        ax.axvline(x=results['n_components_95'], color='g', linestyle=':',
                   label=f'{results["n_components_95"]} components for 95%')
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
        ax.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, n_components + 1))
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'pca_explained_variance.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_2d_projection(self, results, df):
        """Plot 2D PCA projection colored by result"""
        X_pca = results['X_pca']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Color by result
        ax = axes[0]
        result_colors = {'Win': 'green', 'Draw': 'gray', 'Loss': 'red'}
        for result, color in result_colors.items():
            mask = df['result'] == result
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.5,
                       s=20, label=f'{result} ({mask.sum()})')

        ax.set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA 2D Projection - Colored by Result', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Color by time class
        ax = axes[1]
        time_colors = {'bullet': 'red', 'blitz': 'blue', 'rapid': 'green', 'daily': 'purple'}
        for time_class, color in time_colors.items():
            mask = df['time_class'] == time_class
            if mask.sum() > 0:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.5,
                           s=20, label=f'{time_class} ({mask.sum()})')

        ax.set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA 2D Projection - Colored by Time Class', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'pca_2d_projection.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_3d_projection(self, results, df):
        """Plot 3D PCA projection"""
        X_pca = results['X_pca']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        result_colors = {'Win': 'green', 'Draw': 'gray', 'Loss': 'red'}
        for result, color in result_colors.items():
            mask = df['result'] == result
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                       c=color, alpha=0.5, s=20, label=result)

        ax.set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]*100:.1f}%)', fontsize=10)
        ax.set_zlabel(f'PC3 ({results["explained_variance_ratio"][2]*100:.1f}%)', fontsize=10)
        ax.set_title('PCA 3D Projection', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'pca_3d_projection.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_biplot(self, results):
        """Plot biplot showing both data points and feature loadings"""
        X_pca = results['X_pca']
        loadings = self.pca.components_.T

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot data points
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=10, c='gray')

        # Plot feature vectors
        scale = max(abs(X_pca[:, 0]).max(), abs(X_pca[:, 1]).max()) * 0.8
        for i, (feature, loading) in enumerate(zip(self.feature_names, loadings)):
            ax.arrow(0, 0, loading[0] * scale, loading[1] * scale,
                     head_width=0.05 * scale, head_length=0.03 * scale,
                     fc='red', ec='red', alpha=0.8)
            ax.text(loading[0] * scale * 1.15, loading[1] * scale * 1.15,
                    feature, fontsize=10, ha='center', va='center',
                    color='darkred', fontweight='bold')

        ax.set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA Biplot - Features and Data Points', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'pca_biplot.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_loadings_heatmap(self):
        """Plot heatmap of component loadings"""
        loadings = self.get_loadings()

        # Only show first 4 components
        loadings_subset = loadings.iloc[:, :min(4, loadings.shape[1])]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(loadings_subset, annot=True, cmap='RdBu_r', center=0,
                    fmt='.3f', ax=ax, linewidths=0.5)
        ax.set_title('PCA Component Loadings', fontsize=14, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12)
        ax.set_xlabel('Principal Components', fontsize=12)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'pca_loadings_heatmap.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def run_analysis(self):
        """Run complete PCA analysis"""
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
        print("="*60)

        # Prepare data
        X, df = self.prepare_features()
        print(f"\nDataset: {len(X)} games, {len(self.feature_names)} features")
        print(f"Features: {', '.join(self.feature_names)}")

        # Perform PCA
        print("\nPerforming PCA...")
        results = self.perform_pca(X)

        print(f"\nPCA Results:")
        print(f"  Total components: {len(results['explained_variance_ratio'])}")
        print(f"  Components for 95% variance: {results['n_components_95']}")
        print(f"  Total variance explained: {results['total_variance_explained']*100:.1f}%")

        print("\nVariance explained by each component:")
        for i, var in enumerate(results['explained_variance_ratio']):
            print(f"  PC{i+1}: {var*100:.2f}% (cumulative: {results['cumulative_variance'][i]*100:.2f}%)")

        # Get loadings
        loadings = self.get_loadings()
        results['loadings'] = loadings

        print("\nTop features for first 2 components:")
        for i in range(min(2, loadings.shape[1])):
            pc_col = f'PC{i+1}'
            sorted_loadings = loadings[pc_col].abs().sort_values(ascending=False)
            print(f"  {pc_col}: {', '.join([f'{f} ({loadings.loc[f, pc_col]:.3f})' for f in sorted_loadings.head(3).index])}")

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_explained_variance(results)
        self.plot_2d_projection(results, df)
        self.plot_3d_projection(results, df)
        self.plot_biplot(results)
        self.plot_loadings_heatmap()

        return results


def run_pca(df: pd.DataFrame = None) -> dict:
    """Main function to run PCA analysis"""
    if df is None:
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        df = pd.read_csv(csv_path)

    pca_analyzer = ChessPCA(df)
    results = pca_analyzer.run_analysis()
    return results


if __name__ == "__main__":
    results = run_pca()

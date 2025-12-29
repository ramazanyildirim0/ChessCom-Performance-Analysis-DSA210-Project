"""
K-Nearest Neighbors Classifier for Chess Game Outcome Prediction
Finds similar games to predict outcomes using distance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, VISUALIZATIONS_DIR, USERNAME


class ChessKNN:
    """K-Nearest Neighbors classifier for chess game outcome prediction"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.viz_dir = os.path.join(VISUALIZATIONS_DIR, 'ml')
        os.makedirs(self.viz_dir, exist_ok=True)

    def prepare_features(self):
        """Prepare and scale features for KNN"""
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

        # Select numeric features
        feature_cols = ['rating_diff', 'color_encoded', 'hour', 'day_encoded',
                        'move_count', 'time_encoded', 'player_rating', 'opponent_rating']

        self.feature_names = feature_cols
        X = df[feature_cols].fillna(0).astype(float)
        y = df['result']

        return X, y

    def find_optimal_k(self, X_train, X_test, y_train, y_test, k_range=range(1, 31)):
        """Find optimal K using accuracy on test set"""
        accuracies = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        optimal_k = k_range[np.argmax(accuracies)]
        return optimal_k, accuracies, list(k_range)

    def compare_distance_metrics(self, X_train, X_test, y_train, y_test, k=5):
        """Compare different distance metrics"""
        metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        results = {}

        for metric in metrics:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            results[metric] = accuracy_score(y_test, y_pred)

        return results

    def train_model(self, X, y, k=5, test_size=0.2, random_state=42):
        """Train the KNN model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Find optimal K
        optimal_k, k_accuracies, k_values = self.find_optimal_k(
            X_train, X_test, y_train, y_test
        )
        print(f"  Optimal K found: {optimal_k}")

        # Compare distance metrics
        metric_results = self.compare_distance_metrics(X_train, X_test, y_train, y_test, k=optimal_k)

        # Train final model with optimal K
        self.model = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)

        results = {
            'optimal_k': optimal_k,
            'k_values': k_values,
            'k_accuracies': k_accuracies,
            'metric_comparison': metric_results,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

        return results

    def cross_validate(self, X, y, k=5, cv=5):
        """Perform cross-validation"""
        X_scaled = self.scaler.fit_transform(X)
        model = KNeighborsClassifier(n_neighbors=k)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')

        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }

    def find_similar_games(self, game_features, n_neighbors=5):
        """Find most similar games to a given game"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        game_scaled = self.scaler.transform([game_features])
        distances, indices = self.model.kneighbors(game_scaled, n_neighbors=n_neighbors)

        return distances[0], indices[0]

    def plot_k_selection(self, results):
        """Plot K selection (elbow method)"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(results['k_values'], results['k_accuracies'], 'b-o', linewidth=2, markersize=6)
        ax.axvline(x=results['optimal_k'], color='r', linestyle='--',
                   label=f'Optimal K = {results["optimal_k"]}')

        ax.set_xlabel('Number of Neighbors (K)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('KNN - Selecting Optimal K', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'knn_k_selection.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_distance_metric_comparison(self, results):
        """Plot distance metric comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = list(results['metric_comparison'].keys())
        accuracies = list(results['metric_comparison'].values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))

        bars = ax.bar(metrics, accuracies, color=colors)

        ax.set_xlabel('Distance Metric', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('KNN - Distance Metric Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'knn_distance_metrics.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_confusion_matrix(self, results):
        """Plot confusion matrix"""
        cm = results['confusion_matrix']
        labels = self.model.classes_

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'KNN (K={results["optimal_k"]}) - Confusion Matrix',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'knn_confusion_matrix.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def run_analysis(self):
        """Run complete KNN analysis"""
        print("\n" + "="*60)
        print("K-NEAREST NEIGHBORS CLASSIFIER")
        print("="*60)

        # Prepare data
        X, y = self.prepare_features()
        print(f"\nDataset: {len(X)} games, {len(self.feature_names)} features")
        print(f"Features: {', '.join(self.feature_names)}")

        # Train model
        print("\nTraining KNN and finding optimal K...")
        results = self.train_model(X, y)

        print(f"\nOptimal K: {results['optimal_k']}")
        print(f"Accuracy: {results['accuracy']:.4f}")

        print("\nDistance Metric Comparison:")
        for metric, acc in results['metric_comparison'].items():
            print(f"  {metric.capitalize()}: {acc:.4f}")

        print("\nClassification Report:")
        print(results['classification_report'])

        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_results = self.cross_validate(X, y, k=results['optimal_k'])
        print(f"CV Mean: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})")

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_k_selection(results)
        self.plot_distance_metric_comparison(results)
        self.plot_confusion_matrix(results)

        results['cv_results'] = cv_results
        return results


def run_knn(df: pd.DataFrame = None) -> dict:
    """Main function to run KNN analysis"""
    if df is None:
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        df = pd.read_csv(csv_path)

    knn = ChessKNN(df)
    results = knn.run_analysis()
    return results


if __name__ == "__main__":
    results = run_knn()

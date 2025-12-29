"""
Decision Tree Classifier for Chess Game Outcome Prediction
Provides interpretable decision rules and tree visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, VISUALIZATIONS_DIR, USERNAME


class ChessDecisionTree:
    """Decision Tree classifier for chess game outcome prediction"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.pruned_model = None
        self.feature_names = None
        self.viz_dir = os.path.join(VISUALIZATIONS_DIR, 'ml')
        os.makedirs(self.viz_dir, exist_ok=True)

    def prepare_features(self):
        """Prepare features for the model"""
        df = self.df.copy()

        # Encode player color (White=1, Black=0)
        df['color_encoded'] = (df['player_color'] == 'White').astype(int)

        # Encode day of week
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df['day_encoded'] = df['day_of_week'].map(day_map).fillna(0)

        # Encode time_class as numeric
        time_class_map = {'bullet': 0, 'blitz': 1, 'rapid': 2, 'daily': 3}
        df['time_class_encoded'] = df['time_class'].map(time_class_map).fillna(1)

        # Select only numeric features
        feature_cols = ['rating_diff', 'color_encoded', 'hour', 'day_encoded',
                        'move_count', 'time_class_encoded', 'player_rating', 'opponent_rating']

        self.feature_names = feature_cols
        X = df[feature_cols].fillna(0).astype(float)
        y = df['result']

        return X, y

    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train Decision Tree models (full and pruned)"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Full tree (for comparison)
        self.full_model = DecisionTreeClassifier(random_state=random_state)
        self.full_model.fit(X_train, y_train)

        # Pruned tree (using max_depth and min_samples for interpretability)
        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=random_state
        )
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred_full = self.full_model.predict(X_test)
        y_pred_pruned = self.model.predict(X_test)

        results = {
            'full_tree': {
                'accuracy': accuracy_score(y_test, y_pred_full),
                'depth': self.full_model.get_depth(),
                'n_leaves': self.full_model.get_n_leaves()
            },
            'pruned_tree': {
                'accuracy': accuracy_score(y_test, y_pred_pruned),
                'depth': self.model.get_depth(),
                'n_leaves': self.model.get_n_leaves(),
                'classification_report': classification_report(y_test, y_pred_pruned),
                'confusion_matrix': confusion_matrix(y_test, y_pred_pruned)
            },
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred_pruned
        }

        return results

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')

        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }

    def get_decision_rules(self):
        """Extract decision rules from the tree"""
        rules = export_text(self.model, feature_names=self.feature_names, max_depth=5)
        return rules

    def plot_tree_visualization(self):
        """Plot the decision tree"""
        fig, ax = plt.subplots(figsize=(24, 12))
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.model.classes_,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax
        )
        ax.set_title('Decision Tree for Chess Game Outcome Prediction',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'dt_tree_visualization.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_feature_importance(self):
        """Plot feature importance from decision tree"""
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # Filter out zero importance features
        sorted_importance = {k: v for k, v in sorted_importance.items() if v > 0}

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(sorted_importance)))

        bars = ax.barh(list(sorted_importance.keys()), list(sorted_importance.values()), color=colors)
        ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
        ax.set_title('Decision Tree - Feature Importance',
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        for bar, val in zip(bars, sorted_importance.values()):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'dt_feature_importance.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_confusion_matrix(self, results):
        """Plot confusion matrix"""
        cm = results['pruned_tree']['confusion_matrix']
        labels = self.model.classes_

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'dt_confusion_matrix.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_pruning_comparison(self, results):
        """Compare full vs pruned tree"""
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ['Accuracy', 'Depth', 'Leaves']
        full_vals = [results['full_tree']['accuracy'],
                     results['full_tree']['depth'] / 50,  # Normalize
                     results['full_tree']['n_leaves'] / 1000]  # Normalize
        pruned_vals = [results['pruned_tree']['accuracy'],
                       results['pruned_tree']['depth'] / 50,
                       results['pruned_tree']['n_leaves'] / 1000]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width/2, full_vals, width, label='Full Tree', color='coral')
        bars2 = ax.bar(x + width/2, pruned_vals, width, label='Pruned Tree', color='seagreen')

        ax.set_ylabel('Value (normalized for depth/leaves)')
        ax.set_title('Full Tree vs Pruned Tree Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Add actual values as labels
        ax.annotate(f"{results['full_tree']['accuracy']:.3f}", xy=(0 - width/2, full_vals[0]),
                    ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{results['pruned_tree']['accuracy']:.3f}", xy=(0 + width/2, pruned_vals[0]),
                    ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{results['full_tree']['depth']}", xy=(1 - width/2, full_vals[1]),
                    ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{results['pruned_tree']['depth']}", xy=(1 + width/2, pruned_vals[1]),
                    ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{results['full_tree']['n_leaves']}", xy=(2 - width/2, full_vals[2]),
                    ha='center', va='bottom', fontsize=9)
        ax.annotate(f"{results['pruned_tree']['n_leaves']}", xy=(2 + width/2, pruned_vals[2]),
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'dt_pruning_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def run_analysis(self):
        """Run complete Decision Tree analysis"""
        print("\n" + "="*60)
        print("DECISION TREE CLASSIFIER")
        print("="*60)

        # Prepare data
        X, y = self.prepare_features()
        print(f"\nDataset: {len(X)} games, {len(self.feature_names)} features")

        # Train model
        print("\nTraining Decision Trees (Full and Pruned)...")
        results = self.train_model(X, y)

        print(f"\nFull Tree: Accuracy={results['full_tree']['accuracy']:.4f}, "
              f"Depth={results['full_tree']['depth']}, Leaves={results['full_tree']['n_leaves']}")
        print(f"Pruned Tree: Accuracy={results['pruned_tree']['accuracy']:.4f}, "
              f"Depth={results['pruned_tree']['depth']}, Leaves={results['pruned_tree']['n_leaves']}")

        print("\nPruned Tree Classification Report:")
        print(results['pruned_tree']['classification_report'])

        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_results = self.cross_validate(X, y)
        print(f"CV Mean: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})")

        # Decision rules
        print("\nDecision Rules (Top Levels):")
        rules = self.get_decision_rules()
        print(rules[:2000])  # Print first 2000 characters

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_tree_visualization()
        self.plot_feature_importance()
        self.plot_confusion_matrix(results)
        self.plot_pruning_comparison(results)

        results['cv_results'] = cv_results
        results['decision_rules'] = rules
        return results


def run_decision_tree(df: pd.DataFrame = None) -> dict:
    """Main function to run Decision Tree analysis"""
    if df is None:
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        df = pd.read_csv(csv_path)

    dt = ChessDecisionTree(df)
    results = dt.run_analysis()
    return results


if __name__ == "__main__":
    results = run_decision_tree()

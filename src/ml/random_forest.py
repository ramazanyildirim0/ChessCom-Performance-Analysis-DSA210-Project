"""
Random Forest Classifier for Chess Game Outcome Prediction
Predicts Win/Loss/Draw based on game features with feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, VISUALIZATIONS_DIR, USERNAME


class ChessRandomForest:
    """Random Forest classifier for chess game outcome prediction"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
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
        """Train the Random Forest model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)

        # Results
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

        return results

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')

        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }

    def plot_feature_importance(self, results):
        """Plot feature importance"""
        importance = results['feature_importance']
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_importance)))

        bars = ax.barh(list(sorted_importance.keys()), list(sorted_importance.values()), color=colors)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Random Forest - Feature Importance for Game Outcome Prediction',
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, sorted_importance.values()):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'rf_feature_importance.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_confusion_matrix(self, results):
        """Plot confusion matrix"""
        cm = results['confusion_matrix']
        labels = self.model.classes_

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'rf_confusion_matrix.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def run_analysis(self):
        """Run complete Random Forest analysis"""
        print("\n" + "="*60)
        print("RANDOM FOREST CLASSIFIER")
        print("="*60)

        # Prepare data
        X, y = self.prepare_features()
        print(f"\nDataset: {len(X)} games, {len(self.feature_names)} features")
        print(f"Target distribution:\n{y.value_counts().to_string()}")

        # Train model
        print("\nTraining Random Forest...")
        results = self.train_model(X, y)

        print(f"\nAccuracy: {results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])

        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_results = self.cross_validate(X, y)
        print(f"CV Scores: {cv_results['cv_scores']}")
        print(f"CV Mean: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})")

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_feature_importance(results)
        self.plot_confusion_matrix(results)

        # Feature importance summary
        print("\nTop 5 Most Important Features:")
        sorted_imp = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_imp[:5], 1):
            print(f"  {i}. {feat}: {imp:.4f}")

        results['cv_results'] = cv_results
        return results


def run_random_forest(df: pd.DataFrame = None) -> dict:
    """Main function to run Random Forest analysis"""
    if df is None:
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        df = pd.read_csv(csv_path)

    rf = ChessRandomForest(df)
    results = rf.run_analysis()
    return results


if __name__ == "__main__":
    results = run_random_forest()

"""
Machine Learning Module for Chess.com Performance Analysis
Contains 7 ML methods: Random Forest, Decision Tree, KNN, K-Means, DBSCAN, PCA, Time Series
"""

from .random_forest import run_random_forest
from .decision_tree import run_decision_tree
from .knn import run_knn
from .kmeans_clustering import run_kmeans
from .dbscan_clustering import run_dbscan
from .pca_analysis import run_pca
from .time_series_forecast import run_time_series_forecast


def run_all_ml_models(df):
    """Run all machine learning models on the chess game data"""
    print("\n" + "="*80)
    print("MACHINE LEARNING ANALYSIS")
    print("="*80)

    results = {}

    # Supervised Learning
    print("\n" + "-"*40)
    print("SUPERVISED LEARNING")
    print("-"*40)

    results['random_forest'] = run_random_forest(df)
    results['decision_tree'] = run_decision_tree(df)
    results['knn'] = run_knn(df)

    # Unsupervised Learning
    print("\n" + "-"*40)
    print("UNSUPERVISED LEARNING")
    print("-"*40)

    results['kmeans'] = run_kmeans(df)
    results['dbscan'] = run_dbscan(df)
    results['pca'] = run_pca(df)

    # Time Series
    print("\n" + "-"*40)
    print("TIME SERIES ANALYSIS")
    print("-"*40)

    results['time_series'] = run_time_series_forecast(df)

    print("\n" + "="*80)
    print("ML ANALYSIS COMPLETE")
    print("="*80)

    return results


__all__ = [
    'run_random_forest',
    'run_decision_tree',
    'run_knn',
    'run_kmeans',
    'run_dbscan',
    'run_pca',
    'run_time_series_forecast',
    'run_all_ml_models'
]

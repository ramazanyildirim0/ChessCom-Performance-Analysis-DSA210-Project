# Chess.com Performance Analysis - DSA 210 Project

**Username:** ramazanyildirimm
**Analysis Date:** December 29, 2025
**Total Games Analyzed:** 4,264 games
**Date Range:** November 6, 2019 - November 6, 2025 (6 years)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Executive Summary](#executive-summary)
3. [How to Run](#how-to-run)
4. [Project Structure](#project-structure)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Overall Statistics](#overall-statistics)
   - [Rating Progression](#rating-progression)
   - [Results Distribution](#results-distribution)
   - [Time Pattern Analysis](#time-pattern-analysis)
   - [Opening Analysis](#opening-analysis)
   - [Rating Difference Analysis](#rating-difference-analysis)
   - [Game Termination Analysis](#game-termination-analysis)
   - [Game Length Analysis](#game-length-analysis)
6. [Hypothesis Testing Results](#hypothesis-testing-results)
7. [Machine Learning Analysis](#machine-learning-analysis)
   - [Supervised Learning](#supervised-learning)
   - [Unsupervised Learning](#unsupervised-learning)
   - [Time Series Forecasting](#time-series-forecasting)
8. [Key Insights & Recommendations](#key-insights--recommendations)
9. [Technologies Used](#technologies-used)

---

## Project Overview

### Motivation
As an active chess player with over 4,000 games played on Chess.com over the past 6 years, I analyzed my personal chess data to understand patterns in my performance and identify factors that influence my game outcomes. This project explores my playing style, rating progression, and decision-making patterns to gain insights that could help improve my chess skills.

### Dataset
Personal chess game history from Chess.com, which includes:
- Over 4,000 games spanning 6 years
- Game metadata (date, time control, results, ratings)
- Move-by-move data in PGN format
- Opponent information
- Opening variations played
- Rating changes over time

### Data Collection
Data collected from **Chess.com's Public API**:
- `https://api.chess.com/pub/player/{username}` - Player profile
- `https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}` - Monthly archives
- `https://api.chess.com/pub/player/{username}/stats` - Player statistics

---

## Executive Summary

This analysis examines 4,264 chess games played on Chess.com over a 6-year period. The data reveals significant improvement in playing strength, with an average rating increase of **+525 points**.

### Key Metrics at a Glance

| Metric | Value |
|--------|-------|
| Total Games | 4,264 |
| Win Rate | 49.6% |
| Draw Rate | 4.9% |
| Loss Rate | 45.5% |
| Peak Rating | 1,453 |
| Rating Improvement | +525 points |
| Most Played Format | Blitz (51.5%) |

### Key Findings Summary

| Category | Finding |
|----------|---------|
| Best ML Classifier | Random Forest (67.8% accuracy) |
| Most Important Feature | Rating Difference (43.9% importance) |
| Optimal Clusters | K-Means (K=3) |
| Rating Forecast (6 months) | +59 points (Prophet) |

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Username

Edit `src/config.py` and update the `USERNAME` variable:

```python
USERNAME = "your_chess_com_username"
```

### 3. Run the Analysis

```bash
python main.py
```

This opens an interactive menu:

```
============================================================
MAIN MENU
============================================================

  [1] Collect Data from Chess.com API
  [2] Run Exploratory Data Analysis (EDA)
  [3] Run Hypothesis Testing
  [4] Run Machine Learning Analysis
  [5] Run All (Complete Pipeline)
  [0] Exit
```

---

## Project Structure

```
ChessCom-Performance-Analysis-DSA210-Project/
├── main.py                 # Main script with interactive menu
├── requirements.txt        # Python dependencies
├── readme.md              # Project documentation (this file)
├── eda_results.md         # Detailed EDA & hypothesis testing results
├── ml_results.md          # Detailed machine learning results
├── src/
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings
│   ├── data_collection.py # Chess.com API data collection
│   ├── eda.py             # Exploratory Data Analysis
│   ├── hypothesis_tests.py # Statistical hypothesis testing
│   └── ml/                # Machine Learning modules
│       ├── __init__.py
│       ├── random_forest.py    # Random Forest classifier
│       ├── decision_tree.py    # Decision Tree classifier
│       ├── knn.py              # K-Nearest Neighbors
│       ├── kmeans_clustering.py # K-Means clustering
│       ├── dbscan_clustering.py # DBSCAN clustering
│       ├── pca_analysis.py     # PCA dimensionality reduction
│       └── time_series_forecast.py # Prophet/ARIMA forecasting
├── data/                  # Collected data files
│   ├── {username}_games.csv
│   ├── {username}_games_full.json
│   └── hypothesis_test_results.csv
└── visualizations/        # Generated plots
    ├── *.png              # EDA visualizations (9 plots)
    └── ml/                # ML visualizations (24 plots)
```

---

## Exploratory Data Analysis (EDA)

### Overall Statistics

#### Game Distribution by Time Control

| Time Control | Games | Percentage |
|--------------|-------|------------|
| Blitz | 2,195 | 51.5% |
| Bullet | 2,049 | 48.1% |
| Rapid | 19 | 0.4% |
| Daily | 1 | 0.0% |

#### Results Breakdown

| Result | Count | Percentage |
|--------|-------|------------|
| Wins | 2,116 | 49.6% |
| Losses | 1,940 | 45.5% |
| Draws | 208 | 4.9% |

#### Performance by Color

| Color | Games | Win Rate |
|-------|-------|----------|
| White | 2,116 | 51.6% |
| Black | 2,148 | 47.7% |

**Finding:** Playing as White provides a statistically significant advantage of **+3.9%** in win rate.

#### Rating Statistics

| Metric | Value |
|--------|-------|
| Average Rating | 1,120 |
| Highest Rating | 1,453 |
| Lowest Rating | 522 |
| Average Opponent Rating | 1,110 |
| Average Moves per Game | 72.6 |

---

### Rating Progression

#### Overall Rating Over Time

![Rating Over Time](visualizations/rating_over_time.png)

The rating progression shows a clear upward trend over the 6-year period, with the player improving from an initial average of **796** to a recent average of **1,321** - an improvement of **+525 rating points**.

#### Rating by Time Class

![Rating by Time Class](visualizations/rating_by_time_class.png)

Current ratings by time control:
- **Bullet:** 1,343
- **Blitz:** 1,201
- **Rapid:** 832

---

### Results Distribution

![Results Distribution](visualizations/results_distribution.png)

**Analysis:**
1. **Overall Results:** Near 50% win rate indicates well-matched opponents
2. **Color Impact:** White pieces provide a measurable advantage
3. **Time Control:** Performance is consistent across blitz and bullet formats

---

### Time Pattern Analysis

![Games by Time](visualizations/games_by_time.png)

#### Activity Heatmap

![Heatmap Day Hour](visualizations/heatmap_day_hour.png)

#### Games by Time of Day

| Time Period | Games | Win Rate |
|-------------|-------|----------|
| Night (0-6) | 105 | 45.7% |
| Morning (6-12) | 531 | 50.8% |
| Afternoon (12-18) | 2,013 | 49.7% |
| Evening (18-24) | 1,615 | 49.3% |

**Best Performance Day:** Thursday (52.1% win rate)
**Worst Performance Day:** Saturday (46.9% win rate)

---

### Opening Analysis

![Opening Analysis](visualizations/opening_analysis.png)

#### Most Played Openings

| Opening | Games |
|---------|-------|
| Van't Kruijs Opening | 139 |
| Scotch Game | 122 |
| King's Pawn Opening | 112 |
| Scandinavian Defense (Mieses-Kotrč) | 100 |
| King's Pawn - King's Knight Variation | 99 |

**Total Unique Openings:** 393

#### Best & Worst Performing Openings (min 20 games)
- **Best:** Nimzowitsch Defense Kennedy Variation (65.0% win rate)
- **Worst:** Italian Game Knight Attack Normal (26.7% win rate)

---

### Rating Difference Analysis

![Rating Difference Analysis](visualizations/rating_difference_analysis.png)

| Rating Difference | Win Rate | Games |
|-------------------|----------|-------|
| You're much higher rated (>50) | 82.4% | 612 |
| You're lower rated (<-50) | 14.0% | 492 |

**Key Finding:** Rating difference is the strongest predictor of game outcome (p < 0.001).

---

### Game Termination Analysis

![Termination Analysis](visualizations/termination_analysis.png)

| Termination Type | Games | Percentage |
|------------------|-------|------------|
| Checkmate | 1,852 | 43.4% |
| Timeout | 1,344 | 31.5% |
| Resignation | 781 | 18.3% |
| Insufficient Material | 95 | 2.2% |
| Abandoned | 79 | 1.9% |
| Stalemate | 71 | 1.7% |
| Repetition | 40 | 0.9% |

---

### Game Length Analysis

![Game Length Analysis](visualizations/game_length_analysis.png)

- **Average Game Length:** 72.6 moves
- **Correlation with Winning:** r = -0.165 (negative correlation)

**Interpretation:** Shorter games are slightly more likely to result in wins.

---

## Hypothesis Testing Results

Ten statistical hypothesis tests were conducted.

**Terminology Note:** "Win rate" and "win probability" are used interchangeably throughout this analysis. Both refer to the proportion of games won (wins / total games).

**Multiple Testing Correction:** Tests 4 (Time of Day) and 5 (Day of Week) both examine temporal factors affecting win rate. Since these are related hypotheses tested simultaneously, **Bonferroni correction** is applied to control for family-wise error rate:
- Standard significance level: α = 0.05
- Corrected significance level for Tests 4 & 5: α = 0.05 / 2 = **0.025**

### Summary

| Category | Count |
|----------|-------|
| Total Tests | 10 |
| Significant Results | 4 |
| Non-Significant Results | 6 |

### Significant Findings

| Test | Hypothesis | p-value | α | Result |
|------|------------|---------|---|--------|
| **White vs Black Win Rate** | H₀: Win rate as White = Win rate as Black | 0.0121 | 0.05 | **REJECT H₀** |
| **Rating Difference Effect** | H₀: Rating difference doesn't affect win rate | < 0.0001 | 0.05 | **REJECT H₀** |
| **Game Length Correlation** | H₀: No correlation between game length and winning | < 0.0001 | 0.05 | **REJECT H₀** |
| **Rating Progression** | H₀: No significant change in rating over time | < 0.0001 | 0.05 | **REJECT H₀** |

### Non-Significant Findings

| Test | Hypothesis | p-value | α | Result |
|------|------------|---------|---|--------|
| **Time Control Effect** | H₀: Win rate is same across time controls | 0.2042 | 0.05 | Fail to reject H₀ |
| **Time of Day Effect** | H₀: Win rate is same regardless of time of day | 0.3630 | 0.025* | Fail to reject H₀ |
| **Day of Week Effect** | H₀: Win rate is same across all days | 0.3505 | 0.025* | Fail to reject H₀ |
| **Opening Effect** | H₀: Win rate is same across openings | 0.2904 | 0.05 | Fail to reject H₀ |
| **Win Rate vs 50%** | H₀: Overall win rate = 50% | 0.6241 | 0.05 | Fail to reject H₀ |
| **Momentum Effect** | H₀: Previous result doesn't affect next game | 0.5127 | 0.05 | Fail to reject H₀ |

*Bonferroni-corrected significance level

### Detailed Test Results

#### Test 1: White vs Black Advantage
- **Test:** Two-Proportion Z-Test
- **Statistic:** z = 2.508, **p-value:** 0.0121
- **Details:** White Win Rate: 51.56%, Black Win Rate: 47.72%
- **Conclusion:** Statistically significant advantage when playing as White.

#### Test 2: Rating Difference Effect
- **Test:** Two-Proportion Z-Test
- **Statistic:** z = 22.585, **p-value:** < 0.0001
- **Details:** Win rate when higher rated (>50): 82.35%, when lower rated (<-50): 14.02%
- **Conclusion:** Rating difference strongly predicts game outcome.

#### Tests 4 & 5: Temporal Effects (Bonferroni-corrected)
- **Test 4 (Time of Day):** Chi-Square, p = 0.3630 > 0.025 → Not significant
- **Test 5 (Day of Week):** Chi-Square, p = 0.3505 > 0.025 → Not significant
- **Correction Applied:** Since both tests examine temporal factors affecting the same outcome (win rate), Bonferroni correction adjusts α from 0.05 to 0.025.
- **Conclusion:** Performance is consistent regardless of when games are played.

#### Test 7: Rating Progression
- **Test:** Mann-Whitney U Test
- **Statistic:** U = 15.0, **p-value:** < 0.0001
- **Details:** Early avg: 796, Recent avg: 1,321, Change: +525
- **Conclusion:** Significant improvement in rating over time.

---

## Machine Learning Analysis

Seven machine learning methods were applied to analyze chess game data.

### Supervised Learning

#### 1. Random Forest Classifier

**Purpose:** Predict game outcome (Win/Loss/Draw) using ensemble of decision trees.

| Metric | Value |
|--------|-------|
| **Accuracy** | 67.76% |
| **CV Mean** | 67.66% (+/- 3.19%) |
| **Best Class** | Win (F1: 0.72) |

##### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Draw | 0.00 | 0.00 | 0.00 | 42 |
| Loss | 0.68 | 0.65 | 0.66 | 388 |
| Win | 0.68 | 0.77 | 0.72 | 423 |

##### Feature Importance

![Random Forest Feature Importance](visualizations/ml/rf_feature_importance.png)

**Top 5 Most Important Features:**
1. **rating_diff**: 43.88% - The difference between player and opponent rating
2. **move_count**: 15.73% - Number of moves in the game
3. **opponent_rating**: 13.88% - Opponent's rating
4. **player_rating**: 12.30% - Player's rating
5. **hour**: 6.48% - Hour of day when game was played

![Random Forest Confusion Matrix](visualizations/ml/rf_confusion_matrix.png)

---

#### 2. Decision Tree Classifier

**Purpose:** Create interpretable decision rules for predicting game outcomes.

| Model | Accuracy | Depth | Leaves |
|-------|----------|-------|--------|
| Full Tree | 58.15% | 29 | 909 |
| **Pruned Tree** | **65.42%** | 5 | 23 |

##### Decision Tree Visualization

![Decision Tree Visualization](visualizations/ml/dt_tree_visualization.png)

##### Key Decision Rules

```
If rating_diff <= -24.5:
    → Likely LOSS (playing against higher-rated opponent)

If rating_diff > 32.5:
    → Likely WIN (playing against lower-rated opponent)

If -24.5 < rating_diff <= 32.5:
    → Check move_count:
        If move_count > 95.5 and move_count > 149.5:
            → Likely DRAW
        Else:
            → Depends on other factors
```

![Decision Tree Feature Importance](visualizations/ml/dt_feature_importance.png)

![Pruning Comparison](visualizations/ml/dt_pruning_comparison.png)

---

#### 3. K-Nearest Neighbors (KNN)

**Purpose:** Classify games based on similarity to nearest neighbors.

| Metric | Value |
|--------|-------|
| **Optimal K** | 30 |
| **Accuracy** | 63.77% |
| **CV Mean** | 62.71% (+/- 2.08%) |

##### Distance Metric Comparison

| Metric | Accuracy |
|--------|----------|
| Manhattan | 64.13% |
| Euclidean | 63.77% |
| Minkowski | 63.77% |
| Chebyshev | 61.90% |

![KNN K Selection](visualizations/ml/knn_k_selection.png)

![KNN Distance Metrics](visualizations/ml/knn_distance_metrics.png)

![KNN Confusion Matrix](visualizations/ml/knn_confusion_matrix.png)

---

### Unsupervised Learning

#### 4. K-Means Clustering

**Purpose:** Discover natural groupings of games based on their characteristics.

| Metric | Value |
|--------|-------|
| **Optimal K** | 3 |
| **Silhouette Score** | 0.3354 |
| **Inertia** | 15,288.92 |

##### Cluster Summary

| Cluster | Size | % | Avg Rating Diff | Win Rate | Dominant Format |
|---------|------|---|-----------------|----------|-----------------|
| **0** | 2,231 | 52.3% | -1.4 | 48.5% | Bullet |
| **1** | 1,914 | 44.9% | -2.3 | 48.7% | Blitz |
| **2** | 119 | 2.8% | +423.8 | 84.9% | Blitz |

##### Cluster Interpretation
- **Cluster 0 (Bullet Games):** Majority of games, evenly matched opponents
- **Cluster 1 (Blitz Games):** Similar to Cluster 0 but in blitz format
- **Cluster 2 (Easy Wins):** Games against much lower-rated opponents with 85% win rate

![K-Means Elbow and Silhouette](visualizations/ml/kmeans_elbow_silhouette.png)

![K-Means Clusters](visualizations/ml/kmeans_clusters.png)

![K-Means Cluster Analysis](visualizations/ml/kmeans_cluster_analysis.png)

---

#### 5. DBSCAN Clustering

**Purpose:** Find density-based clusters and identify outlier games.

| Parameter | Value |
|-----------|-------|
| **Epsilon (ε)** | 15.06 |
| **Min Samples** | 10 |
| **Clusters Found** | 1 |
| **Noise Points** | 0 (0.0%) |

![DBSCAN Epsilon Selection](visualizations/ml/dbscan_eps_selection.png)

![DBSCAN Clusters](visualizations/ml/dbscan_clusters.png)

**Interpretation:** DBSCAN finds all games belong to a single dense cluster with no outliers, suggesting the game data is uniformly distributed without clear density-based separation.

---

#### 6. Principal Component Analysis (PCA)

**Purpose:** Reduce dimensionality and visualize game patterns.

| Metric | Value |
|--------|-------|
| **Components for 95% Variance** | 7 |
| **PC1 Variance** | 28.09% |
| **PC2 Variance** | 14.09% |

##### Variance Explained by Component

| Component | Individual | Cumulative |
|-----------|------------|------------|
| PC1 | 28.09% | 28.09% |
| PC2 | 14.09% | 42.18% |
| PC3 | 13.19% | 55.37% |
| PC4 | 12.38% | 67.75% |
| PC5 | 12.27% | 80.02% |
| PC6 | 11.50% | 91.52% |
| PC7 | 8.48% | 100.00% |

##### Top Features per Component

**PC1 (Rating Level):**
- opponent_rating: 0.640
- player_rating: 0.620
- time_encoded: -0.422

**PC2 (Game Competitiveness):**
- rating_diff: 0.666
- move_count: -0.552
- time_encoded: -0.273

![PCA Explained Variance](visualizations/ml/pca_explained_variance.png)

![PCA 2D Projection](visualizations/ml/pca_2d_projection.png)

![PCA 3D Projection](visualizations/ml/pca_3d_projection.png)

![PCA Biplot](visualizations/ml/pca_biplot.png)

![PCA Loadings Heatmap](visualizations/ml/pca_loadings_heatmap.png)

---

### Time Series Forecasting

#### 7. Rating Forecasting (Prophet & ARIMA)

**Purpose:** Forecast future rating using Prophet and ARIMA models.

##### Data Summary

| Metric | Value |
|--------|-------|
| **Date Range** | Nov 2019 - Nov 2025 |
| **Total Weeks** | 123 |
| **Rating Range** | 628 - 1,404 |
| **Current Rating** | 1,193 |

##### Prophet Forecast (6 Months)

| Metric | Value |
|--------|-------|
| **Forecasted Rating** | 1,252 |
| **Expected Change** | **+59 points** |
| **Confidence Interval** | 1,184 - 1,323 |

##### ARIMA Forecast (6 Months)

| Metric | Value |
|--------|-------|
| **Forecasted Rating** | 1,199 |
| **Expected Change** | **+6 points** |
| **Model AIC** | 1,276.28 |

![Time Series Decomposition](visualizations/ml/ts_decomposition.png)

![Prophet Forecast](visualizations/ml/ts_prophet_forecast.png)

![ARIMA Forecast](visualizations/ml/ts_arima_forecast.png)

![Forecast Comparison](visualizations/ml/ts_forecast_comparison.png)

**Interpretation:**
- **Prophet** predicts more optimistic growth (+59 points) by capturing long-term trend
- **ARIMA** is more conservative (+6 points), focusing on recent patterns
- Both models agree rating will continue to improve

---

### Model Comparison

#### Supervised Learning Comparison

| Model | Accuracy | CV Mean | Best For |
|-------|----------|---------|----------|
| **Random Forest** | **67.76%** | **67.66%** | Best overall accuracy |
| Decision Tree | 65.42% | 65.31% | Interpretability |
| KNN | 63.77% | 62.71% | Simple baseline |

#### Unsupervised Learning Comparison

| Model | Clusters | Silhouette | Best For |
|-------|----------|------------|----------|
| **K-Means** | 3 | **0.335** | Clear cluster separation |
| DBSCAN | 1 | N/A | Density-based analysis |

---

## Key Insights & Recommendations

### Main Findings

1. **Rating Difference is King**
   - The most important predictor of game outcome is the rating difference (43.9% importance in Random Forest)
   - Win 82% against lower-rated opponents, only 14% against higher-rated

2. **Three Types of Games**
   - K-Means clustering revealed three distinct game patterns:
     - Regular bullet games (52%)
     - Regular blitz games (45%)
     - Easy wins against lower-rated players (3%)

3. **Significant Improvement Over Time**
   - Rating improved by +525 points over 6 years
   - Prophet forecasts additional +59 points in next 6 months

4. **White Piece Advantage**
   - 3.9% higher win rate as White (statistically significant)

5. **Draws are Hard to Predict**
   - All models struggle to predict draws (only 4.9% of games)
   - Near-zero recall for draw prediction

6. **Consistent Performance**
   - Time of day doesn't affect performance
   - Day of week doesn't affect performance
   - No momentum effect (winning streaks don't predict next game)

### Decision Rules for Winning (from Decision Tree)
- Play against lower-rated opponents (rating_diff > 32) → 75%+ win rate
- Avoid higher-rated opponents (rating_diff < -25) → high loss probability
- Very long games (150+ moves) often end in draws

### Recommendations for Improvement

1. **Opening Preparation** - Analyze openings with lower win rates
2. **Play More Rated Games** - Continue playing to maintain rating progression
3. **Analyze Losses** - Focus on games lost to lower-rated opponents
4. **Time Management** - Work on decisive play given negative correlation between game length and winning

---

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib & seaborn** - Data visualization
- **scipy & statsmodels** - Statistical testing
- **scikit-learn** - Machine learning algorithms
- **prophet** - Time series forecasting
- **python-chess** - PGN parsing
- **requests** - API calls
- **tqdm** - Progress bars

---

## Timeline

| Date | Milestone | Status |
|------|-----------|--------|
| Oct 31 | Project proposal submitted | Complete |
| Nov 28 | Data collection, EDA, hypothesis tests | Complete |
| Dec 29 | Apply ML methods (7 algorithms) | Complete |
| Jan 09 | Final submission | Pending |

---

## Visualizations Summary

### EDA Visualizations (9 plots)
| File | Description |
|------|-------------|
| `rating_over_time.png` | Rating progression over time |
| `rating_by_time_class.png` | Rating by time control |
| `results_distribution.png` | Win/Loss/Draw distribution |
| `games_by_time.png` | Activity patterns |
| `heatmap_day_hour.png` | Activity heatmap |
| `opening_analysis.png` | Opening performance |
| `rating_difference_analysis.png` | Rating diff impact |
| `termination_analysis.png` | How games end |
| `game_length_analysis.png` | Move count analysis |

### ML Visualizations (24 plots in `visualizations/ml/`)
| File | Description |
|------|-------------|
| `rf_feature_importance.png` | Random Forest feature importance |
| `rf_confusion_matrix.png` | Random Forest confusion matrix |
| `dt_tree_visualization.png` | Decision Tree structure |
| `dt_feature_importance.png` | Decision Tree feature importance |
| `dt_confusion_matrix.png` | Decision Tree confusion matrix |
| `dt_pruning_comparison.png` | Full vs Pruned tree comparison |
| `knn_k_selection.png` | Optimal K selection |
| `knn_distance_metrics.png` | Distance metric comparison |
| `knn_confusion_matrix.png` | KNN confusion matrix |
| `kmeans_elbow_silhouette.png` | K-Means optimization |
| `kmeans_clusters.png` | K-Means cluster visualization |
| `kmeans_cluster_analysis.png` | Cluster characteristics |
| `dbscan_eps_selection.png` | DBSCAN epsilon selection |
| `dbscan_clusters.png` | DBSCAN cluster visualization |
| `dbscan_cluster_analysis.png` | DBSCAN analysis |
| `pca_explained_variance.png` | PCA variance explained |
| `pca_2d_projection.png` | 2D PCA projection |
| `pca_3d_projection.png` | 3D PCA projection |
| `pca_biplot.png` | PCA biplot |
| `pca_loadings_heatmap.png` | Component loadings |
| `ts_decomposition.png` | Time series decomposition |
| `ts_prophet_forecast.png` | Prophet forecast |
| `ts_arima_forecast.png` | ARIMA forecast |
| `ts_forecast_comparison.png` | Forecast comparison |

---

## Data Files

| File | Description |
|------|-------------|
| `data/ramazanyildirimm_games.csv` | All game data (4,264 games) |
| `data/ramazanyildirimm_games_full.json` | Full data including PGN moves |
| `data/hypothesis_test_results.csv` | Statistical test results |

---

## Author

Ramazan YILDIRIM 32501
DSA 210 - Introduction to Data Science
2025-2026 Fall Term

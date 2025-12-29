"""
Statistical Hypothesis Testing Module
Conducts various statistical tests on chess game data

Note on Terminology:
- "Win rate" and "win probability" are used interchangeably throughout this module.
  Both refer to the proportion of games won (wins / total games).

Note on Multiple Testing Correction:
- Tests 4 (Time of Day) and 5 (Day of Week) both examine temporal factors affecting
  win rate. Since these are related hypotheses tested simultaneously, we apply
  Bonferroni correction to control for family-wise error rate.
- Corrected significance level for Tests 4 & 5: α = 0.05 / 2 = 0.025
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
import os
import warnings

from config import DATA_DIR, USERNAME

warnings.filterwarnings('ignore')

# Bonferroni correction for temporal tests (H4 and H5)
# Since we test 2 related temporal hypotheses simultaneously, adjust alpha
ALPHA_STANDARD = 0.05
ALPHA_TEMPORAL_CORRECTED = 0.05 / 2  # = 0.025 (Bonferroni correction for 2 tests)


class ChessHypothesisTester:
    """Conducts statistical hypothesis tests on chess game data"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()
        self.results = []
        
    def _prepare_data(self):
        """Prepare data for hypothesis testing"""
        # Ensure date is datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
        
        # Create binary columns
        self.df['is_win'] = (self.df['result'] == 'Win').astype(int)
        self.df['is_white'] = (self.df['player_color'] == 'White').astype(int)
        
        # Create time of day categories
        self.df['time_of_day'] = pd.cut(
            self.df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
            include_lowest=True
        )
    
    def _format_result(self, test_name: str, hypothesis: str, statistic: float,
                       p_value: float, conclusion: str, details: str = "",
                       alpha: float = ALPHA_STANDARD, correction_note: str = ""):
        """Format test result for display"""
        result = {
            'test_name': test_name,
            'hypothesis': hypothesis,
            'statistic': statistic,
            'p_value': p_value,
            'conclusion': conclusion,
            'details': details,
            'alpha': alpha,
            'correction_note': correction_note,
            'significant': p_value < alpha if not np.isnan(p_value) else False
        }
        self.results.append(result)
        return result
    
    def test_white_advantage(self) -> dict:
        """
        Test 1: Does playing as White give a significant win rate advantage?
        H0: Win rate as White = Win rate as Black
        H1: Win rate as White ≠ Win rate as Black
        """
        white_games = self.df[self.df['player_color'] == 'White']
        black_games = self.df[self.df['player_color'] == 'Black']
        
        white_wins = white_games['is_win'].sum()
        black_wins = black_games['is_win'].sum()
        
        white_n = len(white_games)
        black_n = len(black_games)
        
        white_winrate = white_wins / white_n * 100
        black_winrate = black_wins / black_n * 100
        
        # Two-proportion z-test
        count = np.array([white_wins, black_wins])
        nobs = np.array([white_n, black_n])
        stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
        
        conclusion = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"
        details = f"White Win Rate: {white_winrate:.2f}% (n={white_n}), Black Win Rate: {black_winrate:.2f}% (n={black_n})"
        
        return self._format_result(
            "White vs Black Win Rate (Two-Proportion Z-Test)",
            "H0: Win rate as White = Win rate as Black",
            stat,
            p_value,
            conclusion,
            details
        )
    
    def test_rating_diff_effect(self) -> dict:
        """
        Test 2: Does rating difference significantly affect win probability?
        H0: Rating difference has no effect on win rate
        H1: Rating difference affects win rate
        """
        # Higher rated games (player is higher rated)
        higher_rated = self.df[self.df['rating_diff'] > 50]
        # Lower rated games (opponent is higher rated)
        lower_rated = self.df[self.df['rating_diff'] < -50]
        
        higher_wins = higher_rated['is_win'].sum()
        lower_wins = lower_rated['is_win'].sum()
        
        higher_n = len(higher_rated)
        lower_n = len(lower_rated)
        
        if higher_n > 0 and lower_n > 0:
            higher_winrate = higher_wins / higher_n * 100
            lower_winrate = lower_wins / lower_n * 100
            
            # Two-proportion z-test
            count = np.array([higher_wins, lower_wins])
            nobs = np.array([higher_n, lower_n])
            stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
            
            conclusion = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"
            details = f"Win rate when higher rated (>50): {higher_winrate:.2f}% (n={higher_n}), Win rate when lower rated (<-50): {lower_winrate:.2f}% (n={lower_n})"
        else:
            stat, p_value = np.nan, np.nan
            conclusion = "INSUFFICIENT DATA"
            details = "Not enough games with significant rating differences"
        
        return self._format_result(
            "Rating Difference Effect (Two-Proportion Z-Test)",
            "H0: Rating difference does not affect win rate",
            stat,
            p_value,
            conclusion,
            details
        )
    
    def test_time_control_performance(self) -> dict:
        """
        Test 3: Is win rate significantly different across time controls?
        H0: Win rate is the same across all time controls
        H1: Win rate differs by time control
        """
        # Create contingency table
        contingency = pd.crosstab(self.df['time_class'], self.df['result'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Calculate win rates per time class
        tc_winrates = self.df.groupby('time_class')['is_win'].agg(['mean', 'count'])
        tc_winrates['win_rate'] = tc_winrates['mean'] * 100
        
        conclusion = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"
        details = "Win rates by time control: " + ", ".join(
            [f"{tc}: {row['win_rate']:.1f}% (n={int(row['count'])})" 
             for tc, row in tc_winrates.iterrows()]
        )
        
        return self._format_result(
            "Win Rate by Time Control (Chi-Square Test)",
            "H0: Win rate is the same across time controls",
            chi2,
            p_value,
            conclusion,
            details
        )
    
    def test_time_of_day_effect(self) -> dict:
        """
        Test 4: Does time of day significantly affect performance?
        H0: Win rate is the same regardless of time of day
        H1: Win rate differs by time of day

        Note: Bonferroni correction applied (α = 0.025) since this test and
        Test 5 (Day of Week) both examine temporal factors affecting win rate.
        """
        # Create contingency table
        contingency = pd.crosstab(self.df['time_of_day'], self.df['result'])

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Calculate win rates per time of day
        tod_winrates = self.df.groupby('time_of_day', observed=True)['is_win'].agg(['mean', 'count'])
        tod_winrates['win_rate'] = tod_winrates['mean'] * 100

        # Use Bonferroni-corrected alpha for temporal tests
        conclusion = "REJECT H0" if p_value < ALPHA_TEMPORAL_CORRECTED else "FAIL TO REJECT H0"
        details = "Win rates by time of day: " + ", ".join(
            [f"{tod}: {row['win_rate']:.1f}% (n={int(row['count'])})"
             for tod, row in tod_winrates.iterrows()]
        )

        return self._format_result(
            "Win Rate by Time of Day (Chi-Square Test)",
            "H0: Win rate is the same regardless of time of day",
            chi2,
            p_value,
            conclusion,
            details,
            alpha=ALPHA_TEMPORAL_CORRECTED,
            correction_note="Bonferroni correction applied (2 temporal tests)"
        )
    
    def test_day_of_week_effect(self) -> dict:
        """
        Test 5: Does day of week affect performance?
        H0: Win rate is the same across all days
        H1: Win rate differs by day of week

        Note: Bonferroni correction applied (α = 0.025) since this test and
        Test 4 (Time of Day) both examine temporal factors affecting win rate.
        """
        # Create contingency table
        contingency = pd.crosstab(self.df['day_of_week'], self.df['result'])

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Find best and worst days
        day_winrates = self.df.groupby('day_of_week')['is_win'].mean() * 100
        best_day = day_winrates.idxmax()
        worst_day = day_winrates.idxmin()

        # Use Bonferroni-corrected alpha for temporal tests
        conclusion = "REJECT H0" if p_value < ALPHA_TEMPORAL_CORRECTED else "FAIL TO REJECT H0"
        details = f"Best day: {best_day} ({day_winrates[best_day]:.1f}%), Worst day: {worst_day} ({day_winrates[worst_day]:.1f}%)"

        return self._format_result(
            "Win Rate by Day of Week (Chi-Square Test)",
            "H0: Win rate is the same across all days",
            chi2,
            p_value,
            conclusion,
            details,
            alpha=ALPHA_TEMPORAL_CORRECTED,
            correction_note="Bonferroni correction applied (2 temporal tests)"
        )
    
    def test_game_length_correlation(self) -> dict:
        """
        Test 6: Is there a correlation between game length and win probability?
        H0: No correlation between game length and win rate
        H1: Significant correlation exists
        """
        # Pearson correlation
        corr, p_value = pearsonr(self.df['move_count'], self.df['is_win'])
        
        # Spearman correlation for robustness
        spearman_corr, spearman_p = spearmanr(self.df['move_count'], self.df['is_win'])
        
        conclusion = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"
        details = f"Pearson r = {corr:.4f}, Spearman ρ = {spearman_corr:.4f}"
        
        return self._format_result(
            "Game Length vs Win Rate Correlation",
            "H0: No correlation between game length and winning",
            corr,
            p_value,
            conclusion,
            details
        )
    
    def test_rating_progression(self) -> dict:
        """
        Test 7: Has rating significantly improved over time?
        H0: No significant change in average rating over time
        H1: Significant change in average rating
        """
        # Compare first 20% of games to last 20%
        n = len(self.df)
        cutoff = int(n * 0.2)
        
        sorted_df = self.df.sort_values('date')
        early_games = sorted_df.head(cutoff)
        recent_games = sorted_df.tail(cutoff)
        
        early_rating = early_games['player_rating'].values
        recent_rating = recent_games['player_rating'].values
        
        # Mann-Whitney U test (non-parametric)
        stat, p_value = mannwhitneyu(early_rating, recent_rating, alternative='two-sided')
        
        early_mean = early_rating.mean()
        recent_mean = recent_rating.mean()
        change = recent_mean - early_mean
        
        conclusion = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"
        details = f"Early avg rating: {early_mean:.0f}, Recent avg rating: {recent_mean:.0f}, Change: {change:+.0f}"
        
        return self._format_result(
            "Rating Progression Over Time (Mann-Whitney U)",
            "H0: No significant change in rating over time",
            stat,
            p_value,
            conclusion,
            details
        )
    
    def test_opening_effect(self) -> dict:
        """
        Test 8: Do different openings lead to significantly different win rates?
        H0: Win rate is the same across all openings
        H1: Win rate differs by opening
        """
        # Filter to openings with at least 20 games
        opening_counts = self.df['opening'].value_counts()
        valid_openings = opening_counts[opening_counts >= 20].index
        filtered_df = self.df[self.df['opening'].isin(valid_openings)]
        
        if len(valid_openings) < 2:
            return self._format_result(
                "Win Rate by Opening (Chi-Square Test)",
                "H0: Win rate is the same across openings",
                np.nan,
                np.nan,
                "INSUFFICIENT DATA",
                "Not enough openings with sufficient games"
            )
        
        # Create contingency table
        contingency = pd.crosstab(filtered_df['opening'], filtered_df['result'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Find best and worst openings
        opening_winrates = filtered_df.groupby('opening')['is_win'].mean() * 100
        best_opening = opening_winrates.idxmax()
        worst_opening = opening_winrates.idxmin()
        
        conclusion = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"
        details = f"Best: {best_opening[:30]}... ({opening_winrates[best_opening]:.1f}%), Worst: {worst_opening[:30]}... ({opening_winrates[worst_opening]:.1f}%)"
        
        return self._format_result(
            "Win Rate by Opening (Chi-Square Test)",
            "H0: Win rate is the same across openings",
            chi2,
            p_value,
            conclusion,
            details
        )
    
    def test_win_rate_vs_fifty_percent(self) -> dict:
        """
        Test 9: Is the overall win rate significantly different from 50%?
        H0: Win rate = 50%
        H1: Win rate ≠ 50%
        """
        wins = self.df['is_win'].sum()
        n = len(self.df)
        expected_wins = n * 0.5
        
        # One-sample proportion z-test
        stat, p_value = proportions_ztest(wins, n, value=0.5, alternative='two-sided')
        
        actual_winrate = wins / n * 100
        
        conclusion = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"
        details = f"Actual win rate: {actual_winrate:.2f}% ({wins}/{n} games)"
        
        return self._format_result(
            "Win Rate vs 50% (One-Sample Z-Test)",
            "H0: Win rate = 50%",
            stat,
            p_value,
            conclusion,
            details
        )
    
    def test_winning_streak_effect(self) -> dict:
        """
        Test 10: Does a winning streak increase probability of winning the next game?
        H0: Previous result doesn't affect next game outcome
        H1: There's a momentum effect (winning begets winning)
        """
        sorted_df = self.df.sort_values('date').reset_index(drop=True)
        
        # Create lagged win column
        sorted_df['prev_win'] = sorted_df['is_win'].shift(1)
        sorted_df = sorted_df.dropna()
        
        # Win rate after a win vs after a loss
        after_win = sorted_df[sorted_df['prev_win'] == 1]
        after_loss = sorted_df[sorted_df['prev_win'] == 0]
        
        win_after_win = after_win['is_win'].sum()
        win_after_loss = after_loss['is_win'].sum()
        
        n_after_win = len(after_win)
        n_after_loss = len(after_loss)
        
        if n_after_win > 0 and n_after_loss > 0:
            wr_after_win = win_after_win / n_after_win * 100
            wr_after_loss = win_after_loss / n_after_loss * 100
            
            # Two-proportion z-test
            count = np.array([win_after_win, win_after_loss])
            nobs = np.array([n_after_win, n_after_loss])
            stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
            
            conclusion = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"
            details = f"Win rate after win: {wr_after_win:.1f}%, Win rate after loss: {wr_after_loss:.1f}%"
        else:
            stat, p_value = np.nan, np.nan
            conclusion = "INSUFFICIENT DATA"
            details = "Not enough sequential games"
        
        return self._format_result(
            "Momentum Effect (Two-Proportion Z-Test)",
            "H0: Previous game result doesn't affect next game",
            stat,
            p_value,
            conclusion,
            details
        )
    
    def run_all_tests(self) -> pd.DataFrame:
        """Run all hypothesis tests and return results"""
        print("\n" + "="*80)
        print("STATISTICAL HYPOTHESIS TESTING")
        print("="*80)
        print(f"\nSignificance level: α = 0.05 (standard)")
        print(f"Bonferroni-corrected α = 0.025 (for temporal tests H4 & H5)")
        print(f"Total games analyzed: {len(self.df)}")
        print(f"\nNote: 'Win rate' and 'win probability' are used interchangeably")
        
        tests = [
            self.test_white_advantage,
            self.test_rating_diff_effect,
            self.test_time_control_performance,
            self.test_time_of_day_effect,
            self.test_day_of_week_effect,
            self.test_game_length_correlation,
            self.test_rating_progression,
            self.test_opening_effect,
            self.test_win_rate_vs_fifty_percent,
            self.test_winning_streak_effect
        ]
        
        for i, test in enumerate(tests, 1):
            result = test()
            self._print_test_result(i, result)
        
        # Summary
        self._print_summary()
        
        # Return as DataFrame
        return pd.DataFrame(self.results)
    
    def _print_test_result(self, test_num: int, result: dict):
        """Print formatted test result"""
        print(f"\n{'─'*80}")
        print(f"TEST {test_num}: {result['test_name']}")
        print(f"{'─'*80}")
        print(f"Hypothesis: {result['hypothesis']}")
        print(f"Test Statistic: {result['statistic']:.4f}" if not np.isnan(result['statistic']) else "Test Statistic: N/A")
        print(f"P-value: {result['p_value']:.6f}" if not np.isnan(result['p_value']) else "P-value: N/A")
        print(f"Details: {result['details']}")

        # Show alpha level and correction note if applicable
        alpha = result.get('alpha', ALPHA_STANDARD)
        correction_note = result.get('correction_note', '')
        if correction_note:
            print(f"Note: {correction_note}")

        if result['significant']:
            print(f"Result: {result['conclusion']} - Statistically significant at α={alpha}")
        else:
            print(f"Result: {result['conclusion']} (α={alpha})")
    
    def _print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*80)
        print("HYPOTHESIS TESTING SUMMARY")
        print("="*80)
        
        significant_tests = [r for r in self.results if r['significant']]
        non_significant_tests = [r for r in self.results if not r['significant'] and not np.isnan(r['p_value'])]
        
        print(f"\nTotal tests conducted: {len(self.results)}")
        print(f"Significant results (p < 0.05): {len(significant_tests)}")
        print(f"Non-significant results: {len(non_significant_tests)}")
        
        if significant_tests:
            print("\n✅ SIGNIFICANT FINDINGS:")
            for r in significant_tests:
                print(f"   • {r['test_name']}: p = {r['p_value']:.6f}")
        
        if non_significant_tests:
            print("\n❌ NON-SIGNIFICANT FINDINGS:")
            for r in non_significant_tests:
                print(f"   • {r['test_name']}: p = {r['p_value']:.6f}")
        
        print("\n" + "="*80)


def run_hypothesis_tests(df: pd.DataFrame = None) -> pd.DataFrame:
    """Main function to run all hypothesis tests"""
    if df is None:
        # Load data from file
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Data file not found: {csv_path}\n"
                "Please run data collection first."
            )
        df = pd.read_csv(csv_path)
    
    tester = ChessHypothesisTester(df)
    results_df = tester.run_all_tests()
    
    # Save results
    os.makedirs(DATA_DIR, exist_ok=True)
    results_path = os.path.join(DATA_DIR, 'hypothesis_test_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return results_df


if __name__ == "__main__":
    run_hypothesis_tests()


"""
Exploratory Data Analysis (EDA) Module
Analyzes chess game data and creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

from config import DATA_DIR, VISUALIZATIONS_DIR, USERNAME

warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChessEDA:
    """Exploratory Data Analysis for Chess.com game data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.username = USERNAME
        self._prepare_data()
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        
    def _prepare_data(self):
        """Prepare and clean data for analysis"""
        # Ensure date is datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['year_month'] = self.df['date'].dt.to_period('M')
        
        # Create numeric result column (1 for win, 0.5 for draw, 0 for loss)
        result_map = {'Win': 1, 'Draw': 0.5, 'Loss': 0}
        self.df['result_numeric'] = self.df['result'].map(result_map)
        
        # Calculate win rate features
        self.df['is_win'] = (self.df['result'] == 'Win').astype(int)
        self.df['is_loss'] = (self.df['result'] == 'Loss').astype(int)
        self.df['is_draw'] = (self.df['result'] == 'Draw').astype(int)
        
    def summary_statistics(self) -> dict:
        """Generate summary statistics"""
        stats = {
            'total_games': len(self.df),
            'wins': len(self.df[self.df['result'] == 'Win']),
            'losses': len(self.df[self.df['result'] == 'Loss']),
            'draws': len(self.df[self.df['result'] == 'Draw']),
            'win_rate': self.df['is_win'].mean() * 100,
            'draw_rate': self.df['is_draw'].mean() * 100,
            'loss_rate': self.df['is_loss'].mean() * 100,
            'avg_rating': self.df['player_rating'].mean(),
            'max_rating': self.df['player_rating'].max(),
            'min_rating': self.df['player_rating'].min(),
            'avg_opponent_rating': self.df['opponent_rating'].mean(),
            'avg_moves_per_game': self.df['move_count'].mean(),
            'games_as_white': len(self.df[self.df['player_color'] == 'White']),
            'games_as_black': len(self.df[self.df['player_color'] == 'Black']),
        }
        
        # Win rate by color
        white_games = self.df[self.df['player_color'] == 'White']
        black_games = self.df[self.df['player_color'] == 'Black']
        stats['win_rate_white'] = white_games['is_win'].mean() * 100 if len(white_games) > 0 else 0
        stats['win_rate_black'] = black_games['is_win'].mean() * 100 if len(black_games) > 0 else 0
        
        return stats
    
    def print_summary(self):
        """Print formatted summary statistics"""
        stats = self.summary_statistics()
        
        print("\n" + "="*70)
        print("CHESS.COM PERFORMANCE ANALYSIS - SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Total Games Played: {stats['total_games']:,}")
        print(f"   Date Range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}")
        
        print(f"\n🏆 RESULTS")
        print(f"   Wins: {stats['wins']:,} ({stats['win_rate']:.1f}%)")
        print(f"   Losses: {stats['losses']:,} ({stats['loss_rate']:.1f}%)")
        print(f"   Draws: {stats['draws']:,} ({stats['draw_rate']:.1f}%)")
        
        print(f"\n📈 RATING")
        print(f"   Average Rating: {stats['avg_rating']:.0f}")
        print(f"   Highest Rating: {stats['max_rating']:.0f}")
        print(f"   Lowest Rating: {stats['min_rating']:.0f}")
        print(f"   Average Opponent Rating: {stats['avg_opponent_rating']:.0f}")
        
        print(f"\n♟️ COLOR STATISTICS")
        print(f"   Games as White: {stats['games_as_white']:,} (Win Rate: {stats['win_rate_white']:.1f}%)")
        print(f"   Games as Black: {stats['games_as_black']:,} (Win Rate: {stats['win_rate_black']:.1f}%)")
        
        print(f"\n🎯 GAME LENGTH")
        print(f"   Average Moves per Game: {stats['avg_moves_per_game']:.1f}")
        
        # Time control breakdown
        print(f"\n⏱️ TIME CONTROL DISTRIBUTION")
        time_dist = self.df['time_class'].value_counts()
        for tc, count in time_dist.items():
            pct = count / len(self.df) * 100
            print(f"   {tc.title()}: {count:,} ({pct:.1f}%)")
        
        print("\n" + "="*70)
    
    def plot_rating_over_time(self, time_class: str = None):
        """Plot rating progression over time"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if time_class:
            data = self.df[self.df['time_class'] == time_class].copy()
            title = f'Rating Progression Over Time ({time_class.title()})'
        else:
            data = self.df.copy()
            title = 'Rating Progression Over Time (All Games)'
        
        data = data.sort_values('date')
        
        # Plot rating over time
        ax.plot(data['date'], data['player_rating'], alpha=0.3, linewidth=0.5)
        
        # Add rolling average
        rolling_avg = data['player_rating'].rolling(window=50, min_periods=1).mean()
        ax.plot(data['date'], rolling_avg, color='red', linewidth=2, label='50-game Rolling Average')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'rating_over_time.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
        
    def plot_rating_by_time_class(self):
        """Plot rating progression for each time class"""
        time_classes = self.df['time_class'].unique()
        n_classes = len(time_classes)
        
        fig, axes = plt.subplots(n_classes, 1, figsize=(14, 4*n_classes))
        if n_classes == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", n_classes)
        
        for ax, tc, color in zip(axes, time_classes, colors):
            data = self.df[self.df['time_class'] == tc].sort_values('date')
            
            ax.plot(data['date'], data['player_rating'], alpha=0.3, linewidth=0.5, color=color)
            rolling_avg = data['player_rating'].rolling(window=30, min_periods=1).mean()
            ax.plot(data['date'], rolling_avg, color=color, linewidth=2, label='30-game Rolling Avg')
            
            ax.set_title(f'{tc.title()} Rating ({len(data):,} games)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Rating')
            ax.legend(loc='upper left')
            
            # Add max rating annotation
            max_idx = data['player_rating'].idxmax()
            max_rating = data.loc[max_idx, 'player_rating']
            max_date = data.loc[max_idx, 'date']
            ax.annotate(f'Peak: {max_rating:.0f}', xy=(max_date, max_rating),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, color='darkred')
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'rating_by_time_class.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_results_distribution(self):
        """Plot overall results distribution"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Overall results pie chart
        results = self.df['result'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        axes[0].pie(results.values, labels=results.index, autopct='%1.1f%%',
                   colors=colors, explode=(0.05, 0.05, 0.05))
        axes[0].set_title('Overall Results Distribution', fontsize=12, fontweight='bold')
        
        # Results by color
        color_results = self.df.groupby(['player_color', 'result']).size().unstack(fill_value=0)
        color_results_pct = color_results.div(color_results.sum(axis=1), axis=0) * 100
        color_results_pct[['Win', 'Draw', 'Loss']].plot(kind='bar', ax=axes[1], 
                                                         color=['#2ecc71', '#95a5a6', '#e74c3c'])
        axes[1].set_title('Results by Color', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Player Color')
        axes[1].set_ylabel('Percentage')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].legend(title='Result')
        
        # Results by time class
        tc_results = self.df.groupby(['time_class', 'result']).size().unstack(fill_value=0)
        tc_results_pct = tc_results.div(tc_results.sum(axis=1), axis=0) * 100
        tc_results_pct[['Win', 'Draw', 'Loss']].plot(kind='bar', ax=axes[2],
                                                      color=['#2ecc71', '#95a5a6', '#e74c3c'])
        axes[2].set_title('Results by Time Class', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time Class')
        axes[2].set_ylabel('Percentage')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        axes[2].legend(title='Result')
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'results_distribution.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_games_by_time(self):
        """Plot game activity patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Games per month
        monthly_games = self.df.groupby('year_month').size()
        ax = axes[0, 0]
        monthly_games.plot(kind='line', ax=ax, marker='o', markersize=3)
        ax.set_title('Games Played per Month', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Games')
        ax.tick_params(axis='x', rotation=45)
        
        # Games by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.df['day_of_week'].value_counts().reindex(day_order)
        ax = axes[0, 1]
        day_counts.plot(kind='bar', ax=ax, color=sns.color_palette("husl", 7))
        ax.set_title('Games by Day of Week', fontsize=12, fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('Number of Games')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # Games by hour
        ax = axes[1, 0]
        hour_counts = self.df['hour'].value_counts().sort_index()
        hour_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Games by Hour of Day', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour (24h)')
        ax.set_ylabel('Number of Games')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # Win rate by hour
        ax = axes[1, 1]
        hourly_winrate = self.df.groupby('hour')['is_win'].mean() * 100
        ax.bar(hourly_winrate.index, hourly_winrate.values, color='green', alpha=0.7)
        ax.axhline(y=self.df['is_win'].mean() * 100, color='red', linestyle='--', 
                   label=f'Overall Win Rate: {self.df["is_win"].mean()*100:.1f}%')
        ax.set_title('Win Rate by Hour of Day', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour (24h)')
        ax.set_ylabel('Win Rate (%)')
        ax.legend()
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'games_by_time.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_opening_analysis(self, top_n: int = 15):
        """Analyze most played openings"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Most played openings
        opening_counts = self.df['opening'].value_counts().head(top_n)
        ax = axes[0]
        opening_counts.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(f'Top {top_n} Most Played Openings', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Games')
        ax.invert_yaxis()
        
        # Win rate by opening (minimum 10 games)
        opening_stats = self.df.groupby('opening').agg({
            'is_win': ['mean', 'count']
        }).droplevel(0, axis=1)
        opening_stats.columns = ['win_rate', 'count']
        opening_stats = opening_stats[opening_stats['count'] >= 10]
        opening_stats['win_rate'] = opening_stats['win_rate'] * 100
        top_openings = opening_stats.nlargest(top_n, 'count')
        
        ax = axes[1]
        colors = ['green' if wr > 50 else 'red' for wr in top_openings['win_rate']]
        ax.barh(range(len(top_openings)), top_openings['win_rate'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_openings)))
        ax.set_yticklabels(top_openings.index)
        ax.axvline(x=50, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'Win Rate by Opening (min 10 games)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Win Rate (%)')
        ax.invert_yaxis()
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'opening_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_rating_difference_analysis(self):
        """Analyze performance based on rating difference"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Create rating difference bins
        bins = [-1000, -200, -100, -50, 0, 50, 100, 200, 1000]
        labels = ['<-200', '-200 to -100', '-100 to -50', '-50 to 0', 
                  '0 to 50', '50 to 100', '100 to 200', '>200']
        self.df['rating_diff_bin'] = pd.cut(self.df['rating_diff'], bins=bins, labels=labels)
        
        # Win rate by rating difference
        ax = axes[0]
        diff_winrate = self.df.groupby('rating_diff_bin', observed=True)['is_win'].mean() * 100
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(diff_winrate)))
        diff_winrate.plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Win Rate by Rating Difference\n(Your Rating - Opponent Rating)', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Rating Difference')
        ax.set_ylabel('Win Rate (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        
        # Scatter plot of rating diff vs result
        ax = axes[1]
        jitter = np.random.normal(0, 0.05, len(self.df))
        colors = {'Win': 'green', 'Draw': 'gray', 'Loss': 'red'}
        for result in ['Win', 'Draw', 'Loss']:
            mask = self.df['result'] == result
            ax.scatter(self.df.loc[mask, 'rating_diff'], 
                      self.df.loc[mask, 'result_numeric'] + jitter[mask],
                      alpha=0.3, s=10, label=result, c=colors[result])
        ax.set_xlabel('Rating Difference (Your Rating - Opponent Rating)')
        ax.set_ylabel('Result')
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['Loss', 'Draw', 'Win'])
        ax.set_title('Game Results vs Rating Difference', fontsize=12, fontweight='bold')
        ax.legend()
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'rating_difference_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def _simplify_termination(self, term: str) -> str:
        """Simplify termination string to basic type"""
        if pd.isna(term):
            return 'Unknown'
        term_lower = term.lower()
        if 'checkmate' in term_lower:
            return 'Checkmate'
        elif 'time' in term_lower and 'insufficient' not in term_lower:
            return 'Timeout'
        elif 'resign' in term_lower:
            return 'Resignation'
        elif 'stalemate' in term_lower:
            return 'Stalemate'
        elif 'repetition' in term_lower:
            return 'Repetition'
        elif 'insufficient' in term_lower:
            return 'Insufficient Material'
        elif 'abandon' in term_lower:
            return 'Abandoned'
        elif 'agreement' in term_lower:
            return 'Draw by Agreement'
        elif '50' in term_lower or 'fifty' in term_lower:
            return '50-Move Rule'
        else:
            return 'Other'
    
    def plot_termination_analysis(self):
        """Analyze how games end"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Simplify termination types
        self.df['termination_simple'] = self.df['termination'].apply(self._simplify_termination)
        
        # Overall termination distribution
        ax = axes[0]
        term_counts = self.df['termination_simple'].value_counts()
        colors_map = {
            'Checkmate': '#e74c3c',
            'Timeout': '#3498db', 
            'Resignation': '#9b59b6',
            'Stalemate': '#95a5a6',
            'Repetition': '#f39c12',
            'Insufficient Material': '#1abc9c',
            'Abandoned': '#e67e22',
            'Draw by Agreement': '#34495e',
            '50-Move Rule': '#7f8c8d',
            'Other': '#bdc3c7'
        }
        bar_colors = [colors_map.get(t, '#bdc3c7') for t in term_counts.index]
        term_counts.plot(kind='barh', ax=ax, color=bar_colors)
        ax.set_title('Game Termination Types', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Games')
        ax.set_ylabel('')
        ax.invert_yaxis()
        
        # Add count labels on bars
        for i, (count, term) in enumerate(zip(term_counts.values, term_counts.index)):
            ax.text(count + 20, i, f'{count:,}', va='center', fontsize=9)
        
        # Win rate by termination type
        ax = axes[1]
        term_stats = self.df.groupby('termination_simple').agg({
            'is_win': 'mean',
            'result': 'count'
        })
        term_stats.columns = ['win_rate', 'count']
        term_stats['win_rate'] = term_stats['win_rate'] * 100
        term_stats = term_stats.sort_values('count', ascending=False)
        
        colors = ['#2ecc71' if wr > 50 else '#e74c3c' if wr < 50 else '#95a5a6' 
                  for wr in term_stats['win_rate']]
        bars = ax.barh(range(len(term_stats)), term_stats['win_rate'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(term_stats)))
        ax.set_yticklabels(term_stats.index)
        ax.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='50%')
        ax.set_title('Win Rate by Termination Type', fontsize=12, fontweight='bold')
        ax.set_xlabel('Win Rate (%)')
        ax.set_xlim(0, 105)
        ax.invert_yaxis()
        
        # Add percentage labels on bars
        for i, (wr, count) in enumerate(zip(term_stats['win_rate'], term_stats['count'])):
            ax.text(wr + 2, i, f'{wr:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'termination_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_game_length_analysis(self):
        """Analyze game length patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribution of game lengths
        ax = axes[0, 0]
        self.df['move_count'].hist(bins=50, ax=ax, color='steelblue', edgecolor='black')
        ax.axvline(x=self.df['move_count'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["move_count"].mean():.1f}')
        ax.axvline(x=self.df['move_count'].median(), color='green', linestyle='--',
                   label=f'Median: {self.df["move_count"].median():.1f}')
        ax.set_title('Distribution of Game Length (Moves)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Moves')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Game length by result
        ax = axes[0, 1]
        self.df.boxplot(column='move_count', by='result', ax=ax)
        ax.set_title('Game Length by Result', fontsize=12, fontweight='bold')
        ax.set_xlabel('Result')
        ax.set_ylabel('Number of Moves')
        plt.suptitle('')
        
        # Game length by time class
        ax = axes[1, 0]
        self.df.boxplot(column='move_count', by='time_class', ax=ax)
        ax.set_title('Game Length by Time Class', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Class')
        ax.set_ylabel('Number of Moves')
        plt.suptitle('')
        
        # Win rate by game length bins
        ax = axes[1, 1]
        self.df['move_bins'] = pd.cut(self.df['move_count'], 
                                       bins=[0, 20, 40, 60, 80, 100, 200],
                                       labels=['0-20', '21-40', '41-60', '61-80', '81-100', '100+'])
        move_winrate = self.df.groupby('move_bins', observed=True)['is_win'].mean() * 100
        move_winrate.plot(kind='bar', ax=ax, color='steelblue')
        ax.axhline(y=self.df['is_win'].mean() * 100, color='red', linestyle='--',
                   label=f'Overall: {self.df["is_win"].mean()*100:.1f}%')
        ax.set_title('Win Rate by Game Length', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Moves')
        ax.set_ylabel('Win Rate (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend()
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'game_length_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_heatmap_day_hour(self):
        """Create heatmap of games by day and hour"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Game count heatmap
        pivot_count = self.df.pivot_table(
            values='game_id', 
            index='day_of_week', 
            columns='hour', 
            aggfunc='count',
            fill_value=0
        ).reindex(day_order)
        
        ax = axes[0]
        sns.heatmap(pivot_count, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Games'})
        ax.set_title('Game Activity Heatmap\n(Day vs Hour)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        
        # Win rate heatmap
        pivot_winrate = self.df.pivot_table(
            values='is_win', 
            index='day_of_week', 
            columns='hour', 
            aggfunc='mean',
            fill_value=0.5
        ).reindex(day_order) * 100
        
        ax = axes[1]
        sns.heatmap(pivot_winrate, cmap='RdYlGn', center=50, ax=ax, 
                    cbar_kws={'label': 'Win Rate (%)'}, vmin=30, vmax=70)
        ax.set_title('Win Rate Heatmap\n(Day vs Hour)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        
        plt.tight_layout()
        filepath = os.path.join(VISUALIZATIONS_DIR, 'heatmap_day_hour.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def run_full_eda(self):
        """Run complete EDA and generate all visualizations"""
        print("\n" + "="*70)
        print("RUNNING EXPLORATORY DATA ANALYSIS")
        print("="*70 + "\n")
        
        # Print summary
        self.print_summary()
        
        # Generate all visualizations
        print("\n📊 Generating visualizations...\n")
        
        self.plot_rating_over_time()
        self.plot_rating_by_time_class()
        self.plot_results_distribution()
        self.plot_games_by_time()
        self.plot_opening_analysis()
        self.plot_rating_difference_analysis()
        self.plot_termination_analysis()
        self.plot_game_length_analysis()
        self.plot_heatmap_day_hour()
        
        print(f"\n✅ All visualizations saved to '{VISUALIZATIONS_DIR}/' directory")
        
        return self.summary_statistics()


def run_eda(df: pd.DataFrame = None) -> dict:
    """Main function to run EDA"""
    if df is None:
        # Load data from file
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Data file not found: {csv_path}\n"
                "Please run data collection first."
            )
        df = pd.read_csv(csv_path)
    
    eda = ChessEDA(df)
    stats = eda.run_full_eda()
    
    return stats


if __name__ == "__main__":
    run_eda()


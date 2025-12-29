"""
Time Series Forecasting for Chess Rating Prediction
Uses Prophet and ARIMA to forecast future rating
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, VISUALIZATIONS_DIR, USERNAME

warnings.filterwarnings('ignore')


class ChessTimeSeries:
    """Time series forecasting for chess rating prediction"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.prophet_model = None
        self.arima_model = None
        self.viz_dir = os.path.join(VISUALIZATIONS_DIR, 'ml')
        os.makedirs(self.viz_dir, exist_ok=True)

    def prepare_data(self, freq='W'):
        """Prepare time series data for forecasting"""
        df = self.df.copy()

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Aggregate by week (or specified frequency)
        df = df.sort_values('date')

        # Group by week and calculate average rating
        df['period'] = df['date'].dt.to_period(freq)
        weekly_rating = df.groupby('period').agg({
            'player_rating': 'mean',
            'result': lambda x: (x == 'Win').mean() * 100,  # win rate
            'game_id': 'count'  # number of games
        }).reset_index()

        weekly_rating.columns = ['period', 'avg_rating', 'win_rate', 'n_games']
        weekly_rating['date'] = weekly_rating['period'].dt.to_timestamp()

        return weekly_rating

    def fit_prophet(self, weekly_data, periods=26):
        """Fit Prophet model and make forecasts"""
        try:
            from prophet import Prophet

            # Prepare data for Prophet
            prophet_df = weekly_data[['date', 'avg_rating']].copy()
            prophet_df.columns = ['ds', 'y']

            # Remove any NaN values
            prophet_df = prophet_df.dropna()

            # Fit Prophet model
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            self.prophet_model.fit(prophet_df)

            # Make future dataframe
            future = self.prophet_model.make_future_dataframe(periods=periods, freq='W')

            # Predict
            forecast = self.prophet_model.predict(future)

            results = {
                'model': self.prophet_model,
                'forecast': forecast,
                'historical': prophet_df,
                'periods': periods
            }

            return results

        except ImportError:
            print("  Prophet not installed. Skipping Prophet forecasting.")
            return None

    def fit_arima(self, weekly_data, order=(2, 1, 2), periods=26):
        """Fit ARIMA model and make forecasts"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Prepare data
            ts = weekly_data.set_index('date')['avg_rating'].dropna()

            # Fit ARIMA model
            self.arima_model = ARIMA(ts, order=order)
            arima_fit = self.arima_model.fit()

            # Forecast
            forecast = arima_fit.forecast(steps=periods)
            conf_int = arima_fit.get_forecast(steps=periods).conf_int()

            # Create forecast dates
            last_date = ts.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(weeks=1),
                                           periods=periods, freq='W')

            results = {
                'model': arima_fit,
                'forecast': forecast,
                'forecast_dates': forecast_dates,
                'conf_int': conf_int,
                'historical': ts,
                'aic': arima_fit.aic,
                'bic': arima_fit.bic
            }

            return results

        except Exception as e:
            print(f"  ARIMA fitting failed: {e}")
            return None

    def plot_prophet_forecast(self, prophet_results, weekly_data):
        """Plot Prophet forecast"""
        if prophet_results is None:
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        forecast = prophet_results['forecast']
        historical = prophet_results['historical']

        # Main forecast plot
        ax = axes[0]
        ax.plot(historical['ds'], historical['y'], 'b.', alpha=0.5, label='Historical')
        ax.plot(forecast['ds'], forecast['yhat'], 'r-', linewidth=2, label='Forecast')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                        alpha=0.2, color='red', label='Confidence Interval')

        # Mark forecast start
        last_historical = historical['ds'].max()
        ax.axvline(x=last_historical, color='green', linestyle='--', alpha=0.7,
                   label='Forecast Start')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        ax.set_title('Prophet Rating Forecast (6 Months)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Trend and components
        ax = axes[1]
        ax.plot(forecast['ds'], forecast['trend'], 'g-', linewidth=2, label='Trend')
        if 'yearly' in forecast.columns:
            yearly_component = forecast['yearly'] + forecast['trend'].mean()
            ax.plot(forecast['ds'], yearly_component, 'orange', linewidth=1,
                    alpha=0.7, label='Yearly Seasonality')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        ax.set_title('Trend Component', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'ts_prophet_forecast.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_arima_forecast(self, arima_results):
        """Plot ARIMA forecast"""
        if arima_results is None:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        historical = arima_results['historical']
        forecast = arima_results['forecast']
        forecast_dates = arima_results['forecast_dates']
        conf_int = arima_results['conf_int']

        # Plot historical
        ax.plot(historical.index, historical.values, 'b-', linewidth=1.5,
                alpha=0.7, label='Historical')

        # Plot forecast
        ax.plot(forecast_dates, forecast.values, 'r-', linewidth=2, label='ARIMA Forecast')
        ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        alpha=0.2, color='red', label='95% Confidence Interval')

        # Mark forecast start
        ax.axvline(x=historical.index[-1], color='green', linestyle='--',
                   alpha=0.7, label='Forecast Start')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        ax.set_title(f'ARIMA Rating Forecast (AIC: {arima_results["aic"]:.1f})',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'ts_arima_forecast.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_rating_decomposition(self, weekly_data):
        """Plot time series decomposition"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            ts = weekly_data.set_index('date')['avg_rating'].dropna()

            # Need at least 2 full cycles for decomposition
            if len(ts) < 52:
                period = len(ts) // 4
            else:
                period = 52  # yearly seasonality

            if period < 2:
                print("  Not enough data for decomposition")
                return

            decomposition = seasonal_decompose(ts, model='additive', period=period)

            fig, axes = plt.subplots(4, 1, figsize=(14, 12))

            axes[0].plot(ts.index, decomposition.observed, 'b-')
            axes[0].set_ylabel('Observed', fontsize=10)
            axes[0].set_title('Time Series Decomposition of Rating', fontsize=14, fontweight='bold')

            axes[1].plot(ts.index, decomposition.trend, 'g-')
            axes[1].set_ylabel('Trend', fontsize=10)

            axes[2].plot(ts.index, decomposition.seasonal, 'orange')
            axes[2].set_ylabel('Seasonal', fontsize=10)

            axes[3].plot(ts.index, decomposition.resid, 'r-', alpha=0.7)
            axes[3].set_ylabel('Residual', fontsize=10)
            axes[3].set_xlabel('Date', fontsize=12)

            for ax in axes:
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filepath = os.path.join(self.viz_dir, 'ts_decomposition.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {filepath}")

        except Exception as e:
            print(f"  Decomposition failed: {e}")

    def plot_comparison(self, weekly_data, prophet_results, arima_results):
        """Plot comparison of actual vs forecasts"""
        fig, ax = plt.subplots(figsize=(14, 7))

        # Historical data
        ax.plot(weekly_data['date'], weekly_data['avg_rating'], 'b-',
                linewidth=1.5, alpha=0.8, label='Historical Rating')

        # Prophet forecast
        if prophet_results is not None:
            forecast = prophet_results['forecast']
            future_mask = forecast['ds'] > weekly_data['date'].max()
            ax.plot(forecast.loc[future_mask, 'ds'], forecast.loc[future_mask, 'yhat'],
                    'r--', linewidth=2, label='Prophet Forecast')

        # ARIMA forecast
        if arima_results is not None:
            ax.plot(arima_results['forecast_dates'], arima_results['forecast'].values,
                    'g--', linewidth=2, label='ARIMA Forecast')

        ax.axvline(x=weekly_data['date'].max(), color='gray', linestyle=':',
                   alpha=0.7, label='Forecast Start')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        ax.set_title('Rating Forecast Comparison: Prophet vs ARIMA',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.viz_dir, 'ts_forecast_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def run_analysis(self, forecast_periods=26):
        """Run complete time series analysis"""
        print("\n" + "="*60)
        print("TIME SERIES FORECASTING")
        print("="*60)

        # Prepare data
        print("\nPreparing weekly aggregated data...")
        weekly_data = self.prepare_data(freq='W')
        print(f"  Data range: {weekly_data['date'].min().date()} to {weekly_data['date'].max().date()}")
        print(f"  Total weeks: {len(weekly_data)}")
        print(f"  Rating range: {weekly_data['avg_rating'].min():.0f} - {weekly_data['avg_rating'].max():.0f}")

        results = {
            'weekly_data': weekly_data
        }

        # Fit Prophet
        print("\nFitting Prophet model...")
        prophet_results = self.fit_prophet(weekly_data, periods=forecast_periods)
        results['prophet'] = prophet_results

        if prophet_results is not None:
            future_forecast = prophet_results['forecast'].tail(forecast_periods)
            print(f"  Prophet forecast (next {forecast_periods} weeks):")
            print(f"    Current rating: {weekly_data['avg_rating'].iloc[-1]:.0f}")
            print(f"    Forecasted rating: {future_forecast['yhat'].iloc[-1]:.0f}")
            print(f"    Range: {future_forecast['yhat_lower'].iloc[-1]:.0f} - {future_forecast['yhat_upper'].iloc[-1]:.0f}")

        # Fit ARIMA
        print("\nFitting ARIMA model...")
        arima_results = self.fit_arima(weekly_data, periods=forecast_periods)
        results['arima'] = arima_results

        if arima_results is not None:
            print(f"  ARIMA forecast (next {forecast_periods} weeks):")
            print(f"    Current rating: {arima_results['historical'].iloc[-1]:.0f}")
            print(f"    Forecasted rating: {arima_results['forecast'].iloc[-1]:.0f}")
            print(f"    Model AIC: {arima_results['aic']:.2f}")
            print(f"    Model BIC: {arima_results['bic']:.2f}")

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_rating_decomposition(weekly_data)
        self.plot_prophet_forecast(prophet_results, weekly_data)
        self.plot_arima_forecast(arima_results)
        self.plot_comparison(weekly_data, prophet_results, arima_results)

        # Summary
        print("\nForecast Summary:")
        current_rating = weekly_data['avg_rating'].iloc[-1]
        print(f"  Current average rating: {current_rating:.0f}")

        if prophet_results is not None:
            prophet_future = prophet_results['forecast']['yhat'].iloc[-1]
            print(f"  Prophet 6-month forecast: {prophet_future:.0f} ({prophet_future - current_rating:+.0f})")

        if arima_results is not None:
            arima_future = arima_results['forecast'].iloc[-1]
            print(f"  ARIMA 6-month forecast: {arima_future:.0f} ({arima_future - current_rating:+.0f})")

        return results


def run_time_series_forecast(df: pd.DataFrame = None) -> dict:
    """Main function to run time series forecasting"""
    if df is None:
        csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
        df = pd.read_csv(csv_path)

    ts = ChessTimeSeries(df)
    results = ts.run_analysis()
    return results


if __name__ == "__main__":
    results = run_time_series_forecast()

#!/usr/bin/env python3
"""
Chess.com Performance Analysis - Main Pipeline
DSA 210 Project - Data Collection, EDA, Hypothesis Testing, and Machine Learning

Author: Ramazan YILDIRIM

This script orchestrates the entire data science pipeline:
1. Data Collection from Chess.com API
2. Exploratory Data Analysis (EDA) with visualizations
3. Statistical Hypothesis Testing
4. Machine Learning Analysis (7 ML methods)
"""

import sys
import os
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import USERNAME, DATA_DIR
from src.data_collection import collect_data
from src.eda import run_eda
from src.hypothesis_tests import run_hypothesis_tests
from src.ml import run_all_ml_models


def print_header():
    """Print project header"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  CHESS.COM PERFORMANCE ANALYSIS  ".center(78) + "█")
    print("█" + "  DSA 210 - Introduction to Data Science  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)


def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("MAIN MENU")
    print("="*60)
    print("\n  [1] Collect Data from Chess.com API")
    print("  [2] Run Exploratory Data Analysis (EDA)")
    print("  [3] Run Hypothesis Testing")
    print("  [4] Run Machine Learning Analysis")
    print("  [5] Run All (Complete Pipeline)")
    print("  [0] Exit")
    print("\n" + "-"*60)


def load_existing_data():
    """Load existing data from CSV file"""
    csv_path = os.path.join(DATA_DIR, f"{USERNAME}_games.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"\n✅ Loaded existing data: {len(df)} games from {csv_path}")
        return df
    else:
        print(f"\n⚠️  No existing data found at {csv_path}")
        print("   Please run option [1] to collect data first.")
        return None


def run_data_collection():
    """Run data collection step"""
    print("\n" + "="*80)
    print("STEP 1: DATA COLLECTION")
    print("="*80)

    try:
        df = collect_data(USERNAME)

        if df.empty:
            print("\n⚠️  No data collected. Please check your username and try again.")
            return None

        print(f"\n✅ Data collection complete! {len(df)} games collected.")
        return df

    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_eda_analysis(df):
    """Run EDA step"""
    print("\n" + "="*80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*80)

    try:
        stats = run_eda(df)
        print(f"\n✅ EDA complete! Visualizations saved to visualizations/")
        return stats

    except Exception as e:
        print(f"\n❌ Error during EDA: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_hypothesis_testing(df):
    """Run hypothesis testing step"""
    print("\n" + "="*80)
    print("STEP 3: HYPOTHESIS TESTING")
    print("="*80)

    try:
        results_df = run_hypothesis_tests(df)
        print(f"\n✅ Hypothesis testing complete!")
        return results_df

    except Exception as e:
        print(f"\n❌ Error during hypothesis testing: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_ml_analysis(df):
    """Run machine learning analysis step"""
    print("\n" + "="*80)
    print("STEP 4: MACHINE LEARNING ANALYSIS")
    print("="*80)

    try:
        ml_results = run_all_ml_models(df)
        print(f"\n✅ Machine learning analysis complete!")
        print(f"   Visualizations saved to visualizations/ml/")
        return ml_results

    except Exception as e:
        print(f"\n❌ Error during ML analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_steps():
    """Run the complete pipeline"""
    # Step 1: Data Collection
    df = run_data_collection()
    if df is None:
        return

    # Step 2: EDA
    run_eda_analysis(df)

    # Step 3: Hypothesis Testing
    run_hypothesis_testing(df)

    # Step 4: Machine Learning
    run_ml_analysis(df)

    # Final Summary
    print_final_summary()


def print_final_summary():
    """Print final summary"""
    print("\n" + "█" * 80)
    print("█" + " ANALYSIS COMPLETE ".center(78, "█") + "█")
    print("█" * 80)
    print("\n📁 Output Files:")
    print(f"   • data/{USERNAME}_games.csv - Game data (CSV)")
    print(f"   • data/{USERNAME}_games_full.json - Full game data with PGN")
    print(f"   • data/hypothesis_test_results.csv - Statistical test results")
    print(f"   • visualizations/ - EDA plots")
    print(f"   • visualizations/ml/ - Machine learning plots")
    print("\n📊 ML Models Applied:")
    print("   • Random Forest Classifier")
    print("   • Decision Tree Classifier")
    print("   • K-Nearest Neighbors")
    print("   • K-Means Clustering")
    print("   • DBSCAN Clustering")
    print("   • PCA Dimensionality Reduction")
    print("   • Time Series Forecasting (Prophet/ARIMA)")


def main():
    """Main function with interactive menu"""
    print_header()

    # Check if username is configured
    if USERNAME == "your_username_here":
        print("\n⚠️  ERROR: Please update your Chess.com username in src/config.py")
        print("   Open src/config.py and change USERNAME = 'your_username_here'")
        print("   to USERNAME = 'your_actual_chess_com_username'")
        sys.exit(1)

    print(f"\n📋 Configuration:")
    print(f"   Username: {USERNAME}")
    print(f"   Data Directory: {DATA_DIR}/")

    df = None  # Will hold the loaded/collected data

    while True:
        print_menu()

        try:
            choice = input("Enter your choice [0-5]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if choice == '0':
            print("\n👋 Thank you for using Chess.com Performance Analysis!")
            print("="*60 + "\n")
            break

        elif choice == '1':
            df = run_data_collection()

        elif choice == '2':
            if df is None:
                df = load_existing_data()
            if df is not None:
                run_eda_analysis(df)

        elif choice == '3':
            if df is None:
                df = load_existing_data()
            if df is not None:
                run_hypothesis_testing(df)

        elif choice == '4':
            if df is None:
                df = load_existing_data()
            if df is not None:
                run_ml_analysis(df)

        elif choice == '5':
            run_all_steps()

        else:
            print("\n❌ Invalid choice. Please enter a number between 0 and 5.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Chess.com Performance Analysis - Main Pipeline
DSA 210 Project - Data Collection, EDA, and Hypothesis Testing

Author: Your Name
Date: November 2024

This script orchestrates the entire data science pipeline:
1. Data Collection from Chess.com API
2. Exploratory Data Analysis (EDA) with visualizations
3. Statistical Hypothesis Testing
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import USERNAME, DATA_DIR
from src.data_collection import collect_data
from src.eda import run_eda
from src.hypothesis_tests import run_hypothesis_tests


def print_header():
    """Print project header"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  CHESS.COM PERFORMANCE ANALYSIS  ".center(78) + "█")
    print("█" + "  DSA 210 - Introduction to Data Science  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)


def main():
    """Main function to run the complete analysis pipeline"""
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
    
    # Step 1: Data Collection
    print("\n" + "="*80)
    print("STEP 1: DATA COLLECTION")
    print("="*80)
    
    try:
        df = collect_data(USERNAME)
        
        if df.empty:
            print("\n⚠️  No data collected. Please check your username and try again.")
            sys.exit(1)
            
        print(f"\n✅ Data collection complete! {len(df)} games collected.")
        
    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        sys.exit(1)
    
    # Step 2: Exploratory Data Analysis
    print("\n" + "="*80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    try:
        stats = run_eda(df)
        print(f"\n✅ EDA complete! Visualizations saved to visualizations/")
        
    except Exception as e:
        print(f"\n❌ Error during EDA: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Hypothesis Testing
    print("\n" + "="*80)
    print("STEP 3: HYPOTHESIS TESTING")
    print("="*80)
    
    try:
        results_df = run_hypothesis_tests(df)
        print(f"\n✅ Hypothesis testing complete!")
        
    except Exception as e:
        print(f"\n❌ Error during hypothesis testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    print("\n" + "█" * 80)
    print("█" + " ANALYSIS COMPLETE ".center(78, "█") + "█")
    print("█" * 80)
    print("\n📁 Output Files:")
    print(f"   • data/{USERNAME}_games.csv - Game data (CSV)")
    print(f"   • data/{USERNAME}_games_full.json - Full game data with PGN")
    print(f"   • data/hypothesis_test_results.csv - Statistical test results")
    print(f"   • visualizations/ - All generated plots")
    print("\n🎉 Thank you for using Chess.com Performance Analysis!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


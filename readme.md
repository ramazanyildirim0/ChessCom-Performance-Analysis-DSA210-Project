# Chess.com Performance Analysis - DSA 210 Project

## Project Proposal

### Motivation
As an active chess player with over 4,000 games played on Chess.com over the past 6 years, I want to analyze my personal chess data to understand patterns in my performance and identify factors that influence my game outcomes. This project will explore my playing style, rating progression, and decision-making patterns to gain insights that could help improve my chess skills.

### Dataset
I will be analyzing my personal chess game history from Chess.com, which includes:
- Over 4,000 games spanning 6 years
- Game metadata (date, time control, results, ratings)
- Move-by-move data in PGN format
- Opponent information
- Opening variations played
- Rating changes over time

The data will be collected from **Chess.com's Public API**, which provides free access to player game archives and statistics.

### Data Collection Plan

**API Endpoints:**
- `https://api.chess.com/pub/player/{username}` - Player profile information
- `https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}` - Monthly game archives
- `https://api.chess.com/pub/player/{username}/stats` - Player statistics

**Collection Process:**
1. Use Python's `requests` library to make API calls
2. Retrieve game archives month-by-month for the past 6 years (2019-2025)
3. Parse JSON responses and extract relevant game data
4. Use `python-chess` library to parse PGN data and extract move information
5. Store data in structured format (CSV/JSON) for analysis
6. Collect approximately 4,000+ games with the following fields per game:
   - Game URL and ID
   - Date and time played
   - Time control type (Bullet, Blitz, Rapid, etc.)
   - Player color (White/Black)
   - Opponent username and rating
   - Game result (Win/Loss/Draw)
   - Rating before and after game
   - Opening name and ECO code
   - Complete move sequence
   - Game termination reason (checkmate, resignation, timeout, etc.)

---

## Project Structure

```
ChessCom-Performance-Analysis-DSA210-Project/
├── main.py                 # Main script to run the entire pipeline
├── requirements.txt        # Python dependencies
├── readme.md              # Project documentation
├── src/
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings (update username here)
│   ├── data_collection.py # Chess.com API data collection
│   ├── eda.py             # Exploratory Data Analysis
│   └── hypothesis_tests.py # Statistical hypothesis testing
├── data/                  # Collected data files (generated)
│   ├── {username}_games.csv
│   ├── {username}_games_full.json
│   └── hypothesis_test_results.csv
└── visualizations/        # Generated plots (generated)
    ├── rating_over_time.png
    ├── rating_by_time_class.png
    ├── results_distribution.png
    ├── games_by_time.png
    ├── opening_analysis.png
    ├── rating_difference_analysis.png
    ├── termination_analysis.png
    ├── game_length_analysis.png
    └── heatmap_day_hour.png
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Username

Edit `src/config.py` and update the `USERNAME` variable with your Chess.com username:

```python
USERNAME = "your_chess_com_username"
```

### 3. Run the Analysis

```bash
python main.py
```

This will:
1. **Collect data** from Chess.com API (all your games from 2019-2025)
2. **Perform EDA** and generate visualizations
3. **Run hypothesis tests** and save results

---

## Analysis Components

### Data Collection (`src/data_collection.py`)
- Fetches player profile and statistics
- Retrieves all game archives via Chess.com API
- Parses PGN data for move information
- Saves data in CSV and JSON formats

### Exploratory Data Analysis (`src/eda.py`)
- **Summary Statistics**: Total games, win/loss/draw rates, rating ranges
- **Rating Progression**: Track rating changes over time by time control
- **Results Analysis**: Win rates by color, time control, and other factors
- **Time Patterns**: Games by hour, day of week, and month
- **Opening Analysis**: Most played openings and their success rates
- **Game Length Analysis**: Move count distributions and correlations

### Hypothesis Testing (`src/hypothesis_tests.py`)

10 statistical tests are conducted:

| Test | Hypothesis |
|------|------------|
| 1 | White vs Black win rate difference |
| 2 | Effect of rating difference on win probability |
| 3 | Win rate differences across time controls |
| 4 | Time of day effect on performance |
| 5 | Day of week effect on performance |
| 6 | Correlation between game length and winning |
| 7 | Rating progression over time |
| 8 | Opening choice effect on win rate |
| 9 | Overall win rate vs 50% |
| 10 | Momentum effect (winning streaks) |

---

## Visualizations Generated

1. **Rating Over Time** - Rating progression with rolling average
2. **Rating by Time Class** - Separate rating charts for bullet/blitz/rapid
3. **Results Distribution** - Pie charts and bar charts of win/loss/draw
4. **Games by Time** - Activity patterns by hour, day, and month
5. **Opening Analysis** - Most played openings and win rates
6. **Rating Difference Analysis** - Performance vs higher/lower rated opponents
7. **Termination Analysis** - How games end (checkmate, resignation, timeout)
8. **Game Length Analysis** - Move count distributions
9. **Activity Heatmap** - Games and win rates by day and hour

---

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib & seaborn** - Data visualization
- **scipy & statsmodels** - Statistical testing
- **python-chess** - PGN parsing
- **requests** - API calls
- **tqdm** - Progress bars

---

## Timeline

| Date | Milestone | Status |
|------|-----------|--------|
| Oct 31 | Project proposal submitted | ✅ Complete |
| Nov 28 | Data collection, EDA, hypothesis tests | ✅ Complete |
| Jan 02 | Apply ML methods | 🔄 Pending |
| Jan 09 | Final submission | 🔄 Pending |

---

## Author
Ramazan Yıldırım 32501
DSA 210 - Introduction to Data Science  
2025-2026 Fall Term

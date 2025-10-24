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

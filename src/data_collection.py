"""
Chess.com Data Collection Module
Collects game data from Chess.com Public API
"""

import requests
import json
import time
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import chess.pgn
import io

from config import (
    USERNAME, START_YEAR, END_YEAR, 
    DATA_DIR, API_BASE_URL, REQUEST_DELAY
)


class ChessComDataCollector:
    """Collects chess game data from Chess.com API"""
    
    def __init__(self, username: str):
        self.username = username
        self.base_url = API_BASE_URL
        self.headers = {
            'User-Agent': 'Chess Performance Analysis Project (Educational)'
        }
        
    def _make_request(self, endpoint: str) -> dict:
        """Make API request with rate limiting"""
        url = f"{self.base_url}/{endpoint}"
        time.sleep(REQUEST_DELAY)
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def get_player_profile(self) -> dict:
        """Get player profile information"""
        return self._make_request(f"player/{self.username}")
    
    def get_player_stats(self) -> dict:
        """Get player statistics"""
        return self._make_request(f"player/{self.username}/stats")
    
    def get_game_archives(self) -> list:
        """Get list of available game archive URLs"""
        data = self._make_request(f"player/{self.username}/games/archives")
        return data.get('archives', []) if data else []
    
    def get_monthly_games(self, year: int, month: int) -> list:
        """Get games from a specific month"""
        month_str = f"{month:02d}"
        data = self._make_request(f"player/{self.username}/games/{year}/{month_str}")
        return data.get('games', []) if data else []
    
    def parse_pgn(self, pgn_string: str) -> dict:
        """Parse PGN string to extract game information"""
        try:
            pgn_io = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                return {}
            
            headers = dict(game.headers)
            
            # Count moves
            moves = list(game.mainline_moves())
            move_count = len(moves)
            
            # Get opening moves (first 10 moves)
            opening_moves = ' '.join([str(m) for m in moves[:20]])
            
            # Extract opening name from ECOUrl (Chess.com specific)
            # Format: https://www.chess.com/openings/Kings-Pawn-Opening-1...e5
            eco_url = headers.get('ECOUrl', '')
            opening = headers.get('Opening', '')
            
            if not opening and eco_url:
                # Extract opening name from URL
                try:
                    opening_part = eco_url.split('/openings/')[-1]
                    # Replace hyphens with spaces and clean up
                    opening = opening_part.replace('-', ' ')
                    # Remove move notation at the end (e.g., "1...e5" or "2.Nf3")
                    import re
                    opening = re.sub(r'\s+\d+\.+\s*\w+.*$', '', opening)
                    opening = opening.strip()
                except:
                    opening = ''
            
            return {
                'event': headers.get('Event', ''),
                'site': headers.get('Site', ''),
                'date': headers.get('Date', ''),
                'white': headers.get('White', ''),
                'black': headers.get('Black', ''),
                'result': headers.get('Result', ''),
                'white_elo': headers.get('WhiteElo', ''),
                'black_elo': headers.get('BlackElo', ''),
                'time_control': headers.get('TimeControl', ''),
                'eco': headers.get('ECO', ''),
                'opening': opening,
                'termination': headers.get('Termination', ''),
                'move_count': move_count,
                'opening_moves': opening_moves
            }
        except Exception as e:
            print(f"Error parsing PGN: {e}")
            return {}
    
    def process_game(self, game_data: dict) -> dict:
        """Process a single game and extract relevant information"""
        # Determine if user played as white or black
        white_username = game_data.get('white', {}).get('username', '').lower()
        black_username = game_data.get('black', {}).get('username', '').lower()
        
        is_white = white_username == self.username.lower()
        
        # Get player and opponent info
        if is_white:
            player_info = game_data.get('white', {})
            opponent_info = game_data.get('black', {})
        else:
            player_info = game_data.get('black', {})
            opponent_info = game_data.get('white', {})
        
        # Determine result from player's perspective
        player_result = player_info.get('result', '')
        
        if player_result == 'win':
            result = 'Win'
        elif player_result in ['checkmated', 'timeout', 'resigned', 'lose', 'abandoned']:
            result = 'Loss'
        else:
            result = 'Draw'
        
        # Parse PGN for additional info
        pgn_data = self.parse_pgn(game_data.get('pgn', ''))
        
        # Extract time control type
        time_control = game_data.get('time_control', '')
        time_class = game_data.get('time_class', '')
        
        # Convert timestamp to datetime
        end_time = game_data.get('end_time', 0)
        game_datetime = datetime.fromtimestamp(end_time) if end_time else None
        
        return {
            'game_url': game_data.get('url', ''),
            'game_id': game_data.get('uuid', ''),
            'date': game_datetime.strftime('%Y-%m-%d') if game_datetime else '',
            'time': game_datetime.strftime('%H:%M:%S') if game_datetime else '',
            'day_of_week': game_datetime.strftime('%A') if game_datetime else '',
            'hour': game_datetime.hour if game_datetime else None,
            'time_control': time_control,
            'time_class': time_class,
            'rated': game_data.get('rated', False),
            'player_color': 'White' if is_white else 'Black',
            'player_username': player_info.get('username', ''),
            'player_rating': player_info.get('rating', 0),
            'opponent_username': opponent_info.get('username', ''),
            'opponent_rating': opponent_info.get('rating', 0),
            'rating_diff': player_info.get('rating', 0) - opponent_info.get('rating', 0),
            'result': result,
            'player_result_detail': player_result,
            'opponent_result_detail': opponent_info.get('result', ''),
            'eco': pgn_data.get('eco', ''),
            'opening': pgn_data.get('opening', ''),
            'termination': pgn_data.get('termination', ''),
            'move_count': pgn_data.get('move_count', 0),
            'opening_moves': pgn_data.get('opening_moves', ''),
            'pgn': game_data.get('pgn', '')
        }
    
    def collect_all_games(self, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
        """Collect all games within the specified year range"""
        all_games = []
        
        print(f"\n{'='*60}")
        print(f"Collecting games for user: {self.username}")
        print(f"Date range: {start_year} - {end_year}")
        print(f"{'='*60}\n")
        
        # Get all archive URLs
        archives = self.get_game_archives()
        
        if not archives:
            print("No game archives found. Please check the username.")
            return pd.DataFrame()
        
        # Filter archives by year range
        filtered_archives = []
        for archive_url in archives:
            # Extract year from URL (format: .../YYYY/MM)
            parts = archive_url.split('/')
            year = int(parts[-2])
            if start_year <= year <= end_year:
                filtered_archives.append(archive_url)
        
        print(f"Found {len(filtered_archives)} monthly archives to process\n")
        
        # Collect games from each month
        for archive_url in tqdm(filtered_archives, desc="Fetching monthly archives"):
            parts = archive_url.split('/')
            year, month = int(parts[-2]), int(parts[-1])
            
            games = self.get_monthly_games(year, month)
            
            for game in games:
                processed = self.process_game(game)
                if processed:
                    all_games.append(processed)
        
        print(f"\nTotal games collected: {len(all_games)}")
        
        # Create DataFrame
        df = pd.DataFrame(all_games)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = None):
        """Save collected data to CSV and JSON"""
        if filename is None:
            filename = f"{self.username}_games"
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save as CSV (without PGN for smaller file size)
        csv_df = df.drop(columns=['pgn', 'opening_moves'], errors='ignore')
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")
        
        # Save full data as JSON
        json_path = os.path.join(DATA_DIR, f"{filename}_full.json")
        df.to_json(json_path, orient='records', date_format='iso')
        print(f"Saved JSON to: {json_path}")
        
        return csv_path, json_path


def collect_data(username: str = None) -> pd.DataFrame:
    """Main function to collect chess.com data"""
    if username is None:
        username = USERNAME
    
    if username == "your_username_here":
        raise ValueError(
            "Please update USERNAME in src/config.py with your Chess.com username"
        )
    
    collector = ChessComDataCollector(username)
    
    # Get and display player profile
    profile = collector.get_player_profile()
    if profile:
        print(f"\nPlayer Profile:")
        print(f"  Username: {profile.get('username', 'N/A')}")
        print(f"  Joined: {datetime.fromtimestamp(profile.get('joined', 0)).strftime('%Y-%m-%d')}")
        print(f"  Country: {profile.get('country', 'N/A').split('/')[-1]}")
        print(f"  Followers: {profile.get('followers', 0)}")
    
    # Get and display player stats
    stats = collector.get_player_stats()
    if stats:
        print(f"\nCurrent Ratings:")
        for time_class in ['chess_rapid', 'chess_blitz', 'chess_bullet']:
            if time_class in stats:
                rating_data = stats[time_class].get('last', {})
                print(f"  {time_class.replace('chess_', '').title()}: {rating_data.get('rating', 'N/A')}")
    
    # Collect all games
    df = collector.collect_all_games()
    
    if not df.empty:
        # Save the data
        collector.save_data(df)
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("DATA COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total games: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nGames by time class:")
        print(df['time_class'].value_counts().to_string())
        print(f"\nResults distribution:")
        print(df['result'].value_counts().to_string())
        print(f"\nGames by color:")
        print(df['player_color'].value_counts().to_string())
    
    return df


if __name__ == "__main__":
    df = collect_data()


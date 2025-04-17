import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def read_data(file_path="combined_atp_matches.csv"):
    """Read the combined ATP matches data.
    
    Args:
        file_path: Path to the combined CSV file
        
    Returns:
        pandas.DataFrame: The matches data as a DataFrame
    """
    logging.info(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Run preprocess.py first.")
    
    df = pd.read_csv(file_path)
    
    # Ensure tourney_date is in datetime format
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    
    # Sort by date to ensure chronological order
    df = df.sort_values(by="tourney_date")
    
    logging.info(f"Data shape: {df.shape}")
    return df

def create_player_history_df(df):
    """Create a DataFrame with player match history.
    
    Args:
        df: DataFrame containing match data
        
    Returns:
        dict: Dictionary mapping player IDs to their match history
    """
    player_history = {}
    
    # Process the matches in chronological order
    for _, match in tqdm(df.iterrows(), total=len(df), desc="Building player history"):
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        match_date = match['tourney_date']
        
        # Initialize player history if not exists
        if winner_id not in player_history:
            player_history[winner_id] = []
        if loser_id not in player_history:
            player_history[loser_id] = []
        
        # Add match to winner's history (as winner)
        winner_history = {
            'player_id': winner_id,
            'opponent_id': loser_id,
            'date': match_date,
            'result': 'win'
        }
        
        # Add match to loser's history (as loser)
        loser_history = {
            'player_id': loser_id,
            'opponent_id': winner_id,
            'date': match_date,
            'result': 'loss'
        }
        
        # Add all relevant stats
        for col in df.columns:
            if col.startswith('w_'):
                stat_name = col[2:]  # Remove 'w_' prefix
                
                # Get the raw values
                w_value = match[col]
                l_value = match[f'l_{stat_name}']
                
                # Standardize stats
                if stat_name in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced']:
                    # Standardize by service games
                    w_svgms = match['w_SvGms'] if pd.notna(match['w_SvGms']) and match['w_SvGms'] > 0 else np.nan
                    l_svgms = match['l_SvGms'] if pd.notna(match['l_SvGms']) and match['l_SvGms'] > 0 else np.nan
                    
                    if pd.notna(w_value) and pd.notna(w_svgms):
                        winner_history[f'{stat_name}_per_svgm'] = w_value / w_svgms
                    
                    if pd.notna(l_value) and pd.notna(l_svgms):
                        loser_history[f'{stat_name}_per_svgm'] = l_value / l_svgms
                    
                    # # Also keep the raw values
                    # winner_history[stat_name] = w_value
                    # loser_history[stat_name] = l_value
                    
                    if stat_name == 'bpSaved':
                        # standardize by break points faced
                        w_bpfaced = match['w_bpFaced'] if pd.notna(match['w_bpFaced']) and match['w_bpFaced'] > 0 else np.nan
                        l_bpfaced = match['l_bpFaced'] if pd.notna(match['l_bpFaced']) and match['l_bpFaced'] > 0 else np.nan
                        
                        if pd.notna(w_value) and pd.notna(w_bpfaced):
                            winner_history['bpSaved_per_bpFaced_per_svgm'] = w_value / w_bpfaced / w_svgms
                        
                        if pd.notna(l_value) and pd.notna(l_bpfaced):
                            loser_history['bpSaved_per_bpFaced_per_svgm'] = l_value / l_bpfaced / l_svgms
                    
                    # # Also keep the raw values
                    # winner_history[stat_name] = w_value
                    # loser_history[stat_name] = l_value
                    
                else:
                    # Keep other stats as is
                    winner_history[stat_name] = w_value
                    loser_history[stat_name] = l_value
        
        player_history[winner_id].append(winner_history)
        player_history[loser_id].append(loser_history)
    
    logging.info(f"Created history for {len(player_history)} players")
    return player_history

def calculate_rolling_stats(player_history, window=10):
    """Calculate rolling statistics for each player based on their past matches.
    
    Args:
        player_history: Dictionary mapping player IDs to their match history
        window: Number of past matches to consider for rolling statistics
        
    Returns:
        dict: Dictionary mapping player IDs to their rolling statistics
    """
    player_rolling_stats = {}
    
    for player_id, matches in tqdm(player_history.items(), desc="Calculating rolling stats"):
        # Sort matches by date
        sorted_matches = sorted(matches, key=lambda x: x['date'])
        
        # Initialize player rolling stats
        player_rolling_stats[player_id] = {}
        
        # Calculate rolling stats for each match
        for i, match in enumerate(sorted_matches):
            match_date = match['date']
            
            # Get past matches within the window
            start_idx = max(0, i - window)
            past_matches = sorted_matches[start_idx:i]
            
            # Skip if no past matches
            if not past_matches:
                continue
            
            # Calculate rolling stats
            rolling_stats = {}
            
            # Get all possible stat keys from the first match
            stat_keys = [k for k in sorted_matches[0].keys() 
                         if k not in ['player_id', 'opponent_id', 'date', 'result']]
            
            for stat in stat_keys:
                try:
                    values = [m[stat] for m in past_matches if stat in m and pd.notna(m[stat])]
                    rolling_stats[f'rolling_{window}_{stat}'] = np.mean(values) if values else np.nan
                except KeyError:
                    # Skip stats that don't exist in some matches
                    continue
            
            # Store rolling stats by date
            if match_date not in player_rolling_stats[player_id]:
                player_rolling_stats[player_id][match_date] = {}
            
            player_rolling_stats[player_id][match_date] = rolling_stats
    
    logging.info(f"Calculated rolling stats for {len(player_rolling_stats)} players")
    return player_rolling_stats

def add_rolling_features(df, player_rolling_stats, window=10):
    """Add rolling features to the matches DataFrame.
    
    Args:
        df: DataFrame containing match data
        player_rolling_stats: Dictionary mapping player IDs to their rolling statistics
        window: Window size used for rolling statistics
        
    Returns:
        pandas.DataFrame: DataFrame with added rolling features
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_features = df.copy()
    
    # Process each match
    for idx, match in tqdm(df.iterrows(), total=len(df), desc="Adding rolling features"):
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        match_date = match['tourney_date']
        
        # Get rolling stats for winner and loser before this match
        if winner_id in player_rolling_stats and match_date in player_rolling_stats[winner_id]:
            winner_stats = player_rolling_stats[winner_id][match_date]
            for stat, value in winner_stats.items():
                df_features.loc[idx, f'w_{stat}'] = value
        
        if loser_id in player_rolling_stats and match_date in player_rolling_stats[loser_id]:
            loser_stats = player_rolling_stats[loser_id][match_date]
            for stat, value in loser_stats.items():
                df_features.loc[idx, f'l_{stat}'] = value
    
    # Drop rows with missing rolling features
    logging.info(f"Data shape before dropping rows with missing rolling features: {df_features.shape}")
    rolling_cols = [col for col in df_features.columns if 'rolling' in col]
    df_features = df_features.dropna(subset=rolling_cols)
    logging.info(f"Data shape after dropping rows with missing rolling features: {df_features.shape}")
    
    return df_features

def add_head_to_head_features(df):
    """Add head-to-head record features for each matchup.
    
    Args:
        df: DataFrame containing match data
        
    Returns:
        pandas.DataFrame: DataFrame with added head-to-head features
    """
    logging.info("Adding head-to-head record features")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_features = df.copy()
    
    # Initialize head-to-head record columns
    df_features['w_previous_wins'] = 0
    df_features['l_previous_wins'] = 0
    
    # Dictionary to track head-to-head records
    h2h_records = {}  # Format: (player1_id, player2_id) -> [player1_wins, player2_wins]
    
    # Process matches in chronological order
    for idx, match in tqdm(df.iterrows(), total=len(df), desc="Adding head-to-head features"):
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        
        # Create a unique key for this matchup (always put the smaller ID first for consistency)
        if winner_id < loser_id:
            matchup_key = (winner_id, loser_id)
            is_winner_first = True
        else:
            matchup_key = (loser_id, winner_id)
            is_winner_first = False
        
        # Get current head-to-head record
        if matchup_key in h2h_records:
            record = h2h_records[matchup_key]
            
            # Assign previous wins to the correct players
            if is_winner_first:
                df_features.loc[idx, 'w_previous_wins'] = record[0]
                df_features.loc[idx, 'l_previous_wins'] = record[1]
            else:
                df_features.loc[idx, 'w_previous_wins'] = record[1]
                df_features.loc[idx, 'l_previous_wins'] = record[0]
            
            # Update the record after the match
            if is_winner_first:
                record[0] += 1  # Winner won and is first in the key
            else:
                record[1] += 1  # Winner won and is second in the key
        else:
            # First meeting between these players
            df_features.loc[idx, 'w_previous_wins'] = 0
            df_features.loc[idx, 'l_previous_wins'] = 0
            
            # Initialize the record
            if is_winner_first:
                h2h_records[matchup_key] = [1, 0]  # Winner is first in the key
            else:
                h2h_records[matchup_key] = [0, 1]  # Winner is second in the key
    
    logging.info(f"Added head-to-head records for {len(h2h_records)} matchups")
    return df_features

def main():
    logging.info("Starting feature creation script")
    
    # Read the combined data
    df = read_data()
    
    # Create player history DataFrame
    player_history = create_player_history_df(df)
    
    # Calculate rolling statistics
    window = 10  # Use past 10 matches
    player_rolling_stats = calculate_rolling_stats(player_history, window)
    
    # Add rolling features to the matches DataFrame
    df_features = add_rolling_features(df, player_rolling_stats, window)
    
    # Add head-to-head features
    df_features = add_head_to_head_features(df_features)
    
    # Write the features to a new CSV
    output_file = "atp_matches_with_features.csv"
    logging.info(f"Writing features to {output_file}")
    df_features.to_csv(output_file, index=False)
    logging.info(f"Features successfully written to {output_file}")

if __name__ == "__main__":
    main()
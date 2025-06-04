import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from opponent_adjustments import get_opponent_adjustments
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
import math
from tqdm import tqdm
import optuna


# Mount with Colab
def mount_with_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    # Define the path to your desired directory in Google Drive
    drive_path = '/content/drive/MyDrive/Betting/BettingModels'

    # Change the current working directory to the desired location
    os.chdir(drive_path)

    # Verify the current working directory (optional)
    print(f"Current working directory: {os.getcwd()}")

# Load Games Full Data
def load_games_data(DB_PATH):
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Load games_full data - Select ALL potentially relevant columns
    # Explicitly listing columns is generally better than SELECT *
    # Make sure this list matches the columns in your 'games_full' table
    # (Derived from your initial dataset_head.csv)
    all_feature_columns = [
        'id', 'season', 'week', 'season_type', 'completed', 'neutral_site',
        'conference_game', 'attendance', 'home_team', 'home_conference',
        'home_division', 'home_points', 'home_post_win_prob', 'home_pregame_elo', # Note: We use our CALC'd elo later
        'home_postgame_elo', 'away_team', 'away_conference', 'away_division',
        'away_points', 'away_post_win_prob', 'away_pregame_elo', # Note: We use our CALC'd elo later
        'away_postgame_elo', 'avg_closing_spread', 'avg_closing_total',
        'avg_opening_spread', 'avg_opening_total',
        # Home Offense Stats
        'home_offense_plays', 'home_offense_drives', 'home_offense_ppa',
        'home_offense_totalPPA', 'home_offense_successRate', 'home_offense_explosiveness',
        'home_offense_powerSuccess', 'home_offense_stuffRate', 'home_offense_lineYards',
        'home_offense_lineYardsTotal', 'home_offense_secondLevelYards',
        'home_offense_secondLevelYardsTotal', 'home_offense_openFieldYards',
        'home_offense_openFieldYardsTotal', 'home_offense_standardDowns_ppa',
        'home_offense_standardDowns_successRate', 'home_offense_standardDowns_explosiveness',
        'home_offense_passingDowns_ppa', 'home_offense_passingDowns_successRate',
        'home_offense_passingDowns_explosiveness', 'home_offense_rushingPlays_ppa',
        'home_offense_rushingPlays_totalPPA', 'home_offense_rushingPlays_successRate',
        'home_offense_rushingPlays_explosiveness', 'home_offense_passingPlays_ppa',
        'home_offense_passingPlays_totalPPA', 'home_offense_passingPlays_successRate',
        'home_offense_passingPlays_explosiveness',
        # Home Defense Stats
        'home_defense_plays', 'home_defense_drives', 'home_defense_ppa',
        'home_defense_totalPPA', 'home_defense_successRate', 'home_defense_explosiveness',
        'home_defense_powerSuccess', 'home_defense_stuffRate', 'home_defense_lineYards',
        'home_defense_lineYardsTotal', 'home_defense_secondLevelYards',
        'home_defense_secondLevelYardsTotal', 'home_defense_openFieldYards',
        'home_defense_openFieldYardsTotal', 'home_defense_standardDowns_ppa',
        'home_defense_standardDowns_successRate', 'home_defense_standardDowns_explosiveness',
        'home_defense_passingDowns_ppa', 'home_defense_passingDowns_successRate',
        'home_defense_passingDowns_explosiveness', 'home_defense_rushingPlays_ppa',
        'home_defense_rushingPlays_totalPPA', 'home_defense_rushingPlays_successRate',
        'home_defense_rushingPlays_explosiveness', 'home_defense_passingPlays_ppa',
        'home_defense_passingPlays_totalPPA', 'home_defense_passingPlays_successRate',
        'home_defense_passingPlays_explosiveness',
        # Away Offense Stats (matches home offense structure)
        'away_offense_plays', 'away_offense_drives', 'away_offense_ppa',
        'away_offense_totalPPA', 'away_offense_successRate', 'away_offense_explosiveness',
        'away_offense_powerSuccess', 'away_offense_stuffRate', 'away_offense_lineYards',
        'away_offense_lineYardsTotal', 'away_offense_secondLevelYards',
        'away_offense_secondLevelYardsTotal', 'away_offense_openFieldYards',
        'away_offense_openFieldYardsTotal', 'away_offense_standardDowns_ppa',
        'away_offense_standardDowns_successRate', 'away_offense_standardDowns_explosiveness',
        'away_offense_passingDowns_ppa', 'away_offense_passingDowns_successRate',
        'away_offense_passingDowns_explosiveness', 'away_offense_rushingPlays_ppa',
        'away_offense_rushingPlays_totalPPA', 'away_offense_rushingPlays_successRate',
        'away_offense_rushingPlays_explosiveness', 'away_offense_passingPlays_ppa',
        'away_offense_passingPlays_totalPPA', 'away_offense_passingPlays_successRate',
        'away_offense_passingPlays_explosiveness',
        # Away Defense Stats (matches home defense structure)
        'away_defense_plays', 'away_defense_drives', 'away_defense_ppa',
        'away_defense_totalPPA', 'away_defense_successRate', 'away_defense_explosiveness',
        'away_defense_powerSuccess', 'away_defense_stuffRate', 'away_defense_lineYards',
        'away_defense_lineYardsTotal', 'away_defense_secondLevelYards',
        'away_defense_secondLevelYardsTotal', 'away_defense_openFieldYards',
        'away_defense_openFieldYardsTotal', 'away_defense_standardDowns_ppa',
        'away_defense_standardDowns_successRate', 'away_defense_standardDowns_explosiveness',
        'away_defense_passingDowns_ppa', 'away_defense_passingDowns_successRate',
        'away_defense_passingDowns_explosiveness', 'away_defense_rushingPlays_ppa',
        'away_defense_rushingPlays_totalPPA', 'away_defense_rushingPlays_successRate',
        'away_defense_rushingPlays_explosiveness', 'away_defense_passingPlays_ppa',
        'away_defense_passingPlays_totalPPA', 'away_defense_passingPlays_successRate',
        'away_defense_passingPlays_explosiveness',
        # Other stats
        'home_turnovers', 'home_possessionTime', 'away_turnovers', 'away_possessionTime'
    ]

    # Construct the SQL query string dynamically
    select_clause = ",\n    ".join([f"g.{col}" for col in all_feature_columns])
    games_query = f"""
    SELECT
        {select_clause}
    FROM
        games_full g
    WHERE
        g.completed = 1 -- Only use completed games
    ORDER BY
        g.season, g.week, g.id;
    """
    # print(games_query) # Optional: Print the generated query to verify

    print("Loading ALL games data (including advanced stats)...")
    games_df = pd.read_sql_query(games_query, conn)
    print(f"Loaded {len(games_df)} completed games with {len(games_df.columns)} columns.")
    conn.close()
    print("Database connection closed.")
    return games_df

def load_returning_prod_data(DB_PATH, RP_METRICS_TO_USE):
    conn = sqlite3.connect(DB_PATH)
    rp_cols_select = ['season', 'team'] + RP_METRICS_TO_USE
    rp_cols_str = ", ".join(rp_cols_select)
    rp_query = f"SELECT {rp_cols_str} FROM returning_production;"
    print(f"Loading returning production data ({RP_METRICS_TO_USE})...")
    rp_df = pd.read_sql_query(rp_query, conn)
    print(f"Loaded {len(rp_df)} returning production records.")
    conn.close()
    print("Database connection closed.")
    return rp_df

# Pre-process Games Data
def preprocess_games_data(games_df):
    # Convert boolean-like columns to integers (0 or 1) for models
    games_df['neutral_site'] = games_df['neutral_site'].astype(int)
    games_df['conference_game'] = games_df['conference_game'].astype(int)

    # Convert spread/total/score columns to numeric, coercing errors to NaN
    numeric_cols = ['avg_closing_spread', 'avg_closing_total', 'avg_opening_spread',
                    'avg_opening_total', 'home_points', 'away_points',
                    'attendance', 'home_possessionTime', 'away_possessionTime',
                    # Add Elo/win prob if needed, though we'll use our own Elo primarily
                    'home_post_win_prob', 'home_pregame_elo', 'home_postgame_elo',
                    'away_post_win_prob', 'away_pregame_elo', 'away_postgame_elo']

    # Convert all advanced stat columns to numeric
    # Identify the first advanced stat column to loop from there
    first_adv_stat_col = 'home_offense_plays'
    first_adv_stat_idx = games_df.columns.get_loc(first_adv_stat_col)
    adv_stat_cols = games_df.columns[first_adv_stat_idx:]

    numeric_cols.extend(adv_stat_cols)

    print("Converting relevant columns to numeric...")
    for col in numeric_cols:
        if col in games_df.columns: # Check if column exists (robustness)
            games_df[col] = pd.to_numeric(games_df[col], errors='coerce')

    # Report missing values for key targets/inputs after conversion
    check_missing_cols = ['avg_closing_spread', 'home_points', 'away_points']
    print("Missing value check (post-numeric conversion):")
    for col in check_missing_cols:
        if col in games_df.columns:
            missing_pct = games_df[col].isnull().mean() * 100
            print(f"  Column '{col}' missing: {missing_pct:.2f}%")

    # Drop rows where essential score data might be missing after conversion
    games_df.dropna(subset=['home_points', 'away_points'], inplace=True)
    return games_df

# Preprocess Returning Production Data
def preprocess_returning_prod_data(rp_df, RP_METRICS_TO_USE, DEFAULT_RP_VALUE):
    print("Preprocessing returning production data...")
    # Convert RP metrics to numeric
    for col in RP_METRICS_TO_USE:
        rp_df[col] = pd.to_numeric(rp_df[col], errors='coerce')
    # Fill missing RP values *before* merging
    print(f"Filling NaNs in RP data with default: {DEFAULT_RP_VALUE}")
    rp_df.fillna({col: DEFAULT_RP_VALUE for col in RP_METRICS_TO_USE}, inplace=True)
    # Prepare for merge - RP data for season S applies to season S games
    rp_df.rename(columns={'season': 'rp_season'}, inplace=True) # Avoid collision with game season
    return rp_df

# Load Pre-Calculated ELO Ratings
def load_ELO_ratings(PRE_GAME_ELO_CSV_PATH):
    print(f"Loading pre-game Elo ratings from: {PRE_GAME_ELO_CSV_PATH}")
    pre_game_elo_df = pd.read_csv(PRE_GAME_ELO_CSV_PATH)
    pre_game_elo_df = pre_game_elo_df[['game_id', 'home_pregame_elo', 'away_pregame_elo']]
    print(f"Loaded Elo ratings for {len(pre_game_elo_df)} games.")

    # Rename columns to avoid conflict with original Elo cols and clarify source
    pre_game_elo_df.rename(columns={
        'home_pregame_elo': 'home_pregame_elo_calc',
        'away_pregame_elo': 'away_pregame_elo_calc'
    }, inplace=True)
    print(f"Loaded pre-game Elo ratings.")
    return pre_game_elo_df

# Merge Games and ELO Data
def merge_elo_to_games(games_df, pre_game_elo_df):
    print("Merging games data with pre-game Elo ratings...")
    master_df = pd.merge(
        games_df,
        pre_game_elo_df,
        left_on='id',
        right_on='game_id',
        how='left'
    )

    # Check for games potentially missed by the merge
    missing_elo_count = master_df['home_pregame_elo_calc'].isnull().sum()
    if missing_elo_count > 0:
        print(f"Warning: {missing_elo_count} games are missing calculated pre-game Elo ratings after merge.")
        # Depending on strategy, might drop these rows later if calc'd Elo is crucial
        # master_df.dropna(subset=['home_pregame_elo_calc', 'away_pregame_elo_calc'], inplace=True)

    master_df.drop(columns=['game_id'], inplace=True)
    return master_df

def merge_returning_production_to_games(master_df, rp_df, RP_METRICS_TO_USE, DEFAULT_RP_VALUE):
    print("Merging returning production data...")
    # Merge for Home Team
    master_df = pd.merge(
        master_df,
        rp_df,
        left_on=['season', 'home_team'],
        right_on=['rp_season', 'team'],
        how='left',
        suffixes=('', '_rp_home_temp')
    )
    # Rename home RP columns and drop temporary merge keys
    home_rp_rename = {col: f'home_rp_{col}' for col in RP_METRICS_TO_USE}
    master_df.rename(columns=home_rp_rename, inplace=True)
    master_df.drop(columns=['rp_season', 'team'], inplace=True, errors='ignore')

    # Merge for Away Team
    master_df = pd.merge(
        master_df,
        rp_df,
        left_on=['season', 'away_team'],
        right_on=['rp_season', 'team'],
        how='left',
        suffixes=('', '_rp_away_temp')
    )
    # Rename away RP columns and drop temporary merge keys
    away_rp_rename = {col: f'away_rp_{col}' for col in RP_METRICS_TO_USE}
    master_df.rename(columns=away_rp_rename, inplace=True)
    master_df.drop(columns=['rp_season', 'team'], inplace=True, errors='ignore')

    # Fill NaNs for teams potentially missing in RP data *after* merge (use default)
    home_rp_cols = list(home_rp_rename.values())
    away_rp_cols = list(away_rp_rename.values())
    rp_cols_merged = home_rp_cols + away_rp_cols
    for col in rp_cols_merged:
        if master_df[col].isnull().any():
            print(f"Filling NaNs in merged RP column '{col}' with {DEFAULT_RP_VALUE}")
            master_df[col].fillna(DEFAULT_RP_VALUE, inplace=True)
    return master_df

# Add Opponent Adjustments and Merge
def add_opponent_adjustments(master_df):    
    opponent_adjustment_df = get_opponent_adjustments(master_df)
    master_df = pd.merge(master_df, opponent_adjustment_df, on=['season', 'week', 'home_team', 'away_team'], how='left', suffixes=("","_y"))
    cols_to_drop = [col for col in master_df.columns if col.endswith('_y')]
    if cols_to_drop:
        print(f"Dropping potentially duplicated columns: {cols_to_drop}")
        master_df.drop(columns=cols_to_drop, inplace=True)
    return master_df

def drop_missing_target_sort_chronologically(master_df):
    print("Sorting final DataFrame chronologically...")
    master_df.sort_values(by=['season', 'week', 'id'], inplace=True)
    master_df.reset_index(drop=True, inplace=True)
    return master_df

# Inspect Consolidated Data... Do I need this?

# Define Target Variable and Basic Features
def define_target_variable_basic_features(master_df):
    # Define the primary target variable
    target_variable = 'avg_closing_spread'
    print(f"\nTarget Variable: '{target_variable}'")

    # Check target missing values
    target_missing_pct = master_df[target_variable].isnull().mean() * 100
    print(f"Missing values in target ('{target_variable}'): {target_missing_pct:.2f}%")

    # Drop rows where the target variable is NaN
    master_df.dropna(subset=[target_variable], inplace=True)
    master_df.reset_index(drop=True, inplace=True)

    # Verify the target column now has no NaNs
    print(f"Missing values in target after dropping: {master_df[target_variable].isnull().sum()}")

    # Define initial, basic features (using our CALCULATED Elo)
    # Note: We exclude the original home/away_pregame_elo from the DB unless needed for comparison
    basic_features = [
        'home_pregame_elo_calc', # Our calculated Elo
        'away_pregame_elo_calc', # Our calculated Elo
        'neutral_site',
        'conference_game',
        'season',
        'week'
    ]
    # Engineer Elo difference feature using our calculated Elo
    master_df['elo_diff_calc'] = master_df['home_pregame_elo_calc'] - master_df['away_pregame_elo_calc']
    basic_features.append('elo_diff_calc')

    print(f"\nBasic Features Selected ({len(basic_features)}):")
    print(basic_features)
    return target_variable, basic_features, master_df

def identify_stats_to_roll(ewma_span):
    # Select key efficiency/explosiveness metrics. Start with a focused list.
    # Expand this list later if needed.
    stats_to_roll = [
        # Overall Offense
        'offense_ppa',
        'offense_successRate',
        'offense_explosiveness',
        # Rushing Offense
        'offense_rushingPlays_ppa',
        'offense_rushingPlays_successRate',
        'offense_rushingPlays_explosiveness',
        'offense_lineYards', # Potentially useful offensive line proxy
        # Passing Offense
        'offense_passingPlays_ppa',
        'offense_passingPlays_successRate',
        'offense_passingPlays_explosiveness',
        # Standard Downs
        'offense_standardDowns_ppa',
        'offense_standardDowns_successRate',
        'offense_standardDowns_explosiveness',
        # Passing Downs
        'offense_passingDowns_ppa',
        'offense_passingDowns_successRate',
        'offense_passingDowns_explosiveness',
        # Overall Defense (using opponent's offensive stats)
        'defense_ppa',
        'defense_successRate',
        'defense_explosiveness',
        # Rushing Defense
        'defense_rushingPlays_ppa',
        'defense_rushingPlays_successRate',
        'defense_rushingPlays_explosiveness',
        'defense_lineYards', # Potentially useful defensive line proxy
        # Passing Defense
        'defense_passingPlays_ppa',
        'defense_passingPlays_successRate',
        'defense_passingPlays_explosiveness',
        # Standard Downs
        'defense_standardDowns_ppa',
        'defense_standardDowns_successRate',
        'defense_standardDowns_explosiveness',
        # Passing Downs
        'defense_passingDowns_ppa',
        'defense_passingDowns_successRate',
        'defense_passingDowns_explosiveness',
        # Special Teams / Other (Add if desired, e.g., avg starting field position - needs raw data)
        'turnovers' # Average turnovers forced/committed
    ]

    # Define EWMA span (adjust as needed, smaller span = more weight on recent)
    # A span of 5 roughly means the last ~5 games have the most influence.


    print(f"Selected {len(stats_to_roll)} stats for EWMA (span={ewma_span}).")
    return stats_to_roll

# Reshape data to team-centric format
def reshape_to_team_centric(master_df, stats_to_roll):
    # Create two temporary dataframes, one for home team stats, one for away
    home_stats = master_df[['id', 'season', 'week', 'home_team', 'away_team']].copy()
    away_stats = master_df[['id', 'season', 'week', 'away_team', 'home_team']].copy()

    home_stats.rename(columns={'home_team': 'team', 'away_team': 'opponent'}, inplace=True)
    away_stats.rename(columns={'away_team': 'team', 'home_team': 'opponent'}, inplace=True)

    # Add actual stats, renaming columns to generic 'offense_*', 'defense_*'
    print("Reshaping data to team-centric format...")
    for stat_base in stats_to_roll:
        # Determine if it's an offense or defense stat based on original column name structure
        # This requires stats_to_roll names to match the generic part after home_/away_
        home_col = f'adj_hybrid_home_{stat_base}'
        away_col = f'adj_hybrid_away_{stat_base}'

        if home_col in master_df.columns and away_col in master_df.columns and stat_base != 'turnovers':
            # Offensive and Defensive stat for the teams
            home_stats[stat_base] = master_df[home_col]
            away_stats[stat_base] = master_df[away_col]

        # Handle turnovers specifically if included
        elif stat_base == 'turnovers':
            home_stats['turnovers_committed'] = master_df['home_turnovers']
            home_stats['turnovers_forced'] = master_df['away_turnovers'] # Home defense forced away turnovers
            away_stats['turnovers_committed'] = master_df['away_turnovers']
            away_stats['turnovers_forced'] = master_df['home_turnovers'] # Away defense forced home turnovers


    # Combine home and away views
    team_game_df = pd.concat([home_stats, away_stats], ignore_index=True)
    # Critical: Sort for Rolling Calculation
    print("Sorting team-centric data...")
    team_game_df.sort_values(by=['team', 'season', 'week', 'id'], inplace=True)
    return team_game_df

# Calculate Lagged EWMA
def calculate_lagged_ewma(team_game_df, stats_to_roll, ewma_span, min_periods_for_ewma):
    print(f"Calculating SEASONAL lagged EWMAs (span={ewma_span})...")
    ewma_cols_generated = []
    stats_to_roll.append('turnovers_committed')
    stats_to_roll.append('turnovers_forced')
    stats_to_roll.remove('turnovers')
    for stat in stats_to_roll:
        if stat in team_game_df.columns: # Ensure stat column was created successfully
            ewma_col_name = f'{stat}_ewma_lag1'
            # Calculate EWMA and shift within each group
            # Use transform for efficiency if possible
            team_game_df[ewma_col_name] = team_game_df.groupby('team')[stat].transform(
                lambda x: x.ewm(span=ewma_span, min_periods=min_periods_for_ewma, adjust=True).mean().shift(1)
            )
            ewma_cols_generated.append(ewma_col_name)
        else:
            print(f"Skipping EWMA for '{stat}' as column not found in team_game_df.")

    print(f"Generated {len(ewma_cols_generated)} EWMA columns.")
    return team_game_df, ewma_cols_generated

# Merge back to Master DF
def merge_ewma_to_master_df(master_df, team_game_df, ewma_cols_generated):
    print("Merging EWMA features back to master DataFrame...")

    # Select only necessary columns from team_game_df for merging
    ewma_features_to_merge = team_game_df[['id', 'team'] + ewma_cols_generated].copy()

    # Merge for Home Team stats
    master_df_merged = pd.merge(
        master_df,
        ewma_features_to_merge,
        left_on=['id', 'home_team'],
        right_on=['id', 'team'],
        how='left',
        suffixes=('', '_y') # Avoid suffix collision initially
    )
    # Rename merged columns for home team
    home_ewma_rename_dict = {col: f'adj_hybrid_home_{col}' for col in ewma_cols_generated}
    master_df_merged.rename(columns=home_ewma_rename_dict, inplace=True)
    master_df_merged.drop(columns=['team'], inplace=True) # Drop the 'team' column from the merge

    # Merge for Away Team stats
    master_df_final = pd.merge(
        master_df_merged,
        ewma_features_to_merge,
        left_on=['id', 'away_team'],
        right_on=['id', 'team'],
        how='left',
        suffixes=('', '_y') # Avoid suffix collision
    )
    # Rename merged columns for away team
    away_ewma_rename_dict = {col: f'adj_hybrid_away_{col}' for col in ewma_cols_generated}
    master_df_final.rename(columns=away_ewma_rename_dict, inplace=True)
    master_df_final.drop(columns=['team'], inplace=True) # Drop the 'team' column from the merge

    # Clean up any potential duplicate '_y' columns if merging caused issues (shouldn't with suffixes)
    cols_to_drop = [col for col in master_df_final.columns if col.endswith('_y')]
    if cols_to_drop:
        print(f"Dropping potentially duplicated columns: {cols_to_drop}")
        master_df_final.drop(columns=cols_to_drop, inplace=True)
    master_df = master_df_final
    return master_df

# Create Matchup Features
def create_matchup_features(master_df,stats_to_roll):
    print("Creating matchup features (differences)...")
    matchup_features_seasonal = []
    # Construct names based on the NEW 'seasonal_' prefixed EWMA columns
    for stat in stats_to_roll:
        base_stat_name = stat.split('_')[-1]
        if stat.startswith('offense_'):
            def_equiv_stat_base = stat.replace("offense_", "")
        elif stat.startswith('defense_'):
            def_equiv_stat_base = stat.replace("defense_", "")
        else:
            def_equiv_stat_base = stat

        home_off_col = f'adj_hybrid_home_offense_{def_equiv_stat_base}_ewma_lag1'
        away_def_col = f'adj_hybrid_away_defense_{def_equiv_stat_base}_ewma_lag1'
        away_off_col = f'adj_hybrid_away_offense_{def_equiv_stat_base}_ewma_lag1'
        home_def_col = f'adj_hyrbid_home_defense_{def_equiv_stat_base}_ewma_lag1'

        matchup_col_name_ho_ad = f'matchup_HO_v_AD_{def_equiv_stat_base}'
        matchup_col_name_ao_hd = f'matchup_AO_v_HD_{def_equiv_stat_base}'

        if home_off_col in master_df.columns and away_def_col in master_df.columns:
            master_df[matchup_col_name_ho_ad] = master_df[home_off_col] - master_df[away_def_col]
            if matchup_col_name_ho_ad not in matchup_features_seasonal: matchup_features_seasonal.append(matchup_col_name_ho_ad)

        if away_off_col in master_df.columns and home_def_col in master_df.columns:
            master_df[matchup_col_name_ao_hd] = master_df[away_off_col] - master_df[home_def_col]
            if matchup_col_name_ao_hd not in matchup_features_seasonal: matchup_features_seasonal.append(matchup_col_name_ao_hd)

    print(f"Generated {len(matchup_features_seasonal)} SEASONAL matchup difference features.")
    return master_df

# Returning Production Feature Creation
def create_returning_prod_features(master_df, RP_METRICS_TO_USE, RP_ACTIVE_WEEKS):
    rp_features = []
    rp_diff_features = []

    # Create difference features and conditional features
    for metric in RP_METRICS_TO_USE:
        home_rp_col = f'home_rp_{metric}'
        away_rp_col = f'away_rp_{metric}'

        # 1. Create Difference Feature
        diff_col_name = f'rp_{metric}_diff'
        master_df[diff_col_name] = master_df[home_rp_col] - master_df[away_rp_col]
        rp_diff_features.append(diff_col_name)

        # 2. Create Conditionally Active RP Features (Active Weeks 1-RP_ACTIVE_WEEKS)
        home_rp_cond_col = f'{home_rp_col}_Wk1_{RP_ACTIVE_WEEKS}'
        away_rp_cond_col = f'{away_rp_col}_Wk1_{RP_ACTIVE_WEEKS}'
        diff_cond_col = f'{diff_col_name}_Wk1_{RP_ACTIVE_WEEKS}'

        master_df[home_rp_cond_col] = np.where(master_df['week'] <= RP_ACTIVE_WEEKS, master_df[home_rp_col], 0.0)
        master_df[away_rp_cond_col] = np.where(master_df['week'] <= RP_ACTIVE_WEEKS, master_df[away_rp_col], 0.0)
        master_df[diff_cond_col] = np.where(master_df['week'] <= RP_ACTIVE_WEEKS, master_df[diff_col_name], 0.0)

        # Add the conditional features to our list
        rp_features.extend([home_rp_cond_col, away_rp_cond_col, diff_cond_col])

    print(f"Created {len(rp_diff_features)} RP difference features.")
    print(f"Created {len(rp_features)} conditionally active RP features (Weeks 1-{RP_ACTIVE_WEEKS}).")


    # --- Update Potential Features List ---
    # Redefine the full list to include new RP features and SEASONAL EWMAs
    print("\nUpdating potential features list...")
    # Basic features (ensure these are still relevant and exist)
    basic_features = [col for col in master_df.columns if col in [
        'home_pregame_elo_calc', 'away_pregame_elo_calc', 'neutral_site',
        'conference_game', 'season', 'week', 'elo_diff_calc']]

    # Get names of the newly created seasonal EWMA and matchup columns
    ewma_cols = [col for col in master_df.columns if '_ewma_lag1' in col]
    matchup_cols = [col for col in master_df.columns if 'matchup_' in col]

    # The new RP features are in the 'rp_features' list already

    potential_features = basic_features + ewma_cols + matchup_cols + rp_features
    # Remove potential duplicates if any column accidentally got added twice
    potential_features = sorted(list(set(potential_features)))

    print(f"Total potential features (incl. Seasonal EWMA & Conditional RP): {len(potential_features)}")
    return master_df, potential_features, basic_features

# Inpsect Engineered Features - Unnecessary

# Identify and Quantify Missing Feature Values - Unnecessary

# Implement Imputation Strategy - Unnecessary as we will let XGBoost Handle

# Drop FCS Games
def drop_fcs_games(master_df):
    # Add this cell AFTER all feature engineering (EWMA, RP, Matchups) is complete
    # and merged into master_df, but BEFORE splitting into train/val/test

    print("\n--- Filtering Out Games Against FCS Opponents ---")

    # Assuming 'away_division' column exists and indicates 'fcs'
    if 'away_division' in master_df.columns:
        initial_rows = len(master_df)
        fcs_games_count = (master_df['away_division'] == 'fcs').sum()

        if fcs_games_count > 0:
            master_df = master_df[master_df['away_division'] != 'fcs'].copy()
            rows_dropped = initial_rows - len(master_df)
            print(f"Dropped {rows_dropped} games where away_team division is 'fcs'.")
            print(f"Remaining rows in master_df: {len(master_df)}")
            # Reset index after dropping
            master_df.reset_index(drop=True, inplace=True)
        else:
            print("No games found with away_team division marked as 'fcs'.")
    elif 'away_conference' in master_df.columns:
        # Alternative: Infer FCS based on conference if division is unreliable
        # You would need a list of known FCS conferences
        fcs_conferences = ['Big Sky', 'CAA', 'Ivy League', 'MEAC', 'MVFC', 'NEC', 'OVC', 'Patriot', 'Pioneer', 'Southern', 'Southland', 'SWAC'] # Example list, verify!
        initial_rows = len(master_df)
        fcs_games_mask = master_df['away_conference'].isin(fcs_conferences)
        fcs_games_count = fcs_games_mask.sum()
        if fcs_games_count > 0:
            master_df = master_df[~fcs_games_mask].copy()
            rows_dropped = initial_rows - len(master_df)
            print(f"Dropped {rows_dropped} games based on away_team conference possibly being FCS.")
            print(f"Remaining rows in master_df: {len(master_df)}")
            master_df.reset_index(drop=True, inplace=True)
        else:
            print("No games found with away_team in known FCS conferences.")

    else:
        print("Warning: Cannot identify FCS opponents ('away_division' or 'away_conference' column missing/unreliable). Skipping FCS game removal.")
    
    return master_df

# Verify - Unncecessary

# Preparation - Temporal Split
def temporal_split(TRAIN_END_SEASON, VALIDATION_END_SEASON, master_df, target_variable, potential_features):
    # Define split points (adjust seasons as needed based on your data range)
    # Example: Train through 2020, Validate on 2021-2022, Test on 2023+
    print(f"Splitting data chronologically:")
    print(f"  Training:   Seasons <= {TRAIN_END_SEASON}")
    print(f"  Validation: Seasons > {TRAIN_END_SEASON} and <= {VALIDATION_END_SEASON}")

    train_df = master_df[master_df['season'] <= TRAIN_END_SEASON].copy()
    val_df = master_df[(master_df['season'] > TRAIN_END_SEASON) &
                    (master_df['season'] <= VALIDATION_END_SEASON)].copy()

    print(f"\nData Shapes:")
    print(f"  Training Set:   {train_df.shape}")
    print(f"  Validation Set: {val_df.shape}")

    # Separate features (X) and target (y) for each set
    y_train = train_df[target_variable]
    X_train = train_df[potential_features].copy() # Use .copy() to avoid SettingWithCopyWarning

    y_val = val_df[target_variable]
    X_val = val_df[potential_features].copy()

    print(f"\nFeature matrix shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")

    # --- Handling NaNs Temporarily for Selection Analysis ---
    # VarianceThreshold and .corr() don't handle NaNs well.
    # We'll impute with the MEDIAN calculated *only from the training set*
    # This is TEMPORARY for analysis; the final model might handle NaNs differently.
    print("\nTemporarily imputing NaNs with training set median for selection analysis...")
    features_with_nan = X_train.columns[X_train.isnull().any()].tolist()
    imputation_values = X_train[features_with_nan].median()

    #Create COPIES for analysis to avoid modifying original X_train/X_val
    X_train_analysis = X_train.copy()
    X_val_analysis = X_val.copy()


    X_train_analysis.fillna(imputation_values, inplace=True)
    # Apply the SAME imputation values calculated from training data to validation set
    X_val_analysis.fillna(imputation_values, inplace=True)
    # X_test.fillna(imputation_values, inplace=True) # Impute test set later

    print(f"Temporarily imputed {len(features_with_nan)} columns.")
    print(f"NaN check after imputation (X_train): {X_train_analysis.isnull().sum().sum()}") # Should be 0
    print(f"NaN check after imputation (X_val):   {X_val_analysis.isnull().sum().sum()}")   # Should be 0
    return y_train, X_train, y_val, X_val, X_train_analysis, X_val_analysis, val_df, train_df

# Initial Filtering
def perform_initial_filtering(X_train_analysis,X_train, train_df):
    # --- 1a: Low Variance Feature Removal ---
    print("\n--- Filtering: Low Variance Features ---")
    variance_threshold = 0.005 # Remove features with variance below this (tweak if needed)
    selector_var = VarianceThreshold(threshold=variance_threshold)

    # Fit ONLY on training data
    selector_var.fit(X_train_analysis)

    # Get boolean mask of features to keep
    features_kept_mask_var = selector_var.get_support()
    features_kept_var = X_train_analysis.columns[features_kept_mask_var].tolist()
    features_dropped_var = X_train_analysis.columns[~features_kept_mask_var].tolist()

    print(f"Initial feature count: {X_train_analysis.shape[1]}")
    print(f"Variance Threshold:    {variance_threshold}")
    if features_dropped_var:
        print(f"Features dropped ({len(features_dropped_var)}): {features_dropped_var}")
    else:
        print("No features dropped by variance threshold.")
    print(f"Features remaining:    {len(features_kept_var)}")

    # Update feature list and DataFrames
    current_features = [f for f in X_train.columns if f in features_kept_var]
    # X_train = X_train[current_features]
    # X_val = X_val[current_features]
    # X_test = X_test[current_features]

    # --- 1b: High Missing Value Feature Removal ---
    # Note: We already imputed NaNs for this analysis step. If we hadn't,
    # this step would calculate missing % on the original data BEFORE imputation.
    # Since XGBoost handles NaNs, we might skip strict missing % filtering,
    # but let's include the check for completeness.
    print("\n--- Filtering: High Missing Values (Check on Original Data) ---")
    # Recalculate missing % on the original training features *before* imputation
    missing_perc_orig = train_df[current_features].isnull().mean().sort_values(ascending=False) * 100
    missing_perc_orig = missing_perc_orig[missing_perc_orig > 0]

    missing_threshold = 90.0 # Drop features missing more than this % (e.g., 90%)
    features_to_drop_missing = missing_perc_orig[missing_perc_orig > missing_threshold].index.tolist()

    print(f"Missing Value Threshold: > {missing_threshold}%")
    if features_to_drop_missing:
        print(f"Features to drop due to high missing values ({len(features_to_drop_missing)}): {features_to_drop_missing}")
        # Remove from current_features list and DataFrames
        current_features = [f for f in current_features if f not in features_to_drop_missing]
        # X_train = X_train[current_features]
        # X_val = X_val[current_features]
        # X_test = X_test[current_features]
        print(f"Features remaining after missing value filter: {len(current_features)}")
    else:
        print("No features dropped by high missing value threshold.")
    return current_features

# Correlation Analysis: Target-Feature
def perform_target_correlation_analysis(X_train_analysis, y_train, current_features, target_variable):
    print("\n--- Analysis: Feature-Target Correlation ---")
    # Combine X_train and y_train temporarily for correlation calculation
    train_corr_df = X_train_analysis[current_features].copy()
    train_corr_df[target_variable] = y_train

    # Calculate correlations with the target variable
    correlations = train_corr_df.corr()[target_variable].drop(target_variable) # Drop self-correlation
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print(f"Top 20 Features correlated with '{target_variable}' (Absolute Value):")
    print(correlations_abs.head(20))
    print(f"\nBottom 10 Features correlated with '{target_variable}' (Absolute Value):")
    print(correlations_abs.tail(10))

    # Plotting the distribution of absolute correlations
    plt.figure(figsize=(10, 5))
    sns.histplot(correlations_abs, bins=30, kde=True)
    plt.title(f'Distribution of Absolute Correlations with {target_variable}')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # --- 2b: Feature-Feature Correlation ---
    print("\n--- Analysis: Feature-Feature Correlation ---")
    feature_corr_matrix = X_train_analysis.corr()

    # Find highly correlated pairs
    correlation_threshold = 0.9 # Identify pairs with correlation > this value
    highly_correlated_pairs = []
    cols = feature_corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)): # Iterate upper triangle
            corr_value = feature_corr_matrix.iloc[i, j]
            if abs(corr_value) > correlation_threshold:
                pair = (cols[i], cols[j], round(corr_value, 4))
                highly_correlated_pairs.append(pair)

    print(f"Found {len(highly_correlated_pairs)} pairs with absolute correlation > {correlation_threshold}")

    # Identify which feature in the pair to potentially drop (lower correlation with target)
    features_to_consider_dropping_corr = set()
    if highly_correlated_pairs:
        print("\nHighly Correlated Pairs (Feature1, Feature2, Correlation):")
        # Sort pairs for consistent display (optional)
        highly_correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for pair in highly_correlated_pairs[:20]: # Show top 20 pairs
            f1, f2, corr_val = pair
            corr_f1_target = correlations_abs.get(f1, 0) # Get correlation with target
            corr_f2_target = correlations_abs.get(f2, 0)

            # Suggest dropping the one with lower absolute correlation to target
            drop_candidate = f1 if corr_f1_target < corr_f2_target else f2
            keep_candidate = f2 if drop_candidate == f1 else f1
            print(f"  - ('{f1}' [{corr_f1_target:.3f}], '{f2}' [{corr_f2_target:.3f}], {corr_val}) -> Suggest dropping: '{drop_candidate}'")
            features_to_consider_dropping_corr.add(drop_candidate) # Add suggestion to a set

        print(f"\nTotal features suggested for dropping due to high correlation: {len(features_to_consider_dropping_corr)}")
        # print("Candidates for dropping:", list(features_to_consider_dropping_corr)) # Uncomment to see full list
    return features_to_consider_dropping_corr, correlations_abs

# Model Based Importance of Features
def perform_model_based_importance(X_train, X_train_analysis, y_train, current_features):
    print("\n--- Analysis: Model-Based Feature Importance (GPU Accelerated XGBoost) ---")

    # Use the imputed data (X_train, y_train)

    # Define basic XGBoost parameters for importance calculation
    xgb_importance_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
        'nthread': -1,
        # --- Enable GPU ---
        'device': 'cuda',
        # ------------------
        # Use relatively simple settings for speed
        'eta': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    num_round_importance = 100 # Fewer rounds might suffice for ranking

    print(f"Training initial XGBoost model on GPU for feature importance...")
    dtrain_importance = xgb.DMatrix(X_train_analysis[current_features], label=y_train)
    model_importance = xgb.train(
        xgb_importance_params,
        dtrain_importance,
        num_boost_round=num_round_importance,
        verbose_eval=False
    )

    # Get feature importances (using 'gain' is often good)
    importance_dict = model_importance.get_score(importance_type='gain')
    if not importance_dict:
        importance_dict = model_importance.get_score(importance_type='weight') # Fallback

    # Create DataFrame - Need to handle features potentially not used
    all_features = X_train.columns.tolist()
    importance_values = [importance_dict.get(f, 0) for f in all_features] # Assign 0 if feature not used

    feature_importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importance_values
    }).sort_values('importance', ascending=False)


    print("\nTop 30 Features by XGBoost Importance (Gain):")
    print(feature_importance_df.head(30))
    print("\nBottom 10 Features by XGBoost Importance (Gain):")
    print(feature_importance_df.tail(10))

    # Optional: Plot top N feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(30), palette='viridis')
    plt.title('Top 30 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.show()

    # Store the current list of features after initial filtering
    features_after_initial_analysis = current_features
    print(f"\nFeatures remaining after initial filtering & analysis: {len(features_after_initial_analysis)}")

    print("\n--- Phase 2, Step 1 (Initial Analysis/Filtering) Complete ---")
    return features_after_initial_analysis, feature_importance_df

# Define Candidate Feature Sets
def define_candidate_feature_sets(basic_features, features_after_initial_analysis, features_to_consider_dropping_corr, feature_importance_df, correlations_abs):
    candidate_feature_sets = {}

    # Set A: Basic Features Only (Baseline)
    candidate_feature_sets['A_Basic'] = [f for f in basic_features if f in features_after_initial_analysis]

    # Set B: Top N by Absolute Correlation with Target
    N_corr = 50 # Example: Top 100
    top_corr_features = correlations_abs.head(N_corr).index.tolist()
    candidate_feature_sets[f'B_Top{N_corr}_Corr'] = [f for f in top_corr_features if f in features_after_initial_analysis]

    # Set C: Top N by Random Forest Importance
    N_rf = 50 # Example: Top 100
    top_rf_features = feature_importance_df['feature'].head(N_rf).tolist()
    candidate_feature_sets[f'C_Top{N_rf}_RF_Importance'] = [f for f in top_rf_features if f in features_after_initial_analysis]

    # Set D: Features after removing high inter-feature correlation candidates
    # Start with all features after initial filtering
    features_after_corr_drop = [f for f in features_after_initial_analysis if f not in features_to_consider_dropping_corr]
    candidate_feature_sets['D_Reduced_Correlation'] = features_after_corr_drop

    # Set E: All features remaining after initial filtering (Variance/Missing)
    candidate_feature_sets['E_All_Initial_Filtered'] = features_after_initial_analysis

    # Optional: Combine Top RF + Basic Features
    combined_rf_basic = list(set(top_rf_features + candidate_feature_sets['A_Basic']))
    candidate_feature_sets[f'F_Top{N_rf}_RF_plus_Basic'] = [f for f in combined_rf_basic if f in features_after_initial_analysis]


    print(f"\nDefined {len(candidate_feature_sets)} candidate feature sets:")
    for name, features in candidate_feature_sets.items():
        print(f"  - {name}: {len(features)} features")
    return candidate_feature_sets

# Create Betting Model
def simulate_betting(simulation_input_df, WIN_PAYOUT, LOSS_AMOUNT, BET_THRESHOLD):
    """
    Simulates the betting strategy on provided data with predictions.
    Expects 'predicted_spread_market', 'avg_opening_spread', 'home_points',
    'away_points', and other game identifiers.
    Returns a DataFrame with results for each potential bet.
    """
    results = []

    # Ensure necessary columns exist and drop rows with missing critical data for simulation
    required_cols = ['id', 'season', 'week', 'home_team', 'away_team', 'home_points', 'away_points',
                     'avg_opening_spread', 'neutral_site', # neutral_site might not be needed if HFA baked into prediction
                     'predicted_spread_market'] # This comes from the model now
    sim_df = simulation_input_df[required_cols].copy()
    sim_df.dropna(subset=['avg_opening_spread', 'home_points', 'away_points',
                           'predicted_spread_market'], inplace=True) # Crucial dropna

    if sim_df.empty:
        print("Warning: No games available for betting simulation after dropping NaNs.")
        return pd.DataFrame() # Return empty DataFrame

    # print(f"Simulating bets for {len(sim_df)} games...") # Optional debug

    for index, game in sim_df.iterrows():
        predicted_spread = game['predicted_spread_market']
        opening_spread = game['avg_opening_spread']

        bet_on = None
        profit_loss = 0.0
        result = 'no_bet' # Default if threshold not met
        difference = predicted_spread - opening_spread

        # REVISED Trigger Logic
        if abs(difference) > BET_THRESHOLD:
            if predicted_spread > opening_spread:
                bet_on = 'away'
            elif predicted_spread < opening_spread:
                bet_on = 'home'
            # else: difference == 0, no bet

        # Grade the bet if one was placed
        if bet_on:
            # REVISED: Actual margin from AWAY team perspective
            actual_margin = game['away_points'] - game['home_points']

            if bet_on == 'away':
                if actual_margin > opening_spread: result, profit_loss = 'win', WIN_PAYOUT
                elif actual_margin < opening_spread: result, profit_loss = 'loss', -LOSS_AMOUNT
                else: result, profit_loss = 'push', 0
            elif bet_on == 'home':
                if actual_margin < opening_spread: result, profit_loss = 'win', WIN_PAYOUT
                elif actual_margin > opening_spread: result, profit_loss = 'loss', -LOSS_AMOUNT
                else: result, profit_loss = 'push', 0

        results.append({
            'game_id': game['id'],
            'season': game['season'],
            'predicted_spread_market': predicted_spread,
            'opening_spread': opening_spread,
            'bet_on': bet_on,
            'result': result,
            'profit_loss': profit_loss
        })

    return pd.DataFrame(results)

# Create Evalucation Function

def evaluate_feature_set_with_bets_nan(feature_set_name, features, xgb_params, num_round, X_train_fs_nan, y_train_fs, X_val_fs_nan, y_val_fs, val_df, val_required_cols):
    """Trains XGBoost and evaluates on the validation set."""
    print(f"\n--- Evaluating Feature Set: {feature_set_name} ({len(features)} features) ---")
    start_time = time.time()

    # Select features - using the PRE-IMPUTED data for this loop
    X_train_subset = X_train_fs_nan[features]
    X_val_subset = X_val_fs_nan[features]

    # Prepare data for XGBoost
    dtrain = xgb.DMatrix(X_train_subset, label=y_train_fs, enable_categorical=False)
    dval = xgb.DMatrix(X_val_subset, label=y_val_fs, enable_categorical=False)

    # Train the model
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_round,
        evals=watchlist,
        verbose_eval=False, # Suppress verbose output during training
        # early_stopping_rounds=10 # Optional: Stop early if validation RMSE doesn't improve
    )

    # Predict on validation set
    y_pred_val = model.predict(dval)

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_val_fs, y_pred_val))
    mae = mean_absolute_error(y_val_fs, y_pred_val)
    # Calculate correlation using pandas Series for correct handling
    predictions_series = pd.Series(y_pred_val, index=y_val_fs.index)
    correlation = y_val_fs.corr(predictions_series)
    bias = np.mean(y_pred_val - y_val_fs)

    # --- Optional: Betting Simulation ---
    betting_units = np.nan # Placeholder
    betting_roi = np.nan
    betting_win_rate = np.nan
    # Uncomment and adapt if you have the simulator and necessary columns in val_df
    try:
      # Create a temporary df with predictions and necessary info for sim
      sim_input_df = val_df[val_required_cols].copy()
      sim_input_df['predicted_spread_market'] = y_pred_val
      #Note: The simulator needs the ELO_SPREAD_DIVISOR and HFA from Elo tuning
      #Pass them appropriately if needed by the simulator function
      betting_results = simulate_betting(sim_input_df) # Need elo params
      betting_units = betting_results['profit_loss'].sum()
      total_bets = len(betting_results[betting_results['bet_on'].notna()])
      total_wins = len(betting_results[betting_results['result'] == 'win'])
      total_losses = len(betting_results[betting_results['result'] == 'loss'])
      if (total_wins + total_losses) > 0:
          betting_win_rate = total_wins / (total_wins + total_losses)
      total_risked = (total_wins + total_losses)  # LOSS_AMOUNT=1
      if total_risked > 0:
          betting_roi = (betting_units / total_risked) * 100
    except Exception as e:
      print(f"Betting simulation failed for {feature_set_name}: {e}")
    # ------------------------------------

    end_time = time.time()
    duration = end_time - start_time

    results = {
        'Set Name': feature_set_name,
        'Num Features': len(features),
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'Bias': bias,
        'Betting Units': betting_units, # Will be NaN if simulation skipped
        'Betting Win Rate': betting_win_rate, # Will be NaN if simulation skipped
        'Betting ROI': betting_roi, # Will be NaN if simulation skipped
        'Eval Time (s)': duration
    }

    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, Correlation: {correlation:.4f}, Bias: {bias:.4f}, Time: {duration:.1f}s")
    # print(f"  Betting: Units={betting_units:.2f}, Win Rate={betting_win_rate:.2%}, ROI={betting_roi:.2f}%") # Uncomment if sim runs

    return results

# Run Feature Set Evaluation

def run_feature_set_evaluation(candidate_feature_sets, xgb_params, num_boost_round, X_train, y_train, X_val, y_val, val_df):
    all_results = []
    for name, feature_list in candidate_feature_sets.items():
        # Ensure feature list is not empty and features exist in X_train
        valid_features = [f for f in feature_list if f in X_train.columns]
        if not valid_features:
            print(f"\n--- Skipping Feature Set: {name} (No valid features found) ---")
            continue
        if len(valid_features) < len(feature_list):
            print(f"\nWarning: Some features for set '{name}' were not found in X_train after filtering.")

        result = evaluate_feature_set_with_bets_nan(name, valid_features, xgb_params, num_boost_round, X_train, y_train, X_val, y_val, val_df)
        all_results.append(result)
    return all_results

# Present Feature Set Evaluation Results
def present_feature_set_evaluation_results(all_results):
    print("\n--- Feature Set Evaluation Summary ---")
    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by='RMSE', ascending=True, inplace=True) # Sort by primary metric (RMSE)

    # Format columns for display
    results_df['RMSE'] = results_df['RMSE'].map('{:.4f}'.format)
    results_df['MAE'] = results_df['MAE'].map('{:.4f}'.format)
    results_df['Correlation'] = results_df['Correlation'].map('{:.4f}'.format)
    results_df['Bias'] = results_df['Bias'].map('{:.4f}'.format)
    results_df['Betting Units'] = results_df['Betting Units'].map('{:.2f}'.format)
    results_df['Betting Win Rate'] = results_df['Betting Win Rate'].map('{:.2%}'.format)
    results_df['Betting ROI'] = results_df['Betting ROI'].map('{:.2f}%'.format)
    results_df['Eval Time (s)'] = results_df['Eval Time (s)'].map('{:.1f}'.format)


    print(results_df.to_string(index=False))
    print("Evaluated candidate feature sets on the validation data.")
    return results_df

# Select Best Feature Set
def select_best_feature_set(results_df, candidate_feature_sets, X_train, X_val):
    best_feature_set_name = results_df.iloc[0]['Set Name'] # Assumes results_df sorted by best metric
    best_features = candidate_feature_sets[best_feature_set_name]

    print(f"Selected best feature set for tuning: '{best_feature_set_name}' ({len(best_features)} features)")

    # Prepare final training and validation data with ONLY the selected features
    # Using the temporarily imputed data from the previous step for consistency during tuning
    X_train_best = X_train[best_features].copy()
    X_val_best = X_val[best_features].copy()

    print(f"Using feature shapes: X_train_best={X_train_best.shape}, X_val_best={X_val_best.shape}")
    return best_features

# Define Objective Function for Optuna
def objective_xgb_nan(trial, X_train_hp_nan, y_train_hp, X_val_hp_nan, y_val_hp, validation_df_hp, val_required_cols, WIN_PAYOUT, LOSS_AMOUNT, BET_THRESHOLD):
    """Objective function for Optuna XGBoost hyperparameter tuning."""

    # --- 3. Define Search Space ---
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
        'nthread': -1, # Use all cores
        'device': 'cuda', # Enable GPU Acceleration
        # Parameters to tune:
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),             # Learning rate
        'max_depth': trial.suggest_int('max_depth', 3, 9),                  # Max tree depth
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),            # Row subsampling
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Feature subsampling
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),   # Min samples in leaf node
        'gamma': trial.suggest_float('gamma', 0, 0.5),                      # Min loss reduction for split
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),       # L2 regularization
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),         # L1 regularization
        # 'missing': np.nan # Enable if using non-imputed data
    }

    # Fixed number of boosting rounds for tuning (can be tuned itself later)
    # Or use early stopping within the training call
    num_boost_round_hp = 200 # Increase rounds for tuning?

    # Prepare data
    dtrain = xgb.DMatrix(X_train_hp_nan, label=y_train_hp)
    dval = xgb.DMatrix(X_val_hp_nan, label=y_val_hp)
    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    # Train model with suggested parameters
    # Using early stopping is highly recommended during tuning
    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round_hp,
        evals=watchlist,
        early_stopping_rounds=25, # Stop if validation RMSE doesn't improve for 25 rounds
        verbose_eval=False # Keep quiet during Optuna trials
    )

    # Evaluate on validation set
    y_pred_val_hp = bst.predict(dval, iteration_range=(0, bst.best_iteration)) # Use best iteration
    rmse_val = np.sqrt(mean_squared_error(y_val_hp, y_pred_val_hp))

    # --- Optional: Calculate betting simulation for this trial ---
    # This adds significant time to each trial but directly optimizes for bets

    run_betting_sim_in_tuning = False # Set to False to speed up tuning based only on RMSE

    if run_betting_sim_in_tuning:
        try:
            sim_input_df_hp = validation_df_hp[val_required_cols].copy()
            sim_input_df_hp['predicted_spread_market'] = pd.Series(y_pred_val_hp, index=validation_df_hp.index) # Align index
            betting_results_hp = simulate_betting(sim_input_df_hp, WIN_PAYOUT, LOSS_AMOUNT, BET_THRESHOLD)

            if not betting_results_hp.empty:
                betting_units_hp = betting_results_hp['profit_loss'].sum()
                bets_placed_df_hp = betting_results_hp[betting_results_hp['bet_on'].notna()]
                wins_hp = len(bets_placed_df_hp[bets_placed_df_hp['result'] == 'win'])
                losses_hp = len(bets_placed_df_hp[bets_placed_df_hp['result'] == 'loss'])
                if (wins_hp + losses_hp) > 0:
                    total_risked_hp = (wins_hp + losses_hp) * LOSS_AMOUNT
                    roi_hp = (betting_units_hp / total_risked_hp) * 100 if total_risked_hp > 0 else 0.0
                else:
                     roi_hp = 0.0 # Or NaN
                 # Store ROI as user attribute to see it, but still optimize RMSE
                trial.set_user_attr("val_roi", roi_hp)
                trial.set_user_attr("val_units", betting_units_hp)
            else:
                 trial.set_user_attr("val_roi", np.nan)
                 trial.set_user_attr("val_units", 0.0)

        except Exception as e:
             print(f"Warning: Betting sim failed in trial {trial.number}: {e}")
             trial.set_user_attr("val_roi", np.nan)
             trial.set_user_attr("val_units", np.nan)

    # Return the metric to minimize (Validation RMSE)
    return rmse_val

# Run Optuna Study

def run_optuna_study(X_train, X_val, y_train, y_val, val_df, best_features, WIN_PAYOUT, LOSS_AMOUNT, BET_THRESHOLD):
    N_TRIALS_HP = 100 # Number of hyperparameter combinations to test (adjust as needed)

    print(f"\nStarting Optuna hyperparameter search ({N_TRIALS_HP} trials)...")
    study_hp = optuna.create_study(direction='minimize', study_name='XGBoost Spread Prediction NaN') # Minimize RMSE

    X_train_best_nan = X_train[best_features].copy()
    X_val_best_nan = X_val[best_features].copy()

    # Pass necessary dataframes via lambda function
    study_hp.optimize(
        lambda trial: objective_xgb_nan(trial, X_train_best_nan, y_train, X_val_best_nan, y_val, val_df, WIN_PAYOUT, LOSS_AMOUNT, BET_THRESHOLD),
        n_trials=N_TRIALS_HP,
        show_progress_bar=True
    )
    print("\nOptimization Finished.")
    print(f"Number of finished trials: {len(study_hp.trials)}")
    print(f"Best trial (Validation RMSE): {study_hp.best_value:.4f}")
    return study_hp

# Identify Best Hyperparameters

def identify_best_hyperparameters(study_hp):
    print("Best hyperparameters:")
    best_xgb_params = study_hp.best_params
    for key, value in best_xgb_params.items():
        print(f"  {key}: {value}")

    # --- Optional: Show best trial's betting performance ---
    best_trial_info = study_hp.best_trial
    if 'val_roi' in best_trial_info.user_attrs:
        print(f"\nBest Trial's Validation Betting Performance:")
        print(f"  Units: {best_trial_info.user_attrs.get('val_units', 'N/A'):.2f}")
        print(f"  ROI:   {best_trial_info.user_attrs.get('val_roi', 'N/A'):.2f}%")


    # --- Retrain Final Model with Best Parameters (Optional here, usually done before Test Set) ---
    # You would typically save these `best_xgb_params` and use them to train a final model
    # on the *combined* training + validation data before predicting on the test set.
    # For now, we have identified the best settings based on validation performance.

    print("\n--- Phase 3 (Hyperparameter Tuning) Complete ---")
    print("Identified best XGBoost hyperparameters based on validation set performance.")
    print("Next Steps: Potentially adding more complex features (Opponent Adj, Returning Prod) OR final evaluation on the Test Set.")
    return best_xgb_params

# Define Train + Validation and Test Sets
def define_train_val_test_sets(master_df, VALIDATION_END_SEASON, TEST_START_SEASON):

    print(f"Using final split points:")
    print(f"  Train+Validation: Seasons <= {VALIDATION_END_SEASON}")
    print(f"  Test:             Seasons >= {TEST_START_SEASON}")

    train_val_df = master_df[master_df['season'] <= VALIDATION_END_SEASON].copy()
    test_df = master_df[master_df['season'] >= TEST_START_SEASON].copy()

    print(f"\nData Shapes:")
    print(f"  Train+Validation Set: {train_val_df.shape}")
    print(f"  Test Set:             {test_df.shape}")

    if test_df.empty:
        print("\nERROR: Test set is empty. Cannot perform final evaluation.")
        # Exit or handle appropriately
    exit()
    return train_val_df, test_df

# Prepare Data for Final Model
def prepare_data_for_final_model(train_val_df, test_df, best_features, target_variable):
    # Select the best features identified earlier
    X_train_val_nan = train_val_df[best_features].copy()
    y_train_val = train_val_df[target_variable]

    X_test_nan = test_df[best_features].copy()
    y_test = test_df[target_variable]

    print(f"\nFeature matrix shapes for final model:")
    print(f"  X_train_val: {X_train_val_nan.shape}")
    print(f"  X_test:      {X_test_nan.shape}")

    # --- Handle Missing Values (Strategy Decision) ---
    # IMPORTANT: Use the SAME strategy as during hyperparameter tuning.
    # If XGBoost's internal NaN handling was used (recommended), do nothing here.
    # If imputation was done (e.g., median), apply it here using values
    # calculated ONLY from the original *training* set (train_df).

    # Assuming XGBoost internal NaN handling (if tree_method='gpu_hist' was used):
    print("\nAssuming XGBoost will handle NaNs internally (no imputation applied).")
    # If you need to impute (e.g., using training median):
    # features_with_nan_final = X_train_val.columns[X_train_val.isnull().any()].tolist()
    # # Calculate median ONLY from original training data (train_df must exist)
    # imputation_values_final = train_df[features_with_nan_final].median()
    # print(f"Applying median imputation based on original training data to {len(features_with_nan_final)} columns...")
    # X_train_val.fillna(imputation_values_final, inplace=True)
    # X_test.fillna(imputation_values_final, inplace=True)
    # print(f"NaN check after imputation (X_train_val): {X_train_val.isnull().sum().sum()}")
    # print(f"NaN check after imputation (X_test):      {X_test.isnull().sum().sum()}")

    # Prepare data for XGBoost
    print("Preparing DMatrix for XGBoost...")
    dtrain_val = xgb.DMatrix(X_train_val_nan, label=y_train_val)
    dtest = xgb.DMatrix(X_test_nan, label=y_test)
    return X_train_val_nan, y_train_val, X_test_nan, y_test, dtrain_val, dtest

# Train Final XGBoost Model
def train_final_model(best_xgb_params, best_features, dtrain_val, dtest):
    # Combine base parameters with the best ones found by Optuna
    final_xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
        'nthread': -1,
        # Add 'tree_method': 'gpu_hist' IF you are using a GPU runtime for this final training
        'device': 'cuda'
    }
    final_xgb_params.update(best_xgb_params) # Add tuned parameters

    # Determine the number of boosting rounds
    # Option 1: Use a fixed number (e.g., the one used in tuning or slightly more)
    # num_boost_round_final = 200 # Example
    # Option 2: If you logged bst.best_iteration from the best Optuna trial, use that.
    # best_iteration = ??? # Need to retrieve this value from tuning results if possible
    # Option 3: Train with early stopping against a small validation split *of the train_val_df*
    # This is safer but adds complexity. Let's use a fixed number for now.
    num_boost_round_final = 200 # Use the value determined during tuning or a reasonable default
    print(f"Training final XGBoost model with {len(best_features)} features for {num_boost_round_final} rounds...")
    print("Using hyperparameters:", final_xgb_params)

    start_train_time = time.time()
    final_model = xgb.train(
        final_xgb_params,
        dtrain_val,
        num_boost_round=num_boost_round_final,
        evals=[(dtrain_val, 'train'), (dtest, 'test')], # Monitor performance on test set during training
        verbose_eval=50 # Print progress every 50 rounds
    )
    end_train_time = time.time()
    print(f"Final model training finished in {end_train_time - start_train_time:.2f} seconds.")
    return final_model

# Predict on the Test Set
def predict_test_set(final_model, dtest, y_test):
    print("\nPredicting on Test Set...")
    y_pred_test = final_model.predict(dtest)
    predictions_test_series = pd.Series(y_pred_test, index=y_test.index)
    return predictions_test_series, y_pred_test

# Evaluate Statistical Metrics on Test Set
def evaluate_model_statistics(y_test, y_pred_test, predictions_test_series, test_df, val_required_cols, LOSS_AMOUNT):
    print("\n--- Test Set Statistical Performance ---")

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    correlation_test = y_test.corr(predictions_test_series)
    bias_test = np.mean(y_pred_test - y_test)
    r2_test = r2_score(y_test, y_pred_test) # R-squared

    print(f"  RMSE:        {rmse_test:.4f}")
    print(f"  MAE:         {mae_test:.4f}")
    print(f"  Correlation: {correlation_test:.4f}")
    print(f"  Bias:        {bias_test:.4f}")
    print(f"  R-squared:   {r2_test:.4f}")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Ideal Fit')
    plt.xlabel("Actual Closing Spread (y_test)")
    plt.ylabel("Predicted Closing Spread (y_pred_test)")
    plt.title(f"Test Set: Actual vs. Predicted Spread (Corr: {correlation_test:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate Betting Performance
    print("\n--- Test Set Betting Performance ---")

    # Prepare input for the simulator
    # Select necessary columns from the original test_df
    sim_input_test_df = test_df[val_required_cols].copy()
    # Add the predictions, ensuring index alignment
    sim_input_test_df['predicted_spread_market'] = predictions_test_series

    # Run the simulation
    test_betting_results = simulate_betting(sim_input_test_df)

    if test_betting_results.empty:
        print("No bets were placed on the test set according to the strategy.")
    else:
        # Aggregate results
        test_total_units = test_betting_results['profit_loss'].sum()
        test_bets_placed_df = test_betting_results[test_betting_results['bet_on'].notna()]
        test_total_bets = len(test_bets_placed_df)
        test_wins = len(test_bets_placed_df[test_bets_placed_df['result'] == 'win'])
        test_losses = len(test_bets_placed_df[test_bets_placed_df['result'] == 'loss'])
        test_pushes = len(test_bets_placed_df[test_bets_placed_df['result'] == 'push'])

        test_win_rate = test_wins / (test_wins + test_losses) if (test_wins + test_losses) > 0 else np.nan
        test_total_risked = (test_wins + test_losses) * LOSS_AMOUNT
        test_roi = (test_total_units / test_total_risked) * 100 if test_total_risked > 0 else 0.0

        print(f"  Total Bets:   {test_total_bets}")
        print(f"  Wins:         {test_wins}")
        print(f"  Losses:       {test_losses}")
        print(f"  Pushes:       {test_pushes}")
        print(f"  Win Rate:     {test_win_rate:.2%}")
        print(f"  Total Units:  {test_total_units:+.2f}")
        print(f"  ROI:          {test_roi:.2f}%")
    # Optional: Plot cumulative units over time for the test set
    test_betting_results = pd.merge(test_betting_results[['game_id', 'profit_loss']],
                                    test_df[['id', 'season', 'week']],
                                    left_on='game_id', right_on='id')
    test_betting_results['game_date_order'] = test_betting_results['season'] * 100 + test_betting_results['week']
    test_betting_results.sort_values('game_date_order', inplace=True)
    test_betting_results['cumulative_units'] = test_betting_results['profit_loss'].cumsum()

    plt.figure(figsize=(12, 6))
    test_betting_results['cumulative_units'].plot()
    plt.title('Test Set Cumulative Units Over Time')
    plt.xlabel('Games (Chronological Order)')
    plt.ylabel('Cumulative Units')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
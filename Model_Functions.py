import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from opponent_adjustments import get_opponent_adjustments

# Config - Major Inputs
DB_PATH = "cfb_data.db"
PRE_GAME_ELO_CSV_PATH = 'games_with_pregame_elo.csv'
#Define RP metrics to load and use
RP_METRICS_TO_USE =['usage','percentPPA']
# Define dfault value for missing RP data (e.g., average)
DEFAULT_RP_VALUE = 0.5
# Define how many weeks RP features should be active
RP_ACTIVE_WEEKS = 4
#Betting Parameters
BET_THRESHOLD = 0.5
WIN_PAYOUT = 0.909
LOSS_AMOUNT = 1
# EWMA Parameters
EWMA_SPAN = 5
min_periods_for_ewma = max(1, EWMA_SPAN // 2)
# Train / Test Split Years
TRAIN_END_SEASON = 2020
VALIDATION_END_SEASON = 2022


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
    return master_df, potential_features

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
def temporal_split(TRAIN_END_SEASON, VALIDATION_END_SEASON, master_df):
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
    return y_train, X_train, y_val, X_val, X_train_analysis, X_val_analysis

# Initial Filtering
def perform_initial_filtering():
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

# Correlation Analysis
def perform_target_correlation_analysis(X_train_analysis, y_train):
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

def perform_feature_feature_correlation_analysis():
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
    return features_to_consider_dropping_corr



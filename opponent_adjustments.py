import pandas as pd
import numpy as np
import warnings
import sqlite3
import os

# Warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Config
DB_PATH = "cfb_data.db"
PRE_GAME_ELO_CSV_PATH = 'games_with_pregame_elo.csv'
#Define RP metrics to load and use
RP_METRICS_TO_USE =['usage','percentPPA']
DEFAULT_RP_VALUE = 0.5
EPSILON = 1e-6
TEAM_NAN_HANDLING = 'zero'

def get_opponent_adjustments(master_df):
    df = master_df.copy()
    # Identify stats to adjust
    base_stats_offense = [
        'ppa', 'successRate', 'explosiveness',
        'standardDowns_ppa', 'standardDowns_successRate', 'standardDowns_explosiveness',
        'passingDowns_ppa', 'passingDowns_successRate', 'passingDowns_explosiveness',
        'rushingPlays_ppa', 'rushingPlays_successRate',
        'rushingPlays_explosiveness', 'passingPlays_ppa',
        'passingPlays_successRate', 'passingPlays_explosiveness'
    ]

    base_stats_defense = base_stats_offense.copy()

    # Generate full column names
    home_offense_cols = [f'home_offense_{stat}' for stat in base_stats_offense]
    away_offense_cols = [f'away_offense_{stat}' for stat in base_stats_offense]
    home_defense_cols = [f'home_defense_{stat}' for stat in base_stats_defense]
    away_defense_cols = [f'away_defense_{stat}' for stat in base_stats_defense]
    all_stat_cols = home_offense_cols + away_offense_cols + home_defense_cols + away_defense_cols

    # Ensure stats are numeric
    for col in all_stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
            
    # Calculate Averages Needed for Adjustment
    # League averages per Season
    final_season_averages = {}
    league_weekly_data_list = []

    for season, season_df in df.groupby('season'):
        season_stats = {}
        for stat in base_stats_offense:
            home_col = f'home_offense_{stat}'
            away_col = f'away_offense_{stat}'
            avg_col_name_off = f'league_avg_off_{stat}'
            if home_col in season_df.columns and away_col in season_df.columns:
                valid_data = pd.concat([season_df[home_col], season_df[away_col]]).dropna()
                # Calculate final average for the whole season
                final_avg = valid_data.mean() if not valid_data.empty else 0
                season_stats[f'{avg_col_name_off}_final'] = final_avg

                # Aggregate weekly data for rolling average calculation later
                weekly_sum = season_df.groupby('week')[home_col].sum().fillna(0) + season_df.groupby('week')[away_col].sum().fillna(0)
                weekly_count = season_df.groupby('week')[home_col].count() + season_df.groupby('week')[away_col].count() # Counts non-NaN implicitly? Let's be explicit.
                weekly_count_h = season_df.groupby('week')[home_col].apply(lambda x: x.notna().sum())
                weekly_count_a = season_df.groupby('week')[away_col].apply(lambda x: x.notna().sum())
                weekly_count = weekly_count_h.add(weekly_count_a, fill_value=0)

                weekly_avg = weekly_sum / weekly_count.replace(0, 1)
                weekly_avg.name = avg_col_name_off # Name series for merging
                league_weekly_data_list.append(weekly_avg.reset_index().assign(season=season))


        for stat in base_stats_defense:
            home_col = f'home_defense_{stat}'
            away_col = f'away_defense_{stat}'
            avg_col_name_def = f'league_avg_def_{stat}'
            if home_col in season_df.columns and away_col in season_df.columns:
                valid_data = pd.concat([season_df[home_col], season_df[away_col]]).dropna()
                final_avg = valid_data.mean() if not valid_data.empty else 0
                season_stats[f'{avg_col_name_def}_final'] = final_avg

                # Aggregate weekly data
                weekly_sum = season_df.groupby('week')[home_col].sum().fillna(0) + season_df.groupby('week')[away_col].sum().fillna(0)
                weekly_count_h = season_df.groupby('week')[home_col].apply(lambda x: x.notna().sum())
                weekly_count_a = season_df.groupby('week')[away_col].apply(lambda x: x.notna().sum())
                weekly_count = weekly_count_h.add(weekly_count_a, fill_value=0)

                weekly_avg = weekly_sum / weekly_count.replace(0, 1)
                weekly_avg.name = avg_col_name_def # Name series for merging
                league_weekly_data_list.append(weekly_avg.reset_index().assign(season=season))


        final_season_averages[season] = season_stats

    # Convert final averages dict to DataFrame for easier mapping
    final_season_averages_df = pd.DataFrame.from_dict(final_season_averages, orient='index').reset_index().rename(columns={'index': 'season'})

    # Combine all weekly league averages
    league_weekly_df = pd.concat(league_weekly_data_list)
    # Pivot to have stats as columns, indexed by season and week
    league_weekly_pivot = league_weekly_df.pivot_table(index=['season', 'week'], values=league_weekly_df.columns.drop(['season', 'week']), aggfunc='mean') # Use mean in case of duplicates (shouldn't happen)
    
    # Lagged Rolling 3 Week Average
    rolling_avg_cols = {}
    for col in league_weekly_pivot.columns:
        # Calculate rolling mean within each season, then shift
        rolling_avg_col_name = f'{col}_rolling_3wk_lag1'
        # Apply rolling calculation within each season group
        # Reset index to allow grouping on season, then set index back for potential joins
        temp_pivot = league_weekly_pivot.reset_index()
        temp_pivot[rolling_avg_col_name] = temp_pivot.groupby('season')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        league_weekly_pivot = temp_pivot.set_index(['season', 'week']) # Set index back
        rolling_avg_cols[col] = rolling_avg_col_name # Store new column names
        
    # Map Previous Season's Final Average;
    df['prev_season'] = df['season'] - 1
    df = pd.merge(df, final_season_averages_df, left_on='prev_season', right_on='season', how='left', suffixes=('', '_prev_final'))
    df.drop(columns=['season_prev_final', 'prev_season'], inplace=True) # Clean up merge columns
    
    # Map Current Season's Lagged Rolling 3 Week Average
    rolling_cols_to_merge = [col for col in league_weekly_pivot.columns if '_rolling_3wk_lag1' in col]
    df = pd.merge(df, league_weekly_pivot[rolling_cols_to_merge], on=['season', 'week'], how='left')
    
    # Create a hybrid League Average Column
    print("\nCreating hybrid league average columns...")
    FIRST_SEASON_FALLBACK = 'zero'
    hybrid_league_avg_cols = []
    first_season = df['season'].min()

    for stat in base_stats_offense:
        base_name = f'league_avg_off_{stat}'
        prev_season_col = f'{base_name}_final'
        rolling_avg_col = f'{base_name}_rolling_3wk_lag1'
        hybrid_col = f'{base_name}_hybrid_lag1'
        hybrid_league_avg_cols.append(hybrid_col)

        if prev_season_col in df.columns and rolling_avg_col in df.columns:
            # --- Handle first season fallback ---
            # Option 1: Fill previous season NaN with 0
            if FIRST_SEASON_FALLBACK == 'zero':
                df[prev_season_col].fillna(0, inplace=True)
            # Option 2: Fill with the rolling average value that starts in week 4
            elif FIRST_SEASON_FALLBACK == 'use_rolling_avg_wk4':
                 # Find the value used in week 4 for that season and stat
                 wk4_values = df.loc[df['week'] == 4, ['season', rolling_avg_col]].set_index('season')
                 # Map these week 4 values to the NaNs in the prev_season_col for the first season
                 is_first_season = df['season'] == first_season
                 df.loc[is_first_season, prev_season_col] = df.loc[is_first_season, 'season'].map(wk4_values[rolling_avg_col])
                 # Still might have NaNs if wk4 value itself was NaN, fill remaining with 0
                 df[prev_season_col].fillna(0, inplace=True)
            else: # Default to zero if option invalid
                 df[prev_season_col].fillna(0, inplace=True)

            # Fill NaNs in rolling average (especially early weeks) - using forward fill within season? Or 0? Let's use 0 for now.
            # A better approach might be to backfill from week 4's value for weeks 1-3 if needed.
            df[rolling_avg_col] = df.groupby('season')[rolling_avg_col].ffill().bfill() # Fill within season
            df[rolling_avg_col].fillna(0, inplace=True) # Fill any remaining NaNs


            # Apply conditional logic
            df[hybrid_col] = np.where(
                df['week'] <= 3,
                df[prev_season_col],  # Use previous season's final average
                df[rolling_avg_col]   # Use current season's lagged rolling 3-week average
            )
        else:
            print(f"Warning: Missing columns for hybrid avg: {prev_season_col} or {rolling_avg_col}")
            df[hybrid_col] = 0 # Assign default if columns missing


    for stat in base_stats_defense:
        base_name = f'league_avg_def_{stat}'
        prev_season_col = f'{base_name}_final'
        rolling_avg_col = f'{base_name}_rolling_3wk_lag1'
        hybrid_col = f'{base_name}_hybrid_lag1'
        hybrid_league_avg_cols.append(hybrid_col)

        if prev_season_col in df.columns and rolling_avg_col in df.columns:
            # Handle first season fallback
            if FIRST_SEASON_FALLBACK == 'zero':
                df[prev_season_col].fillna(0, inplace=True)
            elif FIRST_SEASON_FALLBACK == 'use_rolling_avg_wk4':
                 wk4_values = df.loc[df['week'] == 4, ['season', rolling_avg_col]].set_index('season')
                 is_first_season = df['season'] == first_season
                 df.loc[is_first_season, prev_season_col] = df.loc[is_first_season, 'season'].map(wk4_values[rolling_avg_col])
                 df[prev_season_col].fillna(0, inplace=True) # Fill remaining with 0
            else:
                 df[prev_season_col].fillna(0, inplace=True)

            # Fill NaNs in rolling average
            df[rolling_avg_col] = df.groupby('season')[rolling_avg_col].ffill().bfill() # Fill within season
            df[rolling_avg_col].fillna(0, inplace=True)

            # Apply conditional logic
            df[hybrid_col] = np.where(
                df['week'] <= 3,
                df[prev_season_col],
                df[rolling_avg_col]
            )
        else:
            print(f"Warning: Missing columns for hybrid avg: {prev_season_col} or {rolling_avg_col}")
            df[hybrid_col] = 0
            
    # Calculate Lagged Expanding Team Averages
    print("\nCalculating lagged expanding team averages...")
    # Create the team-centric view again
    home_view = df[['id', 'season', 'week', 'homeTeam', 'awayTeam'] + home_offense_cols + home_defense_cols].copy()
    away_view = df[['id', 'season', 'week', 'awayTeam', 'homeTeam'] + away_offense_cols + away_defense_cols].copy()
    # ... (rest of team_centric_df creation and column renaming is identical to previous version) ...
    home_view.rename(columns={'homeTeam': 'team', 'awayTeam': 'opponent'}, inplace=True)
    home_view.columns = [col.replace('home_offense_', 'offense_') for col in home_view.columns]
    home_view.columns = [col.replace('home_defense_', 'defense_') for col in home_view.columns]
    home_view['is_home'] = 1
    away_view.rename(columns={'awayTeam': 'team', 'homeTeam': 'opponent'}, inplace=True)
    away_view.columns = [col.replace('away_offense_', 'offense_') for col in away_view.columns]
    away_view.columns = [col.replace('away_defense_', 'defense_') for col in away_view.columns]
    away_view['is_home'] = 0
    team_centric_df = pd.concat([home_view, away_view], ignore_index=True)
    team_centric_df.sort_values(by=['season', 'week', 'id', 'is_home'], inplace=True)

    # Calculate lagged expanding means *within each team's season*
    team_avg_cols = []
    calculated_team_global_means = {} # For potential global mean filling if needed
    for stat in base_stats_offense:
        col = f'offense_{stat}'
        if col in team_centric_df.columns:
            avg_col_name = f'avg_{col}_gained_exp_lag1'
            team_centric_df[avg_col_name] = team_centric_df.groupby(['season', 'team'])[col].transform(
                lambda x: x.expanding(min_periods=1).mean().shift(1)
            )
            team_avg_cols.append(avg_col_name)
            calculated_team_global_means[avg_col_name] = team_centric_df[avg_col_name].mean()


    for stat in base_stats_defense:
        col = f'defense_{stat}'
        if col in team_centric_df.columns:
            avg_col_name = f'avg_{col}_allowed_exp_lag1'
            team_centric_df[avg_col_name] = team_centric_df.groupby(['season', 'team'])[col].transform(
                lambda x: x.expanding(min_periods=1).mean().shift(1)
            )
            team_avg_cols.append(avg_col_name)
            calculated_team_global_means[avg_col_name] = team_centric_df[avg_col_name].mean()

    # Handle NaNs for team averages
    if TEAM_NAN_HANDLING == 'zero':
        print("Filling initial team NaNs with 0.")
        for col in team_avg_cols:
             team_centric_df[col].fillna(0, inplace=True)
    elif TEAM_NAN_HANDLING == 'global_mean':
         print("Filling initial team NaNs with global mean of that stat.")
         for col in team_avg_cols:
             fill_val = calculated_team_global_means.get(col, 0)
             team_centric_df[col].fillna(fill_val, inplace=True)
     
    # Merge Lagged Team Averages Back
    cols_to_merge = ['id', 'season', 'week', 'team'] + team_avg_cols

    # Prepare and Merge Home Perspective Averages
    home_avg_slice = team_centric_df[team_centric_df['is_home'] == 1][cols_to_merge].copy()
    # Define renaming dict for clarity
    home_rename_dict = {col: f"{col}_home_perspective" for col in team_avg_cols}
    home_avg_slice.rename(columns=home_rename_dict, inplace=True)

    # Perform the merge
    df = pd.merge(
        df,
        home_avg_slice,
        left_on=['id', 'season', 'week', 'homeTeam'],
        right_on=['id', 'season', 'week', 'team'],
        how='left'
    )
    #drop the redundant 'team' column from the right side of the merge
    df.drop(columns=['team'], inplace=True, errors='ignore')

    # Prepare and Merge Away Perspective Averages
    away_avg_slice = team_centric_df[team_centric_df['is_home'] == 0][cols_to_merge].copy()
    # Define renaming dict
    away_rename_dict = {col: f"{col}_away_perspective" for col in team_avg_cols}
    away_avg_slice.rename(columns=away_rename_dict, inplace=True)

    # Perform the merge
    df = pd.merge(
        df,
        away_avg_slice,
        left_on=['id', 'season', 'week', 'awayTeam'],
        right_on=['id', 'season', 'week', 'team'],
        how='left'
    )
    # Drop the redundant 'team' column
    df.drop(columns=['team'], inplace=True, errors='ignore')


    print(f"DataFrame shape after revised merging: {df.shape}")
    
    # Apply Opponent Adjustment
    print("\nApplying Opponent Adjustments with Hybrid League Average...")
    adj_prefix = 'adj_hybrid_' # New prefix for clarity

    # Adjust Home Offense vs Away Defense
    for stat in base_stats_offense:
        game_stat_col = f'home_offense_{stat}'
        opponent_avg_col = f'avg_defense_{stat}_allowed_exp_lag1_away_perspective' # Away team's lagged DEFENSE ALLOWED
        league_avg_col = f'league_avg_off_{stat}_hybrid_lag1' # Use the HYBRID league average
        adj_col_name = f'{adj_prefix}{game_stat_col}'

        if all(c in df.columns for c in [game_stat_col, opponent_avg_col, league_avg_col]):
            df[adj_col_name] = df[game_stat_col] - df[opponent_avg_col] + df[league_avg_col]
        # else: print(f"Skipping {adj_col_name}: Missing columns")

    # Adjust Away Offense vs Home Defense
    for stat in base_stats_offense:
        game_stat_col = f'away_offense_{stat}'
        opponent_avg_col = f'avg_defense_{stat}_allowed_exp_lag1_home_perspective' # Home team's lagged DEFENSE ALLOWED
        league_avg_col = f'league_avg_off_{stat}_hybrid_lag1' # Use the HYBRID league average
        adj_col_name = f'{adj_prefix}{game_stat_col}'

        if all(c in df.columns for c in [game_stat_col, opponent_avg_col, league_avg_col]):
          df[adj_col_name] = df[game_stat_col] - df[opponent_avg_col] + df[league_avg_col]
        # else: print(f"Skipping {adj_col_name}: Missing columns")

    # Adjust Home Defense vs Away Offense
    for stat in base_stats_defense:
        game_stat_col = f'home_defense_{stat}'
        opponent_avg_col = f'avg_offense_{stat}_gained_exp_lag1_away_perspective' # Away team's lagged OFFENSE GAINED
        league_avg_col = f'league_avg_def_{stat}_hybrid_lag1' # Use the HYBRID league average
        adj_col_name = f'{adj_prefix}{game_stat_col}'

        if all(c in df.columns for c in [game_stat_col, opponent_avg_col, league_avg_col]):
           df[adj_col_name] = df[game_stat_col] - df[opponent_avg_col] + df[league_avg_col]
        # else: print(f"Skipping {adj_col_name}: Missing columns")

    # Adjust Away Defense vs Home Offense
    for stat in base_stats_defense:
        game_stat_col = f'away_defense_{stat}'
        opponent_avg_col = f'avg_offense_{stat}_gained_exp_lag1_home_perspective' # Home team's lagged OFFENSE GAINED
        league_avg_col = f'league_avg_def_{stat}_hybrid_lag1' # Use the HYBRID league average
        adj_col_name = f'{adj_prefix}{game_stat_col}'

        if all(c in df.columns for c in [game_stat_col, opponent_avg_col, league_avg_col]):
          df[adj_col_name] = df[game_stat_col] - df[opponent_avg_col] + df[league_avg_col]
        # else: print(f"Skipping {adj_col_name}: Missing columns")

    print("\nHybrid Opponent Adjustments Applied.")
    return df
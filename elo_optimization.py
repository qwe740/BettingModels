import sqlite3
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import optuna # pip install optuna
from elo_model import connect_db, load_data

# --- Database Path ---
DB_PATH = "cfb_data.db"

# --- Elo Calculation Functions (Modified to accept params) ---

# Keep calculate_mov_multiplier and calculate_expected_score as before,
# assuming the constants are defined globally or passed down.

def run_elo_calculation(games_df, rp_df, params):
    """Calculates Elo ratings iteratively using provided parameters."""
    # Extract parameters from the dictionary
    INITIAL_ELO_FBS = params['INITIAL_ELO_FBS']
    INITIAL_ELO_FCS = params['INITIAL_ELO_FCS']
    HFA = params['HFA']
    K_FACTOR = params['K_FACTOR']
    MEAN_REGRESSION_TARGET_FBS = params['MEAN_REGRESSION_TARGET_FBS']
    MEAN_REGRESSION_TARGET_FCS = params['MEAN_REGRESSION_TARGET_FCS']
    SEASON_REGRESSION_BASE = params['SEASON_REGRESSION_BASE']
    RETURNING_PROD_FACTOR = params['RETURNING_PROD_FACTOR']
    RP_METRIC = params['RP_METRIC']
    DEFAULT_RP = params['DEFAULT_RP']
    MOV_DENOMINATOR_BASE = params['MOV_DENOMINATOR_BASE']
    MOV_ELO_DIFF_SENSITIVITY = params['MOV_ELO_DIFF_SENSITIVITY']
    ELO_SCALING_FACTOR = params['ELO_SCALING_FACTOR'] # Likely keep fixed at 400
    ELO_EXP_BASE = params['ELO_EXP_BASE']           # Likely keep fixed at 10

    # --- Local Helper Functions using Params ---
    def get_initial_rating(is_fbs):
        return INITIAL_ELO_FBS if is_fbs else INITIAL_ELO_FCS

    def get_regression_target(is_fbs):
        return MEAN_REGRESSION_TARGET_FBS if is_fbs else MEAN_REGRESSION_TARGET_FCS

    def calculate_mov_multiplier(margin, elo_diff):
        abs_margin = abs(margin)
        effective_margin = abs_margin # Simplification for now
        log_margin_factor = math.log(max(1, effective_margin) + 1)
        elo_adjustment_factor = MOV_DENOMINATOR_BASE / ( (elo_diff * MOV_ELO_DIFF_SENSITIVITY) + MOV_DENOMINATOR_BASE )
        return log_margin_factor * elo_adjustment_factor

    def calculate_expected_score(rating_home, rating_away):
        exponent = (rating_away - rating_home) / ELO_SCALING_FACTOR
        return 1.0 / (1.0 + ELO_EXP_BASE**exponent)
    # --- End Local Helpers ---

    elo_ratings = {}
    pre_game_elos = []
    current_season = -1

    # Use progress bar only if not running many Optuna trials to avoid clutter
    iterator = games_df.iterrows() # tqdm(games_df.iterrows(), total=games_df.shape[0], leave=False) if verbose else games_df.iterrows()

    for index, game in iterator:
        season = game['season']
        home_team = game['home_team']
        away_team = game['away_team']
        home_score = game['home_points']
        away_score = game['away_points']
        neutral_site = game['neutral_site'] == 1
        home_is_fbs = game['home_is_fbs']
        away_is_fbs = game['away_is_fbs']

        if season != current_season:
            # --- Season Transition Logic (Uses params) ---
            # print(f"DEBUG: Entering Season {season}") # Optional debug
            current_season = season
            new_ratings = {}
            teams_processed = set()
            for team, current_rating in elo_ratings.items():
                 # Rough FBS/FCS check (can be improved)
                 is_fbs = get_regression_target(True) < current_rating + (INITIAL_ELO_FBS - INITIAL_ELO_FCS)/2
                 target_rating = get_regression_target(is_fbs)
                 try:
                     rp_value = rp_df.loc[(season, team), RP_METRIC]
                     if pd.isna(rp_value): rp_value = DEFAULT_RP
                 except KeyError:
                     rp_value = DEFAULT_RP

                 regression_factor = SEASON_REGRESSION_BASE + (1.0 - SEASON_REGRESSION_BASE) * (rp_value - DEFAULT_RP) * (RETURNING_PROD_FACTOR / DEFAULT_RP)
                 regression_factor = max(0.0, min(1.0, regression_factor))

                 new_rating = target_rating + regression_factor * (current_rating - target_rating)
                 new_ratings[team] = new_rating
                 teams_processed.add(team)

            # Apply default regression to any missed teams
            for team, rating in elo_ratings.items():
                if team not in teams_processed:
                     is_fbs = get_regression_target(True) < rating + (INITIAL_ELO_FBS - INITIAL_ELO_FCS)/2
                     target_rating = get_regression_target(is_fbs)
                     new_ratings[team] = target_rating + SEASON_REGRESSION_BASE * (rating - target_rating)

            elo_ratings = new_ratings
            # print(f"DEBUG: Finished regression for {season}, {len(elo_ratings)} teams") # Optional

        # --- Get Pre-Game Ratings ---
        home_rating = elo_ratings.get(home_team, get_initial_rating(home_is_fbs))
        away_rating = elo_ratings.get(away_team, get_initial_rating(away_is_fbs))

        pre_game_elos.append({
            'game_id': game['id'],
            'home_pregame_elo': home_rating,
            'away_pregame_elo': away_rating
        })

        # --- Calculate Elo Update (Uses params) ---
        rating_diff = home_rating - away_rating
        home_rating_adj = home_rating + (0 if neutral_site else HFA)
        away_rating_adj = away_rating # HFA is applied to home team adjustment
        rating_diff_adj = home_rating_adj - away_rating_adj # Recalculate adjusted difference

        expected_home = calculate_expected_score(home_rating_adj, away_rating_adj) # Use adjusted ratings

        margin = home_score - away_score
        actual_home = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5

        mov_mult = calculate_mov_multiplier(margin, rating_diff_adj) # Pass adjusted difference

        update = K_FACTOR * mov_mult * (actual_home - expected_home)

        elo_ratings[home_team] = home_rating + update
        elo_ratings[away_team] = away_rating - update

        # Add new teams if they appeared first time mid-season
        # (Should primarily happen in first season)
        if home_team not in elo_ratings: elo_ratings[home_team] = home_rating + update
        if away_team not in elo_ratings: elo_ratings[away_team] = away_rating - update

    pre_game_elo_df = pd.DataFrame(pre_game_elos)
    return pre_game_elo_df # Return only the pre-game Elos for evaluation

# --- Optuna Objective Function ---
def objective(trial, games_df_static, rp_df_static):
    """Objective function for Optuna to minimize."""

    # --- Define Search Space for Parameters ---
    params = {
        # Core Elo Params
        'HFA': trial.suggest_float('HFA', -200.0, 200.0),
        'K_FACTOR': trial.suggest_float('K_FACTOR', 1.0, 100.0),
        'ELO_SPREAD_DIVISOR': trial.suggest_float('ELO_SPREAD_DIVISOR', 1.0, 100.0), # Key new param!

        # Regression Params
        'SEASON_REGRESSION_BASE': trial.suggest_float('SEASON_REGRESSION_BASE', 0.1, 0.9),
        'RETURNING_PROD_FACTOR': trial.suggest_float('RETURNING_PROD_FACTOR', 0.05, 0.95),
        'DEFAULT_RP': 0.5, # Keep fixed or make tunable later if needed

        # MoV Multiplier Params
        'MOV_DENOMINATOR_BASE': trial.suggest_float('MOV_DENOMINATOR_BASE', 1.0, 10.0),
        'MOV_ELO_DIFF_SENSITIVITY': trial.suggest_float('MOV_ELO_DIFF_SENSITIVITY', 0.0001, 0.01),

        # Fixed/Less Critical Params (Keep fixed initially)
        'INITIAL_ELO_FBS': 1500,
        'INITIAL_ELO_FCS': 1200,
        'MEAN_REGRESSION_TARGET_FBS': 1500,
        'MEAN_REGRESSION_TARGET_FCS': 1200,
        'RP_METRIC': 'usage', # Or 'percentPPA'
        'ELO_SCALING_FACTOR': 400.0,
        'ELO_EXP_BASE': 10.0,
    }

    # --- Run Elo Calculation ---
    # Make copies to avoid modifying the static DFs if run_elo adds columns etc.
    # games_run_df = games_df_static.copy()
    # rp_run_df = rp_df_static.copy()
    # Pass the static DFs directly if run_elo doesn't modify them inplace
    pre_game_elo_df = run_elo_calculation(games_df_static, rp_df_static, params)

    # --- Evaluate vs Closing Spread ---
    # Merge results back - ensure games_df_static has 'id' and 'avg_closing_spread'
    eval_df = pd.merge(games_df_static[['id', 'avg_closing_spread', 'neutral_site']],
                       pre_game_elo_df,
                       left_on='id', right_on='game_id', how='inner') # Inner join ensures we have Elo

    # Drop games missing closing spread for evaluation
    eval_df.dropna(subset=['avg_closing_spread', 'home_pregame_elo', 'away_pregame_elo'], inplace=True)

    if eval_df.empty:
        # Handle cases where no valid data remains (e.g., first trial on very small dataset)
        return float('inf') # Return a large value to indicate failure

    # Calculate predicted spread in market points
    hfa_adj = np.where(eval_df['neutral_site'] == 1, 0, params['HFA'])
    eval_df['predicted_spread_market'] = (eval_df['away_pregame_elo'] - eval_df['home_pregame_elo'] + hfa_adj) / params['ELO_SPREAD_DIVISOR']

    # Calculate RMSE
    rmse = np.sqrt(np.mean((eval_df['predicted_spread_market'] - eval_df['avg_closing_spread'])**2))

    # Handle potential NaN/Inf if calculation goes wrong
    if np.isnan(rmse) or np.isinf(rmse):
        return float('inf')

    return rmse


# --- Main Execution for Tuning ---
if __name__ == "__main__":
    conn = connect_db()
    # Load data ONCE outside the objective function
    games_df_static, rp_df_static = load_data(conn)
    conn.close() # Close connection after loading

    print(f"\nStarting Optuna optimization for {len(games_df_static)} games...")

    # Create study
    study = optuna.create_study(direction='minimize')

    # Run optimization
    # n_trials: Start with ~50-100, increase if needed. More trials = longer runtime.
    N_TRIALS = 1000
    study.optimize(lambda trial: objective(trial, games_df_static, rp_df_static),
                   n_trials=N_TRIALS,
                   show_progress_bar=True) # Requires tqdm installed

    # --- Output Results ---
    print("\nOptimization Finished.")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial (RMSE): {study.best_value:.4f}")

    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # --- Re-run with Best Params for Detailed Evaluation (Optional) ---
    print("\nEvaluating best parameters...")
    best_params = {**study.best_params, **{ # Add back the fixed params
        'INITIAL_ELO_FBS': 1500, 'INITIAL_ELO_FCS': 1200,
        'MEAN_REGRESSION_TARGET_FBS': 1500, 'MEAN_REGRESSION_TARGET_FCS': 1200,
        'RP_METRIC': 'usage', 'DEFAULT_RP': 0.5,
        'ELO_SCALING_FACTOR': 400.0, 'ELO_EXP_BASE': 10.0
    }}

    final_pre_game_elos = run_elo_calculation(games_df_static, rp_df_static, best_params)
    final_eval_df = pd.merge(games_df_static[['id', 'avg_closing_spread', 'neutral_site', 'season', 'week']],
                             final_pre_game_elos,
                             left_on='id', right_on='game_id', how='inner')
    final_eval_df.dropna(subset=['avg_closing_spread', 'home_pregame_elo', 'away_pregame_elo'], inplace=True)

    hfa_adj = np.where(final_eval_df['neutral_site'] == 1, 0, best_params['HFA'])
    final_eval_df['predicted_spread_market'] = (final_eval_df['away_pregame_elo'] - final_eval_df['home_pregame_elo'] + hfa_adj) / best_params['ELO_SPREAD_DIVISOR']

    final_rmse = np.sqrt(np.mean((final_eval_df['predicted_spread_market'] - final_eval_df['avg_closing_spread'])**2))
    final_mae = np.mean(np.abs(final_eval_df['predicted_spread_market'] - final_eval_df['avg_closing_spread']))
    final_bias = np.mean(final_eval_df['predicted_spread_market'] - final_eval_df['avg_closing_spread'])
    final_correlation = final_eval_df['predicted_spread_market'].corr(final_eval_df['avg_closing_spread'])

    print(f"\nFinal Evaluation Metrics (using best params):")
    print(f"  RMSE: {final_rmse:.4f}")
    print(f"  MAE:  {final_mae:.4f}")
    print(f"  Bias: {final_bias:.4f}")
    print(f"  Corr: {final_correlation:.4f}")

    # You might want to save study results or the best parameters
    import joblib
    joblib.dump(study, "elo_tuning_study.pkl")
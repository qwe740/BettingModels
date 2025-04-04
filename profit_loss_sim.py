import sqlite3
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from elo_model import connect_db, load_data
from elo_optimization import run_elo_calculation
import joblib


# Assuming Optuna ran and 'study' object holds the best results
# Or, manually define best_params if you saved them elsewhere
# Example:
# best_params = {
#     'HFA': 68.5, 'K_FACTOR': 22.1, 'ELO_SPREAD_DIVISOR': 27.3,
#     'SEASON_REGRESSION_BASE': 0.75, 'RETURNING_PROD_FACTOR': 0.55,
#     'DEFAULT_RP': 0.5, 'MOV_DENOMINATOR_BASE': 2.1,
#     'MOV_ELO_DIFF_SENSITIVITY': 0.0012, 'INITIAL_ELO_FBS': 1500,
#     'INITIAL_ELO_FCS': 1200, 'MEAN_REGRESSION_TARGET_FBS': 1500,
#     'MEAN_REGRESSION_TARGET_FCS': 1200, 'RP_METRIC': 'usage',
#     'ELO_SCALING_FACTOR': 400.0, 'ELO_EXP_BASE': 10.0
# }
# --- !!! IMPORTANT: Replace above with your actual best_params !!! ---
# If you have the 'study' object from the previous run:
# best_params = study.best_params
# # Add back fixed params if they weren't stored in study.best_params
# fixed_params = {
#     'INITIAL_ELO_FBS': 1500, 'INITIAL_ELO_FCS': 1200,
#     'MEAN_REGRESSION_TARGET_FBS': 1500, 'MEAN_REGRESSION_TARGET_FCS': 1200,
#     'RP_METRIC': 'usage', 'DEFAULT_RP': 0.5,
#     'ELO_SCALING_FACTOR': 400.0, 'ELO_EXP_BASE': 10.0
# }
# best_params.update({k: v for k, v in fixed_params.items() if k not in best_params})


# --- Constants for Betting Simulation ---
BET_THRESHOLD = 0.5 # Bet only if predicted spread differs by more than this amount
WIN_PAYOUT = 0.909    # Units won on a winning bet (assuming -110 odds)
LOSS_AMOUNT = 1.0   # Units lost on a losing bet (assuming -110 odds)
PUSH_PAYOUT = 0.0   # Units won/lost on a push


# --- Betting Simulation Function ---
def simulate_betting(games_with_elo, params):
    """Simulates the betting strategy and calculates P/L."""
    print("\nSimulating betting strategy...")
    results = []
    HFA = params['HFA']
    ELO_SPREAD_DIVISOR = params['ELO_SPREAD_DIVISOR']

    # Ensure necessary columns exist and drop rows with missing critical data for simulation
    required_cols = ['id', 'season', 'week', 'home_team', 'away_team', 'home_points', 'away_points',
                     'avg_opening_spread', 'neutral_site', 'home_pregame_elo', 'away_pregame_elo']
    sim_df = games_with_elo[required_cols].copy()
    sim_df.dropna(subset=['avg_opening_spread', 'home_pregame_elo', 'away_pregame_elo',
                           'home_points', 'away_points'], inplace=True)

    total_games_evaluated = len(sim_df)
    print(f"Evaluating {total_games_evaluated} games with opening spreads and Elo ratings.")

    bet_count = 0
    for index, game in tqdm(sim_df.iterrows(), total=sim_df.shape[0], desc="Simulating Bets"):
        # Calculate predicted spread in market points
        hfa_adj = 0 if game['neutral_site'] == 1 else HFA
        predicted_spread_market = (game['away_pregame_elo'] - game['home_pregame_elo'] + hfa_adj) / ELO_SPREAD_DIVISOR

        opening_spread = game['avg_opening_spread'] # Already checked for NaN above

        # Determine bet based on strategy
        bet_on = None # 'home', 'away', or None
        profit_loss = 0.0
        result = None # 'win', 'loss', 'push', 'no_bet'

        if (predicted_spread_market > opening_spread) and (abs(predicted_spread_market-opening_spread) > BET_THRESHOLD):
            bet_on = 'away'
            bet_count += 1
        elif (predicted_spread_market < opening_spread) and (abs(predicted_spread_market-opening_spread) > BET_THRESHOLD):
            bet_on = 'home'
            bet_count += 1
        else:
            result = 'no_bet'

        # Grade the bet if one was placed
        if bet_on:
            actual_margin = game['away_points'] - game['home_points'] # Home Margin
            if bet_on == 'away':
                # Away covers if (away_points - home_points + opening_spread) > 0
                # Or equivalently: (home_points - away_points) < opening_spread
                if actual_margin > opening_spread:
                    profit_loss = WIN_PAYOUT
                    result = 'win'
                elif actual_margin < opening_spread:
                    profit_loss = -LOSS_AMOUNT
                    result = 'loss'
                else:
                    profit_loss = PUSH_PAYOUT
                    result = 'push'
            elif bet_on == 'home':
                # Home covers if (home_points - away_points + opening_spread) > 0
                if actual_margin < opening_spread:
                    profit_loss = WIN_PAYOUT
                    result = 'win'
                elif actual_margin > opening_spread:
                    profit_loss = -LOSS_AMOUNT
                    result = 'loss'
                else:
                    profit_loss = PUSH_PAYOUT
                    result = 'push'

        results.append({
            'game_id': game['id'],
            'season': game['season'],
            'week': game['week'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'opening_spread': opening_spread,
            'predicted_spread_market': predicted_spread_market,
            'bet_on': bet_on,
            'result': result,
            'profit_loss': profit_loss
        })

    print(f"Simulation complete. Placed {bet_count} bets out of {total_games_evaluated} evaluated games.")
    return pd.DataFrame(results)

# --- Main Execution ---
if __name__ == "__main__":
    # --- !!! Replace with your actual best_params from Optuna !!! ---
    study = joblib.load("elo_tuning_study.pkl")
    best_params = {**study.best_params, **{ # Add back the fixed params
        'INITIAL_ELO_FBS': 1500, 'INITIAL_ELO_FCS': 1200,
        'MEAN_REGRESSION_TARGET_FBS': 1500, 'MEAN_REGRESSION_TARGET_FCS': 1200,
        'RP_METRIC': 'usage', 'DEFAULT_RP': 0.5,
        'ELO_SCALING_FACTOR': 400.0, 'ELO_EXP_BASE': 10.0
    }}
    # ---------------------------------------------------------------

    conn = connect_db()
    games_df_static, rp_df_static = load_data(conn)
    conn.close()

    # 1. Run Elo with Best Params
    print("Running final Elo calculation with best parameters...")
    final_pre_game_elos = run_elo_calculation(games_df_static, rp_df_static, best_params)

    # 2. Merge Data
    games_with_elo = pd.merge(games_df_static, final_pre_game_elos,
                              left_on='id', right_on='game_id', how='left')

    # 3. Simulate Betting
    betting_results_df = simulate_betting(games_with_elo, best_params)

    # 4. Aggregate and Report
    print("\n--- Betting Performance Summary ---")

    # Overall Performance
    total_bets = len(betting_results_df[betting_results_df['bet_on'].notna()])
    total_wins = len(betting_results_df[betting_results_df['result'] == 'win'])
    total_losses = len(betting_results_df[betting_results_df['result'] == 'loss'])
    total_pushes = len(betting_results_df[betting_results_df['result'] == 'push'])
    total_units = betting_results_df['profit_loss'].sum()

    if total_bets > 0:
        win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
        roi = total_units / (total_bets * LOSS_AMOUNT) * 100 # ROI based on total risked amount
        print("\nOverall Results:")
        print(f"  Total Bets:   {total_bets}")
        print(f"  Wins:         {total_wins}")
        print(f"  Losses:       {total_losses}")
        print(f"  Pushes:       {total_pushes}")
        print(f"  Win Rate:     {win_rate:.2%}")
        print(f"  Total Units:  {total_units:+.2f}")
        print(f"  ROI:          {roi:.2f}%")
    else:
        print("\nNo bets were placed based on the strategy.")

    # Performance by Season
    seasonal_summary = betting_results_df.groupby('season')['profit_loss'].agg(
        total_units='sum',
        number_of_bets=lambda x: (x != 0).sum() # Count non-zero P/L as bets
        # number_of_bets = lambda x: x[betting_results_df.loc[x.index, 'bet_on'].notna()].count() # Alternative count
    ).reset_index()

    # Calculate seasonal wins/losses/pushes if needed (more complex groupby)
    seasonal_results_count = betting_results_df[betting_results_df['bet_on'].notna()].groupby(['season', 'result']).size().unstack(fill_value=0)
    seasonal_summary = pd.merge(seasonal_summary, seasonal_results_count, on='season', how='left').fillna(0)


    print("\nResults by Season:")
    # Display settings for better table view
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 100)
    print(seasonal_summary.to_string(index=False, float_format="{:.2f}".format))

    # Optional: Save detailed betting results
    # betting_results_df.to_csv("elo_betting_simulation_results.csv", index=False)
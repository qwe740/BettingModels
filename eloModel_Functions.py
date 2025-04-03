import sqlite3
import pandas as pd
import numpy as np
import math

# --- Configuration Parameters (Tunable) ---
INITIAL_ELO_FBS = 1500
INITIAL_ELO_FCS = 1200
HFA = 65  # Home Field Advantage in Elo points (common value, needs tuning)
K_FACTOR = 25 # Elo K-factor (controls update magnitude)
MEAN_REGRESSION_TARGET_FBS = 1500
MEAN_REGRESSION_TARGET_FCS = 1200
# % of previous season's rating gap (vs mean) to keep. 0 = full reset, 1 = no regression
SEASON_REGRESSION_BASE = 0.60
# How much returning production % influences the regression factor.
# E.g., RP_FACTOR=0.4 means a team with 100% RP keeps an extra 40% of its rating gap,
# a team with 50% RP keeps an extra 20%, 0% RP keeps 0% extra.
RETURNING_PROD_FACTOR = 0.40
# Use 'percentPPA' or 'usage' from returning_production? Let's start with usage.
RP_METRIC = 'usage'
# Fallback RP value if missing for a team (treat as average)
DEFAULT_RP = 0.5
#MOV scalars for tuning
MOV_DENOMINATOR_BASE = 2.2 # Part of the MoV multiplier calculation
MOV_ELO_DIFF_SENSITIVITY = 0.001 # Part of the MoV multiplier calculation (adjusts effect based on elo diff)
# Fundamental Constants for ELO Models
ELO_SCALING_FACTOR = 400.0 # Standard Elo scaling factor
ELO_EXP_BASE = 10.0        # Standard Elo exponent base

# --- Database Connection ---
DB_PATH = "cfb_data.db"

def connect_db(db_path=DB_PATH):
    """Connects to the SQLite database."""
    return sqlite3.connect(db_path)

def load_data(conn):
    """Loads games and returning production data."""
    print("Loading data from database...")
    games_query = "SELECT * FROM games_full ORDER BY season, week, id"
    games_df = pd.read_sql_query(games_query, conn)

    rp_query = f"SELECT season, team, conference, {RP_METRIC} FROM returning_production"
    rp_df = pd.read_sql_query(rp_query, conn)
    print(f"Loaded {len(games_df)} games and {len(rp_df)} returning production records.")

    # Basic preprocessing
    games_df['completed'] = games_df['completed'].astype(bool)
    games_df = games_df[games_df['completed']].copy() # Only use completed games
    games_df = games_df.drop(columns=['home_pregame_elo','home_postgame_elo','away_pregame_elo','away_postgame_elo'])

    # Identify FCS teams (adjust logic if division column isn't reliable)
    # Simple approach: Assume 'fbs' division exists, others are FCS. Needs verification.
    games_df['home_is_fbs'] = games_df['home_division'] == 'fbs'
    games_df['away_is_fbs'] = games_df['away_division'] == 'fbs'

    # Prepare RP data for lookup
    rp_df = rp_df[['season', 'team', RP_METRIC]].set_index(['season', 'team'])

    return games_df, rp_df

def get_initial_rating(is_fbs):
    """Returns the appropriate initial rating based on FBS status."""
    return INITIAL_ELO_FBS if is_fbs else INITIAL_ELO_FCS

def get_regression_target(is_fbs):
    """Returns the appropriate regression target based on FBS status."""
    return MEAN_REGRESSION_TARGET_FBS if is_fbs else MEAN_REGRESSION_TARGET_FCS

def calculate_mov_multiplier(margin, elo_diff):
    """Calculates a multiplier based on margin of victory, adjusted for elo difference."""
    # Simple log-based multiplier - needs tuning!
    # Stronger teams winning big get less boost than weaker teams winning big
    # Weaker teams losing close get less penalty than stronger teams losing close
    abs_margin = abs(margin)
    # Adjust expected margin based on elo diff (rough approximation)

    '''
    We will not include this for now as it is based on a rough heuristic
    expected_margin_adjustment = elo_diff / 28 # ~28 elo points = 1 point spread'
    '''

    # Use effective margin relative to expectation
    effective_margin = abs_margin

    log_margin_factor = math.log(max(1,effective_margin) + 1)

    # Adjust factor based on elo difference
    # Reduces the multiplier slightly when elo_diff is large (expected blowouts)
    elo_adjustment_factor = MOV_DENOMINATOR_BASE / ( ( elo_diff * MOV_ELO_DIFF_SENSITIVITY ) + MOV_DENOMINATOR_BASE )

    return log_margin_factor * elo_adjustment_factor
    # This formula is from FiveThirtyEight's NBA Elo, may need significant tuning for CFB

def calculate_expected_score(rating_home, rating_away):
    """Calculates the expected score (win probability) for the home team."""
    return 1.0 / (1.0 + 10.0**((rating_away - rating_home) / 400.0))

def run_elo_calculation(games_df, rp_df):
    """Calculates Elo ratings iteratively through the game data."""
    print("Calculating Elo ratings...")
    elo_ratings = {} # Dictionary to store current Elo rating {team: rating}
    pre_game_elos = [] # List to store ratings *before* each game

    current_season = -1

    for index, game in games_df.iterrows():
        season = game['season']
        week = game['week']
        home_team = game['home_team']
        away_team = game['away_team']
        home_score = game['home_points']
        away_score = game['away_points']
        neutral_site = game['neutral_site'] == 1
        home_is_fbs = game['home_is_fbs']
        away_is_fbs = game['away_is_fbs']

        # --- Season Transition Logic ---
        if season != current_season:
            print(f"\nProcessing start of Season {season}...")
            current_season = season
            new_ratings = {}
            teams_processed = set()

            # Apply regression and RP adjustment
            for team, current_rating in elo_ratings.items():
                # Determine if team is FBS/FCS (using last known status - might need refinement)
                # A better way might be to store FBS status alongside rating
                # For simplicity, let's guess based on last game's division or initial rating target
                # This is imperfect - ideally, have a master team list with division status per season
                is_fbs = get_regression_target(True) < current_rating + (INITIAL_ELO_FBS - INITIAL_ELO_FCS)/2
                target_rating = get_regression_target(is_fbs)

                # Get Returning Production for *this* season
                try:
                    rp_value = rp_df.loc[(season, team), RP_METRIC]
                    if pd.isna(rp_value):
                        rp_value = DEFAULT_RP
                except KeyError:
                    rp_value = DEFAULT_RP # Team not found in RP data for this season

                # Calculate regression factor influenced by RP
                # Higher RP means factor is closer to 1 (less regression)
                regression_factor = SEASON_REGRESSION_BASE + (1.0 - SEASON_REGRESSION_BASE) * (rp_value - DEFAULT_RP) * (RETURNING_PROD_FACTOR / DEFAULT_RP)
                # Clamp factor between 0 and 1
                regression_factor = max(0.0, min(1.0, regression_factor))

                # Apply regression: Keep 'regression_factor' % of the gap from the mean target
                new_rating = target_rating + regression_factor * (current_rating - target_rating)
                new_ratings[team] = new_rating
                teams_processed.add(team)

            # Add any teams from the previous rating set that somehow weren't processed
            # (Shouldn't happen with current logic but good failsafe)
            for team, rating in elo_ratings.items():
                 if team not in teams_processed:
                     is_fbs = get_regression_target(True) < rating + (INITIAL_ELO_FBS - INITIAL_ELO_FCS)/2
                     target_rating = get_regression_target(is_fbs)
                     new_ratings[team] = target_rating + SEASON_REGRESSION_BASE * (rating - target_rating) # Default regression

            elo_ratings = new_ratings
            print(f"Regressed ratings for {len(elo_ratings)} teams using RP metric '{RP_METRIC}'.")

        # --- Get Pre-Game Ratings ---
        home_rating = elo_ratings.get(home_team, get_initial_rating(home_is_fbs))
        away_rating = elo_ratings.get(away_team, get_initial_rating(away_is_fbs))

        # Store pre-game ratings
        pre_game_elos.append({
            'game_id': game['id'],
            'season': season,
            'week': week,
            'home_team': home_team,
            'home_pregame_elo': home_rating,
            'away_team': away_team,
            'away_pregame_elo': away_rating
        })

        # --- Calculate Elo Update ---
        rating_diff = home_rating - away_rating
        # Apply HFA
        if not neutral_site:
            home_rating_adj = home_rating + HFA
            rating_diff_adj = rating_diff + HFA
        else:
            home_rating_adj = home_rating
            rating_diff_adj = rating_diff

        # Expected outcome
        expected_home = calculate_expected_score(home_rating_adj, away_rating)

        # Actual outcome
        margin = home_score - away_score
        if margin > 0:
            actual_home = 1.0
        elif margin < 0:
            actual_home = 0.0
        else:
            actual_home = 0.5 # Tie

        # Calculate MoV multiplier
        mov_mult = calculate_mov_multiplier(margin, rating_diff_adj) # Use adjusted diff for MoV scaling

        # Elo update calculation
        update = K_FACTOR * mov_mult * (actual_home - expected_home)

        # Update ratings
        elo_ratings[home_team] = home_rating + update
        elo_ratings[away_team] = away_rating - update

        # Ensure new teams that played get added to the main dict if they weren't regressed
        if home_team not in elo_ratings: elo_ratings[home_team] = home_rating + update
        if away_team not in elo_ratings: elo_ratings[away_team] = away_rating - update


    print("Elo calculation complete.")
    # Convert pre-game Elo list to DataFrame
    pre_game_elo_df = pd.DataFrame(pre_game_elos)
    return elo_ratings, pre_game_elo_df
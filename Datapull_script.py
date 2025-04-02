#Cell 1
import requests
import pandas as pd
import sqlite3
import numpy as np

#Cell 2
#API Setup
BASE_URL = 'https://api.collegefootballdata.com/'
API_KEY = 'Y2P4Ex6vaj/fPBURQsf2jz+0R2pXikYv8PtvqoqiMG7ukTvpVscCVjUA10VDv+My'

def get_data(endpoint,params={}):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params)
    return response.json()

#Cell 3
# Initial Data Pull (Games)
games_data = []
for year in range(2013, 2025):
    data = get_data("games", {"year": year, "division": "fbs"})
    games_data.extend(data)
games_df = pd.DataFrame(games_data)
games_df = games_df.drop(['home_line_scores','away_line_scores'], axis=1)
games_df.head()  # Quick check

#Cell 4
#SQLite Setup
conn = sqlite3.connect("cfb_data.db")
games_df.to_sql("games", conn, if_exists="replace", index=False)

#Cell 5
#Pull Lines Data
lines_data = []
for year in range(2013, 2025):
    #print(f"Pulling lines for {year}...")
    data = get_data("lines", {"year": year, "division": "fbs"})
    lines_data.extend(data)

# Flatten the nested structure
flat_lines = []
for game in lines_data:
    game_id = game['id']
    home_team = game['homeTeam']
    away_team = game['awayTeam']
    if game['lines']:  # Check if lines exist
        for line in game['lines']:
            flat_lines.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'sportsbook': line['provider'],
                'spread_open': line.get('spreadOpen', None), #Opening spread
                'spread': line.get('spread', None),  # Closing spread
                'overUnder_open': line.get('overUnderOpen'), #Opening total
                'overUnder': line.get('overUnder', None) #Closing total
            })

lines_df = pd.DataFrame(flat_lines)

#Cell 6
#Save to SQLite
conn = sqlite3.connect("cfb_data.db")
lines_df.to_sql("lines", conn, if_exists="replace", index=False)
conn.close()

#Cell 7
# Average Spreads and Totals and Update Games Table
conn = sqlite3.connect("cfb_data.db")

# Query to average closing spreads
query = """
SELECT 
    g.*, 
    AVG(l.spread) AS avg_closing_spread,
    AVG(l.overUnder) AS avg_closing_total
FROM 
    games g
LEFT JOIN 
    lines l ON g.id = l.game_id
GROUP BY 
    g.id, g.season, g.week, g.start_date, g.home_team, g.home_points, 
    g.away_team, g.away_points
"""

# Load into DataFrame
merged_df = pd.read_sql_query(query, conn)

# Overwrite games table with new data (no spread_open)
merged_df.to_sql("games", conn, if_exists="replace", index=False)


conn.close()

#Cell 8
# Cell 8: Pull Advanced Stats
advanced_stats_data = []
for year in range(2013, 2025):
    #print(f"Pulling advanced stats for {year}...")
    data = get_data("stats/game/advanced", {"year": year, "excludeGarbageTime":"true", "division": "fbs"})
    advanced_stats_data.extend(data)

# Create DataFrame
advanced_stats_df = pd.DataFrame(advanced_stats_data)

#Cell 9
# Flatten the nested JSON
flat_stats = []
for game in advanced_stats_data:
    row = {
        'gameId': game['gameId'],
        'week': game['week'],
        'team': game['team'],
        'opponent': game['opponent']
    }
    # Flatten top-level offense stats
    for key, value in game['offense'].items():
        if isinstance(value, dict):  # Subkeys like standardDowns
            for subkey, subvalue in value.items():
                row[f'offense_{key}_{subkey}'] = subvalue
        else:
            row[f'offense_{key}'] = value
    # Flatten top-level defense stats
    for key, value in game['defense'].items():
        if isinstance(value, dict):  # Subkeys like passingPlays
            for subkey, subvalue in value.items():
                row[f'defense_{key}_{subkey}'] = subvalue
        else:
            row[f'defense_{key}'] = value
    flat_stats.append(row)


# Create DataFrame
advanced_stats_df = pd.DataFrame(flat_stats)

#Cell 10
# Save to SQLite
conn = sqlite3.connect("cfb_data.db")
advanced_stats_df.to_sql("advanced_stats", conn, if_exists="replace", index=False)
conn.close()

#Cell 11
# Merge Games and Advanced Stats (All Columns)
conn = sqlite3.connect("cfb_data.db")

query = """
SELECT 
    g.*,
    h.offense_plays AS home_offense_plays,
    h.offense_drives AS home_offense_drives,
    h.offense_ppa AS home_offense_ppa,
    h.offense_totalPPA AS home_offense_totalPPA,
    h.offense_successRate AS home_offense_successRate,
    h.offense_explosiveness AS home_offense_explosiveness,
    h.offense_powerSuccess AS home_offense_powerSuccess,
    h.offense_stuffRate AS home_offense_stuffRate,
    h.offense_lineYards AS home_offense_lineYards,
    h.offense_lineYardsTotal AS home_offense_lineYardsTotal,
    h.offense_secondLevelYards AS home_offense_secondLevelYards,
    h.offense_secondLevelYardsTotal AS home_offense_secondLevelYardsTotal,
    h.offense_openFieldYards AS home_offense_openFieldYards,
    h.offense_openFieldYardsTotal AS home_offense_openFieldYardsTotal,
    h.offense_standardDowns_ppa AS home_offense_standardDowns_ppa,
    h.offense_standardDowns_successRate AS home_offense_standardDowns_successRate,
    h.offense_standardDowns_explosiveness AS home_offense_standardDowns_explosiveness,
    h.offense_passingDowns_ppa AS home_offense_passingDowns_ppa,
    h.offense_passingDowns_successRate AS home_offense_passingDowns_successRate,
    h.offense_passingDowns_explosiveness AS home_offense_passingDowns_explosiveness,
    h.offense_rushingPlays_ppa AS home_offense_rushingPlays_ppa,
    h.offense_rushingPlays_totalPPA AS home_offense_rushingPlays_totalPPA,
    h.offense_rushingPlays_successRate AS home_offense_rushingPlays_successRate,
    h.offense_rushingPlays_explosiveness AS home_offense_rushingPlays_explosiveness,
    h.offense_passingPlays_ppa AS home_offense_passingPlays_ppa,
    h.offense_passingPlays_totalPPA AS home_offense_passingPlays_totalPPA,
    h.offense_passingPlays_successRate AS home_offense_passingPlays_successRate,
    h.offense_passingPlays_explosiveness AS home_offense_passingPlays_explosiveness,
    h.defense_plays AS home_defense_plays,
    h.defense_drives AS home_defense_drives,
    h.defense_ppa AS home_defense_ppa,
    h.defense_totalPPA AS home_defense_totalPPA,
    h.defense_successRate AS home_defense_successRate,
    h.defense_explosiveness AS home_defense_explosiveness,
    h.defense_powerSuccess AS home_defense_powerSuccess,
    h.defense_stuffRate AS home_defense_stuffRate,
    h.defense_lineYards AS home_defense_lineYards,
    h.defense_lineYardsTotal AS home_defense_lineYardsTotal,
    h.defense_secondLevelYards AS home_defense_secondLevelYards,
    h.defense_secondLevelYardsTotal AS home_defense_secondLevelYardsTotal,
    h.defense_openFieldYards AS home_defense_openFieldYards,
    h.defense_openFieldYardsTotal AS home_defense_openFieldYardsTotal,
    h.defense_standardDowns_ppa AS home_defense_standardDowns_ppa,
    h.defense_standardDowns_successRate AS home_defense_standardDowns_successRate,
    h.defense_standardDowns_explosiveness AS home_defense_standardDowns_explosiveness,
    h.defense_passingDowns_ppa AS home_defense_passingDowns_ppa,
    h.defense_passingDowns_successRate AS home_defense_passingDowns_successRate,
    h.defense_passingDowns_explosiveness AS home_defense_passingDowns_explosiveness,
    h.defense_rushingPlays_ppa AS home_defense_rushingPlays_ppa,
    h.defense_rushingPlays_totalPPA AS home_defense_rushingPlays_totalPPA,
    h.defense_rushingPlays_successRate AS home_defense_rushingPlays_successRate,
    h.defense_rushingPlays_explosiveness AS home_defense_rushingPlays_explosiveness,
    h.defense_passingPlays_ppa AS home_defense_passingPlays_ppa,
    h.defense_passingPlays_totalPPA AS home_defense_passingPlays_totalPPA,
    h.defense_passingPlays_successRate AS home_defense_passingPlays_successRate,
    h.defense_passingPlays_explosiveness AS home_defense_passingPlays_explosiveness,
    a.offense_plays AS away_offense_plays,
    a.offense_drives AS away_offense_drives,
    a.offense_ppa AS away_offense_ppa,
    a.offense_totalPPA AS away_offense_totalPPA,
    a.offense_successRate AS away_offense_successRate,
    a.offense_explosiveness AS away_offense_explosiveness,
    a.offense_powerSuccess AS away_offense_powerSuccess,
    a.offense_stuffRate AS away_offense_stuffRate,
    a.offense_lineYards AS away_offense_lineYards,
    a.offense_lineYardsTotal AS away_offense_lineYardsTotal,
    a.offense_secondLevelYards AS away_offense_secondLevelYards,
    a.offense_secondLevelYardsTotal AS away_offense_secondLevelYardsTotal,
    a.offense_openFieldYards AS away_offense_openFieldYards,
    a.offense_openFieldYardsTotal AS away_offense_openFieldYardsTotal,
    a.offense_standardDowns_ppa AS away_offense_standardDowns_ppa,
    a.offense_standardDowns_successRate AS away_offense_standardDowns_successRate,
    a.offense_standardDowns_explosiveness AS away_offense_standardDowns_explosiveness,
    a.offense_passingDowns_ppa AS away_offense_passingDowns_ppa,
    a.offense_passingDowns_successRate AS away_offense_passingDowns_successRate,
    a.offense_passingDowns_explosiveness AS away_offense_passingDowns_explosiveness,
    a.offense_rushingPlays_ppa AS away_offense_rushingPlays_ppa,
    a.offense_rushingPlays_totalPPA AS away_offense_rushingPlays_totalPPA,
    a.offense_rushingPlays_successRate AS away_offense_rushingPlays_successRate,
    a.offense_rushingPlays_explosiveness AS away_offense_rushingPlays_explosiveness,
    a.offense_passingPlays_ppa AS away_offense_passingPlays_ppa,
    a.offense_passingPlays_totalPPA AS away_offense_passingPlays_totalPPA,
    a.offense_passingPlays_successRate AS away_offense_passingPlays_successRate,
    a.offense_passingPlays_explosiveness AS away_offense_passingPlays_explosiveness,
    a.defense_plays AS away_defense_plays,
    a.defense_drives AS away_defense_drives,
    a.defense_ppa AS away_defense_ppa,
    a.defense_totalPPA AS away_defense_totalPPA,
    a.defense_successRate AS away_defense_successRate,
    a.defense_explosiveness AS away_defense_explosiveness,
    a.defense_powerSuccess AS away_defense_powerSuccess,
    a.defense_stuffRate AS away_defense_stuffRate,
    a.defense_lineYards AS away_defense_lineYards,
    a.defense_lineYardsTotal AS away_defense_lineYardsTotal,
    a.defense_secondLevelYards AS away_defense_secondLevelYards,
    a.defense_secondLevelYardsTotal AS away_defense_secondLevelYardsTotal,
    a.defense_openFieldYards AS away_defense_openFieldYards,
    a.defense_openFieldYardsTotal AS away_defense_openFieldYardsTotal,
    a.defense_standardDowns_ppa AS away_defense_standardDowns_ppa,
    a.defense_standardDowns_successRate AS away_defense_standardDowns_successRate,
    a.defense_standardDowns_explosiveness AS away_defense_standardDowns_explosiveness,
    a.defense_passingDowns_ppa AS away_defense_passingDowns_ppa,
    a.defense_passingDowns_successRate AS away_defense_passingDowns_successRate,
    a.defense_passingDowns_explosiveness AS away_defense_passingDowns_explosiveness,
    a.defense_rushingPlays_ppa AS away_defense_rushingPlays_ppa,
    a.defense_rushingPlays_totalPPA AS away_defense_rushingPlays_totalPPA,
    a.defense_rushingPlays_successRate AS away_defense_rushingPlays_successRate,
    a.defense_rushingPlays_explosiveness AS away_defense_rushingPlays_explosiveness,
    a.defense_passingPlays_ppa AS away_defense_passingPlays_ppa,
    a.defense_passingPlays_totalPPA AS away_defense_passingPlays_totalPPA,
    a.defense_passingPlays_successRate AS away_defense_passingPlays_successRate,
    a.defense_passingPlays_explosiveness AS away_defense_passingPlays_explosiveness
FROM 
    games g
LEFT JOIN 
    advanced_stats h ON g.id = h.gameId AND g.home_team = h.team
LEFT JOIN 
    advanced_stats a ON g.id = a.gameId AND g.away_team = a.team
"""

# Load into DataFrame
games_with_stats_df = pd.read_sql_query(query, conn)

# Save as new table
games_with_stats_df.to_sql("games_with_stats", conn, if_exists="replace", index=False)


conn.close()

#Cell 12
#Conversion Helper Function
def time_to_seconds(time_str):
    if time_str is None or time_str == '':
        return None
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except (ValueError, AttributeError):
        return None

# Pull Games/Teams Data (Turnovers and PossessionTime Only)
games_teams_data = []
for year in range(2013, 2025):
    for week in range(1, 17):  # Weeks 1-15
        #print(f"Pulling games/teams for {year}, Week {week}...")
        data = get_data("games/teams", {"year": year, "week": week, "division": "fbs"})
        games_teams_data.extend(data)

# Flatten and convert possessionTime to seconds
flat_teams = []
for game in games_teams_data:
    game_id = game['id']
    teams = game['teams']
    if len(teams) == 2:  # Ensure both teams are present
        home_team = next(t for t in teams if t['homeAway'] == 'home')
        away_team = next(t for t in teams if t['homeAway'] == 'away')
        
        row = {'gameId': game_id}
        # Home team stats
        home_stats = {stat['category']: stat['stat'] for stat in home_team['stats']}
        row['home_turnovers'] = home_stats.get('turnovers', None)
        row['home_possessionTime'] = time_to_seconds(home_stats.get('possessionTime', None))
        # Away team stats
        away_stats = {stat['category']: stat['stat'] for stat in away_team['stats']}
        row['away_turnovers'] = away_stats.get('turnovers', None)
        row['away_possessionTime'] = time_to_seconds(away_stats.get('possessionTime', None))
        flat_teams.append(row)


# Create DataFrame
teams_stats_df = pd.DataFrame(flat_teams)

#Cell 13
# Save to SQLite
conn = sqlite3.connect("cfb_data.db")
teams_stats_df.to_sql("teams_stats", conn, if_exists="replace", index=False)
conn.close()

#Cell 14
# Merge with Games_with_Stats (Specific Stats)
conn = sqlite3.connect("cfb_data.db")

query = """
SELECT 
    g.*,
    t.home_turnovers AS home_turnovers,
    t.home_possessionTime AS home_possessionTime,
    t.away_turnovers AS away_turnovers,
    t.away_possessionTime AS away_possessionTime
FROM 
    games_with_stats g
LEFT JOIN 
    teams_stats t ON g.id = t.gameId
"""

# Load and save
games_full_df = pd.read_sql_query(query, conn)
games_full_df.to_sql("games_full", conn, if_exists="replace", index=False)

conn.close()
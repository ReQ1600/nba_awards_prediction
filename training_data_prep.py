import pandas as pd
import time
import unicodedata
from nba_api.stats.endpoints import leaguedashplayerstats

SEASON_DATA_BEGIN = 2014
SEASON_DATA_END = 2025
SEASONS = [f"{year}-{str(year+1)[2:]}" for year in range(SEASON_DATA_BEGIN, SEASON_DATA_END)]

ALL_NBA_TARGETS_FILE = "basketball-reference_all_nba_2014-2025_rewards.csv"
ALL_ROOKIE_TARGETS_FILE = "basketball-reference_all_rookie_2014-2025_rewards.csv"

score_map_all_nba = {"1st": 3, "2nd": 2, "3rd": 1}
score_map_all_rookie = {"1st": 2, "2nd": 1}

def normalize_name(name):
    name = name.lower().strip()
    
    name = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    
    return name

def parseTarget(target_src, score_map, remove_position = True):
    data = []
    
    df_target = pd.read_csv(target_src)
    for _, row in df_target.iterrows():
        season = row['Season']
        team_rank = row['Tm']
        score = score_map.get(team_rank, 0)
        
        players = row.values[4:]
        for p in players:
            #remove position
            clean_name = " ".join(p.split()[:-1]) if remove_position else p
            data.append([season, clean_name, score])

    return data

def fetchSeasonData(season):
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame', measure_type_detailed_defense='Base')

    df_stats = stats.get_data_frames()[0]
    time.sleep(1)

    advanced_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame', measure_type_detailed_defense='Advanced')
    df_advanced = advanced_stats.get_data_frames()[0]

    cols = df_advanced.columns.difference(df_stats.columns).to_list()
    cols.append('PLAYER_ID')
    df_season = pd.merge(df_stats, df_advanced[cols], on='PLAYER_ID')
    
    df_season['SEASON_ID'] = season
    print(f"parsed season {season}")

    return df_season

every_season_data = []
for season in SEASONS:
    try:
        every_season_data.append(fetchSeasonData(season))
    except Exception as e:
        print(f"exception occured during {season} season fetching")

final_df = pd.concat(every_season_data, ignore_index=True)
final_df['PLAYER_NAME_CLEAN'] = final_df['PLAYER_NAME'].apply(normalize_name)

all_nba_targets = parseTarget(ALL_NBA_TARGETS_FILE, score_map_all_nba)
df_all_nba_targets = pd.DataFrame(all_nba_targets, columns=['SEASON_ID', 'PLAYER_NAME', 'ALL_NBA_TARGET_SCORE'])
df_all_nba_targets['PLAYER_NAME_CLEAN'] = df_all_nba_targets['PLAYER_NAME'].apply(normalize_name)

final_df = pd.merge(final_df, df_all_nba_targets[['SEASON_ID', 'PLAYER_NAME_CLEAN', 'ALL_NBA_TARGET_SCORE']], on=["SEASON_ID", "PLAYER_NAME_CLEAN"], how="left")
final_df["ALL_NBA_TARGET_SCORE"] = final_df["ALL_NBA_TARGET_SCORE"].fillna(0)

all_rookie_targets = parseTarget(ALL_ROOKIE_TARGETS_FILE, score_map_all_rookie, False)
df_all_rookie_targets = pd.DataFrame(all_rookie_targets, columns=['SEASON_ID', 'PLAYER_NAME', 'ALL_ROOKIE_TARGET_SCORE'])
df_all_rookie_targets['PLAYER_NAME_CLEAN'] = df_all_rookie_targets['PLAYER_NAME'].apply(normalize_name)

final_df = pd.merge(final_df, df_all_rookie_targets[['SEASON_ID', 'PLAYER_NAME_CLEAN', 'ALL_ROOKIE_TARGET_SCORE']], on=["SEASON_ID", "PLAYER_NAME_CLEAN"], how="left")
final_df["ALL_ROOKIE_TARGET_SCORE"] = final_df["ALL_ROOKIE_TARGET_SCORE"].fillna(0)

final_df.to_csv(f"nba_{SEASON_DATA_BEGIN}-{SEASON_DATA_END}_season_train_data")

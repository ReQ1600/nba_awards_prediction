import pandas as pd
import time
import unicodedata
from nba_api.stats.endpoints import leaguedashplayerstats

SEASON = "2025-26"

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

data = []
try:
    data.append(fetchSeasonData(SEASON))
except Exception as e:
    print(f"exception occured during {SEASON} season fetching")

final_df = pd.concat(data, ignore_index=True)
final_df['PLAYER_NAME_CLEAN'] = final_df['PLAYER_NAME'].apply(normalize_name)

final_df.to_csv(f"nba_{SEASON}_season_prediction_data")

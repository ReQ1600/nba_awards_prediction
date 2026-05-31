import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
import json
import sys

outfile = sys.argv[1]
print(f"outfile: {outfile}")

TRAINING_DATA_FILE = "nba_2014-2025_season_train_data"
PREDICTION_DATA_FILE = "nba_2025-26_season_prediction_data"
SEASON_TO_IGNORE_ROOKIES = "2014-15" # the first season in dataset

def scale_nba_data(df, features):
    scaler = StandardScaler()
    df_scaled = df.copy()
    
    df_scaled[features] = df_scaled[features].astype(float)

    #per season
    for SEASON in df_scaled['SEASON_ID'].unique():
        mask = df_scaled['SEASON_ID'] == SEASON
        if mask.any():
            df_scaled.loc[mask, features] = scaler.fit_transform(df_scaled.loc[mask, features])
            
    return df_scaled

nba_train_df = pd.read_csv(TRAINING_DATA_FILE)
nba_predict_df = pd.read_csv(PREDICTION_DATA_FILE)

print(f"training: {TRAINING_DATA_FILE}\nprediction: {PREDICTION_DATA_FILE}")

id_cols = ['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID', 'TEAM_ABBREVIATION', 'PLAYER_NAME_CLEAN']

target_cols = ['ALL_NBA_TARGET_SCORE', 'ALL_ROOKIE_TARGET_SCORE']

features = [
    'AGE', 'GP', 'W_PCT', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV', 'PF', 'PLUS_MINUS',
    'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'TS_PCT', 'PIE', 'USG_PCT', 
    'AST_PCT', 'REB_PCT', 'EFG_PCT'
]

all_nba_train_df = scale_nba_data(nba_train_df, features)
all_nba_predict_df = scale_nba_data(nba_predict_df, features)

# training and test data preparation
training_cols_to_drop = [
    'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'PLAYER_NAME_CLEAN', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_ID',
    'ALL_NBA_TARGET_SCORE', 'ALL_ROOKIE_TARGET_SCORE'
]
predict_cols_to_drop = [
    'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'PLAYER_NAME_CLEAN', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_ID'
]

all_nba_X_train = all_nba_train_df.drop(columns=training_cols_to_drop)
all_nba_y_train = all_nba_train_df["ALL_NBA_TARGET_SCORE"]

#find players first season in dataset to filterout non rookies
first_seasons = all_nba_train_df.groupby('PLAYER_ID')['SEASON_ID'].min().reset_index()
first_seasons.columns = ['PLAYER_ID', 'FIRST_SEASON']

df_with_rookie_info = all_nba_train_df.merge(first_seasons, on='PLAYER_ID')

all_nba_X_predict = all_nba_predict_df.drop(columns=predict_cols_to_drop)

all_rookie_mask = (df_with_rookie_info["SEASON_ID"] == df_with_rookie_info["FIRST_SEASON"]) \
    & (df_with_rookie_info["SEASON_ID"] != SEASON_TO_IGNORE_ROOKIES)

all_rookie_train_df = df_with_rookie_info[all_rookie_mask]

training_cols_to_drop_all_rookie = [
    'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'PLAYER_NAME_CLEAN', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_ID',
    'ALL_NBA_TARGET_SCORE', 'ALL_ROOKIE_TARGET_SCORE', 'FIRST_SEASON'
]

predict_cols_to_drop_all_rookie = [
    'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'PLAYER_NAME_CLEAN', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_ID'
]

all_rookie_X_train = all_rookie_train_df.drop(columns=training_cols_to_drop_all_rookie)
all_rookie_y_train = all_rookie_train_df["ALL_ROOKIE_TARGET_SCORE"]

historical_players = set(all_nba_train_df["PLAYER_ID"].unique())
all_rookie_predict_mask = ~all_nba_predict_df["PLAYER_ID"].isin(historical_players)
all_rookie_predict_df = all_nba_predict_df[all_rookie_predict_mask].copy()

all_rookie_X_predict = all_rookie_predict_df.drop(columns=predict_cols_to_drop_all_rookie)


# all_nba
model_all_nba = VotingRegressor([
    ('gb', GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42)),
])
model_all_nba.fit(all_nba_X_train, all_nba_y_train)

all_nba_y_pred = model_all_nba.predict(all_nba_X_predict)

all_nba_results = pd.DataFrame({
    'Player': all_nba_predict_df['PLAYER_NAME'],
    'score_halu': all_nba_y_pred
})

top_predictions = all_nba_results.sort_values(by="score_halu", ascending=False)
top = top_predictions.head(15)
output_all_nba = {
    "first all-nba team": top.iloc[0:5]['Player'].tolist(),
    "second all-nba team": top.iloc[5:10]['Player'].tolist(),
    "third all-nba team": top.iloc[10:15]['Player'].tolist(),
}
print("top 15 all_nba")
print(top_predictions.head(15))

# all_rookie
model_all_rookie = VotingRegressor([
    ('gb', GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=21)),
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42)),
])

model_all_rookie.fit(all_rookie_X_train, all_rookie_y_train)

all_rookie_y_pred = model_all_rookie.predict(all_rookie_X_predict)

all_rookie_results = pd.DataFrame({
    'Player': all_rookie_predict_df['PLAYER_NAME'],
    'score_halu': all_rookie_y_pred
})

top_predictions = all_rookie_results.sort_values(by="score_halu", ascending=False)
top = top_predictions.head(10)
output_all_rookie = {
    "first rookie all-nba team": top.iloc[0:5]['Player'].tolist(),
    "second rookie all-nba team": top.iloc[5:10]['Player'].tolist(),
}


with open(outfile, "w") as f:
    json.dump({**output_all_nba, **output_all_rookie}, f, indent=2)

print("top 10 all_rookie")
print(top_predictions.head(10))


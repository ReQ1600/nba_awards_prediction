import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import json
import sys

outfile = sys.argv[1]
print(f"outfile: {outfile}")

DATA_FILE = "nba_2014-2026_season_train_data"
SEASON_TO_IGNORE_ROOKIES = "2014-15" # the first SEASON in dataset

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

# def drop_correlated(threshold, df):
#     corr_matrix = df.corr(numeric_only=True).abs()

#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

#     to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

#     return df.drop(columns=to_drop)

nba_df = pd.read_csv(DATA_FILE)
print(f"working with {DATA_FILE} df")

#scale data for every SEASON
id_cols = ['PLAYER_ID', 'PLAYER_NAME', 'SEASON_ID', 'TEAM_ABBREVIATION', 'PLAYER_NAME_CLEAN']

target_cols = ['ALL_NBA_TARGET_SCORE', 'ALL_ROOKIE_TARGET_SCORE']

features = [
    'AGE', 'GP', 'W_PCT', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV', 'PF', 'PLUS_MINUS',
    'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'TS_PCT', 'PIE', 'USG_PCT', 
    'AST_PCT', 'REB_PCT', 'EFG_PCT'
]

#noise reduction
# filtered = nba_df[nba_df['GP'] >= 65].copy()

# filtered = filtered.sort_values('GP', ascending=False)
# filtered = filtered.drop_duplicates(subset=['PLAYER_NAME_CLEAN', 'SEASON_ID'], keep='first')

scaled_df = scale_nba_data(nba_df, features)
SEASON = "2025-26"


r2_final = 0
mse_final = 0

# training and test data preparation
cols_to_drop = [
    'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'PLAYER_NAME_CLEAN', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_ID',
    'ALL_NBA_TARGET_SCORE', 'ALL_ROOKIE_TARGET_SCORE'
]

test_mask = scaled_df["SEASON_ID"] == SEASON
# train_mask = filtered_df["SEASON_ID"].isin(["2023-24", "2022-23"])

all_nba_train_df = scaled_df[~test_mask]
all_nba_test_df = scaled_df[test_mask]

all_nba_X_train = all_nba_train_df.drop(columns=cols_to_drop)
all_nba_y_train = all_nba_train_df["ALL_NBA_TARGET_SCORE"]

all_nba_X_test = all_nba_test_df.drop(columns=cols_to_drop)
all_nba_y_test = all_nba_test_df["ALL_NBA_TARGET_SCORE"]

# cols_to_remove = drop_correlated(0.95, all_nba_X_train)
# all_nba_X_train = all_nba_X_train.drop(columns=cols_to_remove)
# all_nba_X_test = all_nba_X_test.drop(columns=cols_to_remove)

#find first players first SEASON in dataset to filterout non rookies
first_seasons = scaled_df.groupby('PLAYER_ID')['SEASON_ID'].min().reset_index()
first_seasons.columns = ['PLAYER_ID', 'FIRST_SEASON']

df_with_rookie_info = scaled_df.merge(first_seasons, on='PLAYER_ID')

all_rookie_test_mask = df_with_rookie_info["SEASON_ID"] == SEASON
all_rookie_rookie_mask = (df_with_rookie_info["SEASON_ID"] == df_with_rookie_info["FIRST_SEASON"]) \
    & (df_with_rookie_info["SEASON_ID"] != SEASON_TO_IGNORE_ROOKIES)

all_rookie_train_df = df_with_rookie_info[~all_rookie_test_mask & all_rookie_rookie_mask]
all_rookie_test_df = df_with_rookie_info[all_rookie_test_mask & all_rookie_rookie_mask]

cols_to_drop_all_rookie = [
    'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'PLAYER_NAME_CLEAN', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_ID',
    'ALL_NBA_TARGET_SCORE', 'ALL_ROOKIE_TARGET_SCORE', 'FIRST_SEASON'
]

all_rookie_X_train = all_rookie_train_df.drop(columns=cols_to_drop_all_rookie)
all_rookie_y_train = all_rookie_train_df["ALL_ROOKIE_TARGET_SCORE"]

all_rookie_X_test = all_rookie_test_df.drop(columns=cols_to_drop_all_rookie)
all_rookie_y_test = all_rookie_test_df["ALL_ROOKIE_TARGET_SCORE"]

# cols_to_remove = drop_correlated(0.95, all_rookie_X_train)
# all_rookie_X_train = all_rookie_X_train.drop(columns=cols_to_remove)
# all_rookie_X_test = all_rookie_X_test.drop(columns=cols_to_remove)


#trainig
# model = RandomForestRegressor(random_state=42, n_jobs=-1)
# model = GradientBoostingRegressor(
#     n_estimators=500,
#     max_depth=4,
#     learning_rate=0.05,
#     subsample=0.8,
#     random_state=42
# )
# model = Lasso(alpha=0.01)
# model = SVR(kernel='rbf', C=700, epsilon=0.1)

# model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)

# all_nba
model_all_nba = VotingRegressor([
    ('gb', GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)),
    # ('rf', RandomForestRegressor(n_estimators=500, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42)),
])
model_all_nba.fit(all_nba_X_train, all_nba_y_train)

all_nba_y_pred = model_all_nba.predict(all_nba_X_test)

mse = mean_squared_error(all_nba_y_test, all_nba_y_pred)
r2 = r2_score(all_nba_y_test, all_nba_y_pred)

mse_final += mse
r2_final += r2

print(f"dla gbet {SEASON}:\n\tmse:{mse}\n\tr2:{r2}")

all_nba_results = pd.DataFrame({
    'Player': all_nba_test_df['PLAYER_NAME'],
    'score_rl': all_nba_y_test,
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
    # ('rf', RandomForestRegressor(n_estimators=500, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42)),
])

model_all_rookie.fit(all_rookie_X_train, all_rookie_y_train)

all_rookie_y_pred = model_all_rookie.predict(all_rookie_X_test)

mse = mean_squared_error(all_rookie_y_test, all_rookie_y_pred)
r2 = r2_score(all_rookie_y_test, all_rookie_y_pred)
print(f"dla all_rookie gbet {SEASON}:\n\tmse:{mse}\n\tr2:{r2}")

all_rookie_results = pd.DataFrame({
    'Player': all_rookie_test_df['PLAYER_NAME'],
    'score_rl': all_rookie_y_test,
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


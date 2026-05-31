# nba_awards_prediction
All-NBA and All-Rookie rewards prediction done with machine learning
## Instalation
```
pip install -r requirements.txt
```

## Files
- `training_data_prep.py` - prepare data for training

- `prediction_data_prep.py`- prepare data for prediction

- `train_and_predict.py`- train and predict nba awards. Outputs result into a json file given as the first arg

- `train_and_predict.py`- train, predict and measure score of prediction of nba awards. Outputs result into a json file given as the first arg

## Data sources
- [nba_prediction](https://github.com/swar/nba_api) - player data
- [basketball reference](https://www.basketball-reference.com/) - reward data

## Packages used
- numpy
- pandas
- scikit-learn
- scipy
- nba_api
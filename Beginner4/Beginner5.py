import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings # 実行に関係ない警告を無視
warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 範囲ごとに分類
def func(df):
    if 15 < df < 20:
        return 0
    elif 20 <= df <= 25:
        return 1
    elif 25 < df <= 30:
        return 2
    elif 30 < df < 40:
        return 3
    else:
        return 4


# データの整頓
train['group'] = train['mpg'].apply(func)

train.replace({'horsepower': {'?': np.nan}}, inplace=True)
train['horsepower'] = train['horsepower'].astype(float)
test.replace({'horsepower': {'?': np.nan}}, inplace=True)
test['horsepower'] = test['horsepower'].astype(float)
train['horsepower'].fillna(train['horsepower'].mean(), inplace=True)
test['horsepower'].fillna(train['horsepower'].mean(), inplace=True)

train.drop(columns=['car name', 'weight', 'acceleration', 'mpg'], inplace=True)
test.drop(columns=['car name', 'weight', 'acceleration'], inplace=True)
categorical_feature = ['model year', 'cylinders', 'displacement', 'horsepower', 'origin']

# split
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train, train['group'], test_size=0.3, random_state=0, stratify=train['group']) # <- 連続値には不可

y_true = X_valid['group']
X_train.drop(columns='group', inplace=True)
X_valid.drop(columns='group', inplace=True)

# learning
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_feature)

param = { 'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass', # 目的 : 多クラス分類
        'num_class': 5, # クラス数 : 3
        'metric': {'multi_error'}
        }

model = lgb.train(param, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=100)

y_pred = model.predict(test, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred, axis=1) # 一番大きい予測確率のクラスを予測クラスに

Id = test.id.astype(int)
my_solution = pd.DataFrame(y_pred, Id, columns=['group'])

# return median(?)
def func(df):
    if df == 0:
        return 17.513410
    elif df == 1:
        return 22.812471
    elif df == 2:
        return 28.316270
    elif df == 3:
        return 35.681927
    else:
        return 44.033390


my_solution['mpg'] = my_solution['group'].apply(func)

my_solution.drop('group', axis=1, inplace=True)

# my_solution.to_csv("my_prediction_data.csv", header=False)

# y_pred_pre = model.predict(X_valid, num_iteration=model.best_iteration)
# mean_absolute_error(y_true, y_pred_pre)

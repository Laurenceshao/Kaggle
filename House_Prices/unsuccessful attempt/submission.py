# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import pandas_profiling as pp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', 2000)
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df.head()

y = df[['Id', 'SalePrice']]

# inplace = True?
df = df.drop(['SalePrice'], axis = 1)

all_df = [df, test_df]
all_df = pd.concat(all_df).reset_index(drop = True)

# very useful code
all_df['BsmtCond'] = all_df['BsmtCond'].fillna(all_df['BsmtCond'].mode()[0])
all_df['BsmtQual'] = all_df['BsmtQual'].fillna(all_df['BsmtQual'].mode()[0])

# df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
all_df['GarageType'] = all_df['GarageType'].fillna(all_df['GarageType'].mode()[0])

all_df['GarageFinish'] = all_df['GarageFinish'].fillna(all_df['GarageFinish'].mode()[0])
all_df['GarageQual'] = all_df['GarageQual'].fillna(all_df['GarageQual'].mode()[0])
all_df['GarageCond'] = all_df['GarageCond'].fillna(all_df['GarageCond'].mode()[0])

all_df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)

# needs to be fully thought over
all_df.drop(['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis = 1, inplace = True)

all_df.drop(['Id'], axis = 1, inplace = True)

all_df['MasVnrType'].fillna(value='None',inplace=True)
all_df['MasVnrArea'].fillna(0,inplace=True)


# needs to be fully thought over, why do they put mean value here?
all_df['LotFrontage'] = all_df['LotFrontage'].fillna(all_df['LotFrontage'].mean())
# needs to be fully thought over
all_df.drop(['GarageYrBlt'], axis = 1, inplace = True)

all_df = pd.get_dummies(all_df)

Scaler = StandardScaler()
all_scaled = pd.DataFrame(Scaler.fit_transform(all_df))


train_scaled = pd.DataFrame(all_scaled[:1460])
test_scaled = pd.DataFrame(all_scaled[1460:2919])

# copied
X = train_scaled
X_train, X_test, y_train, y_test = train_test_split(X, y['SalePrice'], test_size=0.1, random_state=42)

from xgboost import XGBRegressor
XGB = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)
XGB.fit(X_train,y_train)

print ("Training score:",XGB.score(X_train,y_train),"Test Score:",XGB.score(X_test,y_test))

y_pred = pd.DataFrame(XGB.predict(test_scaled))
y_pred['Id'] = test_df['Id']
y_pred['SalePrice'] = y_pred[0]
y_pred.drop(0,axis=1,inplace=True)

y_pred.to_csv('House_Prices.csv', index = False)
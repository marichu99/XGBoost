import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


data = pd.read_csv('melb_data.csv')
cols_to_use=["Rooms","Distance","Landsize","BuildingArea","YearBuilt"]
y=data.Price
x=data[cols_to_use]

# perform splits
X_train,X_valid,y_train,y_valid=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)

# instantiate the xgboost model
xgb=XGBRegressor(n_estimators=200)

xgb.fit(X_train,y_train,early_stopping_rounds=5,verbose=False)

preds=xgb.predict(X=X_valid)

mae=mean_absolute_error(y_true=y_valid,y_pred=preds)
print(mae)


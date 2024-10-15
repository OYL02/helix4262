# 1. Importing relevant modules
import pandas as pd
import joblib 

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint


# 2. Importing training sets 
X_train = pd.read_csv()
y_train = pd.read_csv()
y_train = y_train['label']


# 3. Creation of model
xgboost = XGBClassifier()

# With hyperparameter tuning: Use random search to find the best hyperparameters
param_dist = {'n_estimators': randint(50, 500),
              'max_depth': randint(1, 20)}
random_search = RandomizedSearchCV(xgboost,
                                   param_distributions = param_dist, 
                                   n_iter = 10, 
                                   cv = 10)
random_search.fit(X_train, y_train)


# 4. Save the trained model 
joblib.dump(random_search, 'xgboost_model.pkl')
print('Model Saved')
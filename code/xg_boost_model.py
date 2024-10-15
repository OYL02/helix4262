# 1. Importing relevant modules
import pandas as pd
import numpy as np
import random
import joblib 

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV 


# 2. Importing training sets 
X_train = pd.read_csv()
y_train = pd.read_csv()


# 3. Creation of model
xgboost = XGBClassifier()

# With hyperparameter tuning: Use random search to find the best hyperparameters
param_dist = {'n_estimators': random.randint(50, 500),
              'max_depth': random.randint(1, 20)}
random_search = RandomizedSearchCV(xgboost,
                                   param_distributions = param_dist, 
                                   n_iter = 10, 
                                   cv = 10)
random_search.fit(X_train, y_train)


# 4. Save the trained model 
joblib.dump(random_search, 'xgboost_model.pkl')
print('Model Saved')
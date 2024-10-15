# 1. Importing relevant modules
import pandas as pd
import joblib 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint


# 2. Importing training sets 
X_train = pd.read_csv()
y_train = pd.read_csv()
y_train = y_train['label']


# 3. Creation of model
# Intiailise a random forest classifier
rf = RandomForestClassifier()

# With hyperparameter tuning: Use random search to find the best hyperparameters
param_dist = {'n_estimators': randint(50, 500),
              'max_depth': randint(1, 20)}
random_search = RandomizedSearchCV(rf,
                                   param_distributions = param_dist, 
                                   n_iter = 10, 
                                   cv = 10)
#print('Best hyperparameters:', random_search.best_params_)
random_search.fit(X_train, y_train)
# rf.fit(X_train, y_train)


# 4. Save the trained model 
joblib.dump(random_search, 'random_forest_model.pkl')
print('Model Saved')
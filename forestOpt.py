import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from Intro import X_train, X_valid, y_train, y_valid , best_max_leaf_nodes

'''
Optimize the forest model to improve performance.
'''
#note: count of data = 6196

'''
#fit model
rfOpt = RandomForestRegressor(max_leaf_nodes=best_max_leaf_nodes, random_state=1)
rfOpt.fit(X_train, y_train)

#get predictions for the training and validation sets
rfOpt_train_pred = rfOpt.predict(X_train)
rfOpt_val_pred = rfOpt.predict(X_valid)

#Evaluate the model
train_r2 = r2_score(y_train, rfOpt_train_pred)
val_r2 = r2_score(y_valid, rfOpt_val_pred)

print("R^2 Score (Train):", train_r2)
print("R^2 Score (Validation):", val_r2)
#results: high train score, lower validation score --> high variance/overfitting
'''

#Optimization using randomized search
if __name__ == "__main__":
    param_distributions = {

        #search for best max leaf nodes with max known at 550 from last optimized data
        'max_leaf_nodes': np.arange(100, 575, 25).tolist(),

        #search for best min sample leaf
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],

        #search for best max features for tree diversity
        'max_features': ['sqrt', 0.3, 0.5, 0.7, 1.0],

        #search for best tree count, using high count for more variability and stability
        'n_estimators': [100, 200, 300, 400, 500]
    }

random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=1, n_jobs=1), 
    param_distributions=param_distributions, 
    n_iter=100, cv=3, scoring='r2', random_state=1, n_jobs=1)

random_search.fit(X_train, y_train)

print("Best Score:", random_search.best_score_)
print("Best Parameters:", random_search.best_params_)
'''
Optimization through one parameter: max_leaf_nodes

#Plot the validation curve
param_range = np.arange(100, 1000, 50)
train_scores, val_scores = validation_curve(
    RandomForestRegressor(min_samples_leaf=10,max_features=0.5), X_train, y_train, param_name='max_leaf_nodes', param_range=param_range,
    scoring='r2', n_jobs=1, cv=6
)

mean_train_scores = train_scores.mean(axis=1)
mean_val_scores = val_scores.mean(axis=1)


peak_val_score = mean_val_scores.max()
print("max validation score:", peak_val_score)

optimal_max_leaf_nodes = param_range[mean_val_scores.argmax()]
print("Optimal max leaf nodes:", optimal_max_leaf_nodes)
'''
'''
#Plot the validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, mean_train_scores, label='Training Score', color='blue')
plt.plot(param_range, mean_val_scores, label='Validation Score', color='orange')
plt.title('Validation Curve')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('R^2 Score')
plt.legend()
plt.grid()
plt.show()
'''
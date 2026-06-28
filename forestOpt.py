import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from Intro import X_train, X_valid, y_train, y_valid , best_max_leaf_nodes

'''
Optimize the forest model to improve performance.

Note: All functions centered around the specific practice dataset I'm on. 
Further improvements and tuning when dealing with other datasets.

Conclusion:
r2 on data with engineered features (only training) was ~0.05 less than the original features.
However, when testing r2 against the validation set, the score went up by ~0.05, 
resulting in a score closer to 0.8 or 0.1 diff from training r2. 
The engineered features performed better, indicating improved generalization.
'''
#note: count of data = 6196

#feature engineering function
def feature_engineering(df):
    #add one for zero division protection on formulas
    df_copy = df.copy()

    #spatial distribution by bathroom
    df_copy['Room_Bathroom_Ratio'] = df_copy['Rooms'] / (df_copy['Bathroom'] + 1) 

    #average room size
    df_copy['Avg_Room_Size'] = df_copy['Landsize'] / (df_copy['Rooms'] + 1)

    return df_copy

#fit and evaluate model
def fit_eval_model(X_train, y_train, X_valid, y_valid, 
                   best_mg_search):

    rfOpt = best_mg_search
    rfOpt.fit(X_train, y_train)

    #get predictions for the training and validation sets
    rfOpt_train_pred = rfOpt.predict(X_train)
    rfOpt_val_pred = rfOpt.predict(X_valid)

    #Evaluate the model
    train_r2 = r2_score(y_train, rfOpt_train_pred)
    val_r2 = r2_score(y_valid, rfOpt_val_pred)

    print("R^2 Score (Train):", train_r2)
    print("R^2 Score (Validation):", val_r2)

    return train_r2, val_r2

#pre optimized results: high train score, lower validation score --> high variance/overfitting

#Optimization using randomized search
def rs_optimization(X_train, y_train, max_nodes_init):
    param_distributions = {

        #search for best max leaf nodes with max known at 550 from last optimized data
        'max_leaf_nodes': [max_nodes_init],

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
        n_iter=30, cv=3, scoring='r2', random_state=1, n_jobs=1)

    random_search.fit(X_train, y_train)

    best_score_rs = random_search.best_score_
    best_params_rs = random_search.best_params_
    best_rs = random_search.best_estimator_
    
    print("Best Score:", best_score_rs)
    print("Best Parameters:", best_params_rs)

    return best_params_rs, best_rs

#Best Score (og features): 0.7500885065086619
#Winning Parameters (og features): {'n_estimators': 200, 'min_samples_leaf': 2, 'max_leaf_nodes': 550, 'max_features': 1.0}
#best score (engineered features): 0.7460535406487069
#winning parameters (engineered features): {'n_estimators': 100, 'min_samples_leaf': 2, 'max_leaf_nodes': np.int64(850), 'max_features': 1.0}

def mg_optimization(X_train, y_train, max_nodes_init, best_rs,
                    best_min_samples_leaf, best_max_features, best_n_estimators):
    #Micro Grid search
    param_grid = {
        'max_leaf_nodes': [max_nodes_init - 25, max_nodes_init, max_nodes_init + 25],
        'min_samples_leaf': [1, best_min_samples_leaf, best_min_samples_leaf + 1],
        'max_features': [best_max_features - 0.1, best_max_features, 1.0],
        'n_estimators': [best_n_estimators - 10, best_n_estimators, best_n_estimators + 10]
    }

    micro_grid_search = GridSearchCV(
        estimator=best_rs,
        param_grid=param_grid,
        cv=3, scoring='r2', n_jobs=1
    )

    micro_grid_search.fit(X_train, y_train)

    score_mg = micro_grid_search.best_score_
    best_params_mg = micro_grid_search.best_params_
    best_mg_search = micro_grid_search.best_estimator_

    print("Best Score after micro grid search:", score_mg)
    print("Best Parameters after micro grid search:", best_params_mg)

    return score_mg, best_params_mg, best_mg_search

#Best new score (og features): 0.7508257090876502
#Best new parameters (og features): {'max_features': 1.0, 'max_leaf_nodes': 550, 'min_samples_leaf': 3, 'n_estimators': 200}
#Best new score (engineered features): 0.7465171773791904
#Best new parameters (engineered features): {'max_features': 0.9, 'max_leaf_nodes': np.int64(875), 'min_samples_leaf': 2, 'n_estimators': 90}

#Optimization through one parameter: max_leaf_nodes
#Create the validation curve, start with this first for redefining features
def val_curve_init(X_train, y_train):
    param_range = np.arange(100, 1000, 50)
    train_scores, val_scores = validation_curve(
        RandomForestRegressor(min_samples_leaf=2,max_features=1.0, random_state=1), X_train, y_train, param_name='max_leaf_nodes', param_range=param_range,
        scoring='r2', n_jobs=1, cv=6)

    mean_train_scores = train_scores.mean(axis=1)
    mean_val_scores = val_scores.mean(axis=1)


    peak_val_score = mean_val_scores.max()
    print("max validation score:", peak_val_score)

    optimal_max_leaf_nodes = param_range[mean_val_scores.argmax()]
    print("Optimal max leaf nodes:", optimal_max_leaf_nodes)

    return param_range, mean_train_scores, mean_val_scores, optimal_max_leaf_nodes

#get new features
feat_eng_X_train = feature_engineering(X_train)
feat_eng_X_valid = feature_engineering(X_valid)

#Get optimized parameters
param_range_init, mean_train_scores_init, mean_val_scores_init, max_leaf_nodes_init = val_curve_init(feat_eng_X_train, y_train)

best_rs_params, best_rs = rs_optimization(feat_eng_X_train, y_train, max_leaf_nodes_init)

min_samples_leaf = best_rs_params['min_samples_leaf']
max_features = best_rs_params['max_features']
n_estimators = best_rs_params['n_estimators']

best_score, best_params, best_mg_search = mg_optimization(feat_eng_X_train, y_train, max_leaf_nodes_init, best_rs, 
                                          min_samples_leaf, max_features, n_estimators)

#re-eval of model using optimized parameters
train_r2, val_r2 = fit_eval_model(feat_eng_X_train, y_train, feat_eng_X_valid, y_valid, best_mg_search)

'''
#Sort feature importances
    best_rf = micro_grid_search.best_estimator_
    importances = best_rf.feature_importances_

    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    print(feature_importances)

'''
#feature_importances df:
#      Feature  Importance
#0       Rooms    0.286167
#3   Lattitude    0.260526
#4  Longtitude    0.238174
#2    Landsize    0.148464
#1    Bathroom    0.066669

'''
#Note: only 5 features used
#Use Correlation matrix to understand feature relationships, 
#whther or not i should drop lowest valued feature.
correlation_matrix = X_train.corr()
print(correlation_matrix) #Rooms to bathrooms: 0.61 --> keep in dataset for edge cases
'''

'''
#Plot the initial validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range_init, mean_train_scores_init, label='Training Score', color='blue')
plt.plot(param_range_init, mean_val_scores_init, label='Validation Score', color='orange')
plt.title('Validation Curve')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('R^2 Score')
plt.legend()
plt.grid()
plt.show()
'''
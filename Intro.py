'''
Reminder:
1. Create a new branch
git checkout -b my-feature-branch

2. Make changes to your files
echo "Some changes" >> myfile.txt

3. Stage your changes
git add myfile.txt

4. Commit your changes
git commit -m "Add some changes"

5. Push your changes to the remote repository
git push origin my-feature-branch   
'''

import pandas as pd 

melb_path_name = "/Users/shelleysugiharto/Documents/PredictiveModelling/melb_data.csv"

melbourne_data = pd.read_csv(melb_path_name)
melb_data_summary = melbourne_data.describe()

'''
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(melb_data_summary)
'''

avg_lot_size = round(melb_data_summary.loc['mean', 'Landsize'])
newest_home_age = melb_data_summary.loc['max', 'YearBuilt']
'''
print("avg lot size:", avg_lot_size)
print("newest home age:", newest_home_age)
'''

cols_melb_data = melbourne_data.columns
#print("Columns in Melbourne data:", cols_melb_data)

melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price

melbourne_features = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]
X = melbourne_features

from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
#random state set to a number in order for reproducibility
'''
Decision tree regression --> funnelling down binary paths, gets to the decision (leaf)
'''
melbourne_model.fit(X, y)
exact_prices = y.head()
predictions = melbourne_model.predict(X.head())
print("Making predictions for the following 5 house prices:")
print(X.head())
print("Predictions:", predictions)
print("Exact prices:", exact_prices) #results will =predictions because houses are already priced

'''
Model Validation:
Starting with Mean Absolute Error (MAE)...
error = actual - predicted
MAE is the avg distance between the actual and predicted values.
'''
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(exact_prices, predictions) #using js head
print("Mean Absolute Error (MAE):", mae) #would be 0 in this case because the data is static and exact

#Again, but with the full dataset
predictions_full = melbourne_model.predict(X)
mae_full = mean_absolute_error(y, predictions_full)
print("Mean Absolute Error (MAE) with full dataset:", mae_full)

'''
in-sample score was done above. This can provide inaccurate results.
To avoid this in a straightforward fashion, we can exclude some data from the model-building process.

Then the excluded data can be used for validation (validation dataset).
'''
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X_train, y_train)

val_predictions = melbourne_model.predict(X_valid)
mae_val = mean_absolute_error(y_valid, val_predictions)

print("Mean Absolute Error (MAE) with validation dataset:", mae_val)

'''
The model is not very accurate.
We can try to improve it by using a more complex model or by tuning the hyperparameters.
Underfitting and overfitting:


We will move on to a more complex model: Random Forest.
'''
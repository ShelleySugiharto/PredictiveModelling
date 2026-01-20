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




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

data = pd.read_csv('tenis.csv')

# =============================================================================
# play = data.iloc[:,-1:].values
# 
# # print(play)
# 
# le = LabelEncoder()
# 
# play[:,0] = le.fit_transform(play[:,0])
# print(play)
# 
# le = LabelEncoder()
# 
# windy = data.iloc[:,3:4].values
# # print(windy[:,0])
# windy2 = le.fit_transform(windy[:,0])
# print(windy2)
# =============================================================================

# ============================= Preprocessing =================================
data_labeled = data.apply(LabelEncoder().fit_transform)

# print(data_labeled)

outlook_encoded = data.iloc[:,:1]
ohe = OneHotEncoder()
outlook_encoded = ohe.fit_transform(outlook_encoded).toarray()
# print(outlook_encoded)

final_outlook = pd.DataFrame(data = outlook_encoded, index= range(14), columns=['o', 'r', 's'])

final_data = pd.concat([final_outlook, data_labeled.iloc[:,1:]], axis = 1)

final_train_data_part1 = pd.DataFrame(data = final_data.iloc[:, :3].values, index = range(14), columns = ['o', 'r', 's'])
final_train_data_part2 = pd.DataFrame(data = final_data.iloc[:, 5:].values, index = range(14), columns =['windy', 'play'])
final_train_data = pd.concat([final_train_data_part1, final_train_data_part2, data.iloc[:,1:2]], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(final_train_data.iloc[:,:], data.iloc[:,2:3], test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_prediction = regressor.predict(x_test)

# Sabit terim ekleme (bias)
final_train_data_with_bias = pd.concat([pd.DataFrame(np.ones((final_train_data.shape[0], 1)).astype(int), columns=['bias']), final_train_data], axis=1)

# İlgili sütunları seçme
X_l = final_train_data_with_bias.iloc[:, [ 4,6]].values

r_ols = sm.OLS(endog = data.iloc[:,1:2], exog =X_l)
r = r_ols.fit()
print(r.summary())

new_dataframe = final_train_data_with_bias.iloc[:, [4, 6]]

x_train, x_test, y_train, y_test = train_test_split(new_dataframe.iloc[:,:], data.iloc[:,2:3], test_size=0.33, random_state=0)

regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)

y_prediction2 = regressor2.predict(x_test)

# sklearn-Feature-Scaling-project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler

data = pd.read_csv("real_estate_price_size_year.csv")
data.head()

data.describe()

x = data[["size","year"]]
y = data["price"]

from sklearn.linear_model import LinearRegression
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

reg = LinearRegression()
reg.fit(x_scaled,y)

reg.intercept_

reg.coef_

reg.score(x_scaled,y)

def adj_r2(x_scaled,y):
    r2 = reg.score(x_scaled,y)
    n = x_scaled.shape[0]
    p = x_scaled.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
    
adj_r2(x_scaled,y)

#predicted price of an apartment that has a size of 750 sq.ft. from 2009.
new_data = [[750,2009]]
new_data_scaled = scaler.transform(new_data)
reg.predict(new_data_scaled)


from sklearn.feature_selection import f_regression
f_regression(x_scaled,y)

p_values = f_regression(x,y)[1]
p_values

p_values.round(3)

reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p-values'] = p_values.round(3)
reg_summary

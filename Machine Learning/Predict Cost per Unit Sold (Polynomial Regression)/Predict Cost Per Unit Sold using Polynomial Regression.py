import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cost_df = pd.read_csv("EconomiesOfScale.csv")
cost_df.describe()
cost_df.info()

#### Exploratory Data Analysis

sns.jointplot(x='Number of Units', y='Manufacturing Cost', data = cost_df)

sns.lmplot(x='Number of Units', y='Manufacturing Cost', data=cost_df)

X = cost_df[['Number of Units']]
y = cost_df['Manufacturing Cost']

# Used the entire dataset for training only 
X_train = X
y_train = y



#### Model Training and Results - Linear

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept =True)

regressor.fit(X_train,y_train)
print('Model Coefficient: ', regressor.coef_)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.ylabel('Cost Per Unit Sold [dollars]')
plt.xlabel('Number of Units [in Millions]')
plt.title('Unit Cost vs. Number of Units [in Millions](Training dataset)')



#### Model Training and Results - Linear

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)

X_columns = poly_regressor.fit_transform(X_train)

regressor = LinearRegression()
regressor.fit(X_columns, y_train)

y_predict = regressor.predict(poly_regressor.fit_transform(X_train))

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_predict, color = 'blue')
plt.ylabel('Cost Per Unit Sold [dollars]')
plt.xlabel('Number of Units [in Millions]')
plt.title('Unit Cost vs. Number of Units [in Millions](Training dataset)')



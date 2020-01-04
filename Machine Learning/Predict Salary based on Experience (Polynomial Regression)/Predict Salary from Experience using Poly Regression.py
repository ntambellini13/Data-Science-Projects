import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

salary = pd.read_csv("Employee_Salary.csv")
salary.describe()
salary.info()

#### Exploratory Data Analysis

sns.jointplot(x='Years of Experience', y='Salary', data = salary)

sns.lmplot(x='Years of Experience', y='Salary', data=salary)

X = salary[['Years of Experience']]
y = salary['Salary']

# Entire Dataset used for training
X_train = X
y_train = y



#### Model Training and Results - Linear
from sklearn.linear_model import LinearRegression

regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train,y_train)

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.ylabel('Salary/Year [dollars]')
plt.xlabel('Years of Experience')
plt.title('Salary vs. Years of Experience (Training dataset)')



#### Model Training and Results - Polynomial

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=3)

X_columns = poly_regressor.fit_transform(X_train)

regressor = LinearRegression()
regressor.fit(X_columns, y_train)

print('Model Coefficients: ', regressor.coef_)

y_predict = regressor.predict(poly_regressor.fit_transform(X_train))

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_predict, color = 'blue')
plt.ylabel('Salary/Year [dollars]')
plt.xlabel('Years of Experience')
plt.title('Salary vs. Years of Experience (Training dataset)')







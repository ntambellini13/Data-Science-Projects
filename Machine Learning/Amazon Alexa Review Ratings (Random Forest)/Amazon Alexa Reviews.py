import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_alexa = pd.read_csv('amazon_alexa.tsv', sep='\t')
df_alexa.head()

#### Exploratory Data Analysis

positive = df_alexa[df_alexa['feedback']==1]
negative = df_alexa[df_alexa['feedback']==0]

sns.countplot(df_alexa['feedback'], label = "Count") 
sns.countplot(x = 'rating', data = df_alexa)

df_alexa['rating'].hist(bins = 5)
plt.figure(figsize = (40,15))

sns.barplot(x = 'variation', y='rating', data=df_alexa, palette = 'deep')


#### Data Preprocessing
df_alexa = df_alexa.drop(['date', 'rating'],axis=1)

variation_dummies = pd.get_dummies(df_alexa['variation'], drop_first = True)
variation_dummies

df_alexa.drop(['variation'], axis=1, inplace=True)
df_alexa = pd.concat([df_alexa, variation_dummies], axis=1)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
alexa_countvectorizer = vectorizer.fit_transform(df_alexa['verified_reviews'])

print(vectorizer.get_feature_names())
print(alexa_countvectorizer.toarray())  

df_alexa.drop(['verified_reviews'], axis=1, inplace=True)
reviews = pd.DataFrame(alexa_countvectorizer.toarray())

df_alexa = pd.concat([df_alexa, reviews], axis=1)

X = df_alexa.drop(['feedback'],axis=1)
y = df_alexa['feedback']



#### Model Training

## Splitting Test and Train Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

## Build the model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

randomforest_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
randomforest_classifier.fit(X_train, y_train)



#### Evaluating the Model
y_predict_train = randomforest_classifier.predict(X_train)
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)
print(classification_report(y_train, y_predict_train))

y_predict = randomforest_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict))



#### Improving the Model
df_alexa = pd.read_csv('amazon_alexa.tsv', sep='\t')
df_alexa = pd.concat([df_alexa, pd.DataFrame(alexa_countvectorizer.toarray())], axis = 1)
df_alexa['length'] = df_alexa['verified_reviews'].apply(len)

X = df_alexa.drop(['rating', 'date', 'variation', 'verified_reviews', 'feedback'], axis = 1)
y = df_alexa['feedback']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

randomforest_classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
randomforest_classifier.fit(X_train, y_train)
y_predict = randomforest_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict))







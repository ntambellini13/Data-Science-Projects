import pandas as pd
import seaborn as sns

spam_df = pd.read_csv("emails.csv")
spam_df.head(10)
spam_df.info()

#### Exploratory Data Analysis

ham = spam_df[spam_df['spam']==0]
spam = spam_df[spam_df['spam']==1]

print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")
print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")

sns.countplot(spam_df['spam'], label = "Count") 



#### Build the Model

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])

print(vectorizer.get_feature_names())
print(spamham_countvectorizer.toarray())  



## Train the Model with Entire Dataset

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
label = spam_df['spam'].values
NB_classifier.fit(spamham_countvectorizer, label)

testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)

test_predict = NB_classifier.predict(testing_sample_countvectorizer)


## Split in Test and Train Set before Training

X = spamham_countvectorizer
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)



#### Evaluating the Model

from sklearn.metrics import classification_report, confusion_matrix

y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)

y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, average_precision_score

df = pd.read_csv('adult.data', header=None, index_col=False,  names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                                                     'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

df = df.dropna(how='any')

# based on marital-status and relationship predict work-hours

marital_status_n_rela = df['marital-status'].to_frame().join(
    df.relationship.to_frame())
print(marital_status_n_rela.head())
work_hours = df['hours-per-week']

x_train, x_test, y_train, y_test = train_test_split(
    marital_status_n_rela, work_hours, random_state=4)

# print(marital_status_n_rela.shape)
print(x_train.shape)

tfidf = TfidfVectorizer(stop_words='english', analyzer='word')

x_train_tf = tfidf.fit_transform(x_train)
x_test_tf = tfidf.transform(x_test)

print(x_train_tf.shape)

mnb = MultinomialNB()

# Train the data
mnb.fit(x_train_tf, y_train)
predict = mnb.predict(x_test_tf)

print(accuracy_score(y_test, predict))

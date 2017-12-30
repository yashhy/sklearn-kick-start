import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, average_precision_score
import pickle

df=pd.read_csv('smsspam.txt', sep='\t', names=['Status', 'Message'])

# Print the first 5 rows
# print(df.head())

# Print only spam
# print(df[df.Status=='spam'])

# Change lables Spam/Ham to 0 and 1 respectively
df.loc[df['Status']=='ham',  'Status'] = 1
df.loc[df['Status']=='spam', 'Status'] = 0

# Print only ham
# print(df[df.Status == 1])

# Seperate Data and Label seperately
data_x  = df['Message']
label_y = df['Status']

# Split the train n test sets using train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, label_y, test_size=0.2, random_state=4)

# # Splitted train and test set
# print(x_train)
# print('---------')
# print(x_test)
# print('---------')
# print(y_train)
# print('---------')
# print(y_test)
# print('---------')


# # Convert the text (in Message) to numbers for ML Algos to understand
# # so use CountVectorize to convert the text to numbers
# cv = CountVectorizer()

# # The below is the method to convert text to numbers
# # Example:

# cv_example = cv.fit_transform(["Hello world, this is Yashwanth", "Hello again! again", "loving python"])

# print(cv_example.toarray())
# print(cv.get_feature_names()) # get all words in the above gives sentenses
# arr = cv_example.toarray()

# print(cv.inverse_transform(arr[1]))  # ['again', 'hello']


# Instead of Count Vectorize use TF-IDF Vectorizer which is more advanced

tfidf = TfidfVectorizer(stop_words='english',analyzer='word',min_df=1)

x_train_tf = tfidf.fit_transform(x_train) # transform the train df
x_test_tf = tfidf.transform(x_test) # transform the test df
y_train = y_train.astype('int') # convert to int
y_test = y_test.astype('int')  # convert to int

# Use MNB Classifier
mnb=MultinomialNB()

# Use Bernoulli Classifier
bnb=BernoulliNB()

mnb.fit(x_train_tf, y_train)
predict = mnb.predict(x_test_tf)

print(average_precision_score(y_test, predict))

bnb.fit(x_train_tf, y_train)
predict = bnb.predict(x_test_tf)

print(average_precision_score(y_test, predict))

sentense = ['URGENT! You have won a 1 week FREE membership']
sentense = pd.Series(sentense)
x_test_tf = tfidf.transform(sentense)

predict = mnb.predict(x_test_tf)

print(predict)

# Pickle the trained model for future predictions
# https://stackoverflow.com/a/11218504/1778834
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

tfidf_file = open("./test-ur-sentence/tfidf.pickle", "wb")
pickle.dump(tfidf, tfidf_file)
tfidf_file.close()

mnb_pickle_file = open("./test-ur-sentence/spam_trained_model.pickle", "wb")
pickle.dump(mnb, mnb_pickle_file)
mnb_pickle_file.close()


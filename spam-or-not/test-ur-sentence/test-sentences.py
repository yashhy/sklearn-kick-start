import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

mnb_pickle_file = open("spam_trained_model.pickle", "rb")
mnb = pickle.load(mnb_pickle_file)
mnb_pickle_file.close()

train_data_file = open("train_data.pickle", "rb")
train_data = pickle.load(train_data_file)
train_data_file.close()

sentences = pd.Series([
    'hi honey', 
    'how are you?',
    'Oh k...i\'m watching here:)',
    'Free entry in 2 a wkly comp to win FA Cup final',
    'URGENT! You have won a 1 week FREE membership',
    'URGENT honey you have won a lottery!!',
    'URGENT! dear You have won a 1 week FREE membership',
    ])

tfidf = TfidfVectorizer(stop_words='english', analyzer='word', min_df=1)
tfidf.fit_transform(train_data)

x_test_tf = tfidf.transform(sentences)

predict = mnb.predict(x_test_tf)

for idx, item in np.ndenumerate(predict):
  print("Spam" if item == 0 else "Not Spam", " --> " + sentences.iloc[idx])

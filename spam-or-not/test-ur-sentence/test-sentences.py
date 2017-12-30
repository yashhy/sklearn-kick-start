import pandas as pd
import numpy as np
import pickle

mnb_pickle_file = open("spam_trained_model.pickle", "rb")
mnb = pickle.load(mnb_pickle_file)
mnb_pickle_file.close()

tfidf_file = open("tfidf.pickle", "rb")
tfidf = pickle.load(tfidf_file)
tfidf_file.close()

sentences = pd.Series([
    'hi honey', 
    'how are you?',
    'Oh k...i\'m watching here:)',
    'Free entry in 2 a wkly comp to win FA Cup final',
    'URGENT! You have won a 1 week FREE membership',
    'URGENT honey you have won a lottery!!',
    'URGENT! dear You have won a 1 week FREE membership',
    ])

x_test_tf = tfidf.transform(sentences)

predict = mnb.predict(x_test_tf)

for idx, item in np.ndenumerate(predict):
  print("Spam" if item == 0 else "Not Spam", " --> " + sentences.iloc[idx])

# This code gives an idea of how Feature Extraction works in sklearn using CountVectorizer and TfidfVectorizer
-----


## Problem Statement:

Classify whether a given text message is Spam or Not. So, here we use the data set provided by [semicolon](https://www.youtube.com/channel/UCwB7HrnRlOfasrbCJoiZ9Lg) [here](https://github.com/shreyans29/thesemicolon/blob/master/smsspam).

### Steps:
1. Libraries needed: sklearn and pandas, 
2. Read the data set using panda.read_csv()
3. After reading, change the "Status" column items, 'spam' = 0 and 'ham' = 1
4. Store the 'Messages' column as data_x and 'Status' column as 'label_y'
5. Split the data set for training and testing using sklearn's [train_test_split()](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#examples-using-sklearn-model-selection-train-test-split)
6. Uncomment the code and play with CountVectorizer
7. But we will explore more on TfidfVectorizer as this one is more powerful than CountVectorizer. Read more [here](https://www.quora.com/What-is-the-difference-between-TfidfVectorizer-and-CountVectorizer-1), [here](https://www.quora.com/How-does-TfidfVectorizer-work-in-laymans-terms)
8. Use tfidf.fit_transform(x_train) to convert 'Text' to 'Fractions' becasue thats what ML Algos understand
9. Same applies for tfidf.transform(x_test)
10. Change y_train to integers like y_train.astype('int')
11. Note: x_train, x_test, y_train, y_test are variable obtained from train_test_split()
10. Here we use MultinomialNB() classifier, import it from sklearn.naive_bayes
11. Do mnb.fit(x_train_tf, y_train)
12. Later predict using mnb.predict(x_test_tf)
13. Use average_precision_score from sklearn.metrics to see the actual predication percentage
14. Bouns: I have add prediciton percentage of BernoulliNB() classifier too



> Prediction percentage of 0.953518893211 for MultinomialNB()

> Prediction percentage of 0.966168623976 for BernoulliNB()

## References:
1. http://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/


This example is taken from the tuts by semicolon:
1. [Machine Learning with Text - Count Vectorizer](https://www.youtube.com/watch?v=RZYjsw6P4nI)
2. [Machine Learning with Text - TFIDF Vectorizer](https://www.youtube.com/watch?v=bPYJi1E9xeM)  
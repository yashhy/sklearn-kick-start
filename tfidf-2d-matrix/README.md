# This example shows how pass a training data with Two-Dimention (2D) DataFrame into fit_transform():


Dataset used : [Census Bureau Database](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/) (downloaded and available in this folder)

## Steps :
1. [Initailly](https://github.com/yashhy/sklearn-kick-start/tree/master/spam-or-not) we have seen how to pass single dimension training set to fit_transform() and predict a [your own](https://github.com/yashhy/sklearn-kick-start/tree/master/spam-or-not/test-ur-sentence) output.
2. Here our data set is different and we have to pass two attributes as part of training data.
3. Giving a 2D DataFrame to fit_transform() threw me a error 
> ValueError: Found input variables with inconsistent numbers of samples: [2, 24420]
4. Stackoverflow link for this problem is up [here](https://datascience.stackexchange.com/questions/26652/sklearn-tfidf-vectorize-returns-different-shape-after-fit-transform)


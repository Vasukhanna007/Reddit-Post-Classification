# coding: utf-8
#------------------------------------------------------------------------------
# Goal: compare our custom version of Muticalss Bernouilli Naive Bayes
#       with Sklearn's version
#------------------------------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from naive_bayes import MultiClassBernouilliNB
        

def main():
    pp_data = "../train_df_processed.csv"

    if not os.path.isfile(pp_data):
        print("ERROR: file %s does not exist.")
        sys.exit(1)

    # Read the pre-processed train set
    train = pd.read_csv(pp_data)
    train.set_index('id', inplace=True, drop=True)
    train['comments']   = train['comments'].astype(str)
    print(train.head())

    # Convert text to numbers
    vectorizer  = CountVectorizer(max_features=2000, min_df=5, max_df=0.7, binary=True)
    train.subreddits = pd.Categorical(train.subreddits)
    train['y']  = train.subreddits.cat.codes

    X = vectorizer.fit_transform(list(train['comments'])).toarray()
    y = train['y'].to_numpy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Models to compare
    sklearn_nb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
    custom_nb  = MultiClassBernouilliNB(alpha=1.0)

    sklearn_nb.fit(X_train, y_train)
    custom_nb.fit(X_train, y_train)

    sklearn_pred = sklearn_nb.predict(X_test)
    custom_pred  = custom_nb.predict(X_test)

    print(accuracy_score(sklearn_test, y_pred))
    print(accuracy_score(custom_test, y_pred))


if __name__ == "__main__":
    main()


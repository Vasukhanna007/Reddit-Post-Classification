#------------------------------------------------------------------------------
# coding : utf-8
# Goal   : Run experiments with Sklearn's Multinomial Naive Bayes model
#------------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



def main():

    clean_train_data = "./train_df_processed.csv"
    train = pd.read_csv(clean_train_data)
    train.set_index('id', inplace=True, drop=True)
    train['comments']   = train['comments'].astype(str)
    train.subreddits    = pd.Categorical(train.subreddits)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(train['comments'], train['subreddits'], test_size=0.2)

    text_clf.fit(X_train, y_train)

    acc = np.mean(y_test==text_clf.predict(X_test))

    print("Accuracy on test set = %.3f" % acc)

   

if __name__ == "__main__":
    main()



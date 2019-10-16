# coding: utf-8

import sys
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords



def get_vectoriezed_df(df):
    """
    Expects a dataframe where 
    - the column 'comments' are the preprocessed comments
    - the columns 'y' represents the numerical categories to be predicted
    Also returns the vocabulary
    """
    vectorizer = CountVectorizer(
        max_features=20000,
        min_df=5, max_df=0.7,
        stop_words=stopwords.words('english')
    )
    X = vectorizer.fit_transform(list(df['comments'])).toarray()
    X = TfidfTransformer().fit_transform(X).toarray()
    vocab   = vectorizer.vocabulary_
    vec_df  = pd.DataFrame(X, columns= vocab)
    vec_df['y'] = df['y']
    return vec_df, vocab


def split_dataframe_kfold(df, k=7):
    """
    Returns a list containing each fold
    as well as the vocabulary
    """
    # Shuffle dataframe
    df       = df.copy().sample(frac=1).reset_index(drop=True)
    fold_len = int(len(df.index)/k)
    cur_fold = 1
    idx_min  = 0
    df_folds = []
    vec_df, vocab = get_vectoriezed_df(df)

    while cur_fold < k:
        idx_max = idx_min + fold_len
        fold    = vec_df[idx_min:idx_max]
        df_folds.append(fold)

        # move to next fold
        idx_min = idx_max
        cur_fold = cur_fold + 1

    # Last fold
    fold = vec_df[idx_min:]
    df_folds.append(fold)
    return df_folds, vocab



def get_accuracy(y_test, y_pred, y_prob=None):

    categories = set(y_test)
    accuracies = {}

    for cat in categories:
        tp  = 0.0
        pop = 0.0
        maxp = 0.0
        others = pd.DataFrame(0, index=range(1), columns=categories)

        for i in range(len(y_test)):
            if  y_test[i] == cat:
                pop = pop+1.0
                if y_test[i] == y_pred[i]:
                    tp = tp+1.0
                # Update the classification statistics
                others.at[0,y_pred[i]] = others.at[0,y_pred[i]] + 1.0
                # Compute average maximum predicted probability
                maxp = maxp + max(y_prob[i])

        accuracies[cat] = {
            'acc'    : tp/pop,
            'details': others,
            'avg_prob' : maxp/pop
        }
    return accuracies


def print_accuracies(df, acc):

    # Category dict
    cat_dict = dict(enumerate(df['subreddits'].cat.categories))

    # Accuracies
    n = 0.0
    v = 0.0
    for key, val in acc.items():
        n = n + 1.0
        v = v + val['acc']
        print("#%s  %s => %.4f" % (key, cat_dict[key], val['acc']))
        print("Average confidence = %s" % val['avg_prob'])
        print("Details")
        print(val['details'])
        print("\n")



def kfoldCV(df, model, k=7):
    """
    Expects a dataframe where 
    - the column 'comments' are the preprocessed comments
    - the columns 'y' represents the numerical categories to be predicted
    """

    folds, vocab = split_dataframe_kfold(df, k)
    acc     = []

    for i_fold in range(k):

        print("\n=== FOLD #%s ===\n" % i_fold)

        # Train test split
        df_test     = folds[i_fold]
        df_train    = pd.concat([folds[i] for i in range(k) if i != i_fold])
        X_test      = df_test[vocab]
        y_test      = df_test['y']
        X_train     = df_train[vocab]
        y_train     = df_train['y']

        # Fit the model
        model.fit(X_train, y_train)

        # Get the model predictions on the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Compute detailed model accuracy
        acc = get_accuracy(list(y_test), list(y_pred), y_prob)

        print_accuracies(df, acc)

        # Sanity check
        acc = np.mean(y_test == y_pred)
        print("Mean Accuracy = %.4f" % acc)
 
 

def main():

    clean_train_data = "./train_df_processed.csv"
    train = pd.read_csv(clean_train_data)
    train.set_index('id', inplace=True, drop=True)
    train['comments']   = train['comments'].astype(str)
    train.subreddits    = pd.Categorical(train.subreddits)
    train['y']          = train.subreddits.cat.codes

    model = MultinomialNB()
    kfoldCV(train, model, 7)
    
   

if __name__ == "__main__":
    main()



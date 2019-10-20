#------------------------------------------------------------------------------
# coding : utf-8
# Goal   : Validation pipeline, using a held out test set
#------------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def custom_train_test_split(df, text_col, y_col, test_size):
    """
    Custom version of sklearn.model_selection's train_test_split
        df          dataframe to split
        text_col    comments' column name
        y_col       target column name
        test_size   test set ratio from 0 to 1
    """
    # Shuffle the dataframe
    data = df.copy().sample(frac=1).reset_index(drop=True)
    # Index where the split will happen
    split_idx = int((1.0 - test_size) * len(df.index))

    train = df[:split_idx]
    test  = df[split_idx:]

    return train[text_col], test[text_col], train[y_col], test[y_col]
    

def main():

    #------------------------------------------------------------------------------
    # Read the pre-processed training set
    #------------------------------------------------------------------------------
    clean_train_data = "./train_df_processed.csv"
    train = pd.read_csv(clean_train_data)
    train.set_index('id', inplace=True, drop=True)
    train['comments']   = train['comments'].astype(str)
    train.subreddits    = pd.Categorical(train.subreddits)

    # Train/test split
    X_train, X_test, y_train, y_test = custom_train_test_split(train, 'comments', 'subreddits', test_size=0.2)

    #------------------------------------------------------------------------------
    # Models to be compared
    #------------------------------------------------------------------------------
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine" : LinearSVC()
    }
    accuracies = {
        "Multinomial Naive Bayes": np.nan,
        "Support Vector Machine" : np.nan
    }

    for name,clf in models.items():

        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', clf),
        ])

        text_clf.fit(X_train, y_train)
        accuracies[name] = np.mean(y_test==text_clf.predict(X_test))


    # Print the accuracies
    for name,acc in accuracies.items():
        print("Test-set accuracy for %s = %.3f" % (name, acc))

   

if __name__ == "__main__":
    main()



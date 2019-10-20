#------------------------------------------------------------------------------
# coding : utf-8
# Goal   : Generate the predictions file to upload to kaggle
#------------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Custom pre-processing
from pre_processing import preprocess


def main():

    #------------------------------------------------------------------------------
    # Read the pre-processed training set
    #------------------------------------------------------------------------------
    clean_train_data = "./train_df_processed.csv"
    train = pd.read_csv(clean_train_data)
    train.set_index('id', inplace=True, drop=True)
    train['comments']   = train['comments'].astype(str)
    train.subreddits    = pd.Categorical(train.subreddits)

    #------------------------------------------------------------------------------
    # Train the best model from our experiments (Multinomial Naive Bayes)
    # on the entire train set
    #------------------------------------------------------------------------------
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    X_train = train['comments']
    y_train = train['subreddits']

    text_clf.fit(X_train, y_train)

    #------------------------------------------------------------------------------
    # Prepare the output predictions file
    #------------------------------------------------------------------------------

    # preprocess the test set 
    test_data = "../data/reddit_test.csv"
    test = pd.read_csv(test_data)
    test = preprocess(test)

    # use the trained model to predict the test set
    X_test = test['comments']
    x = text_clf.predict(X_test)
    x = np.array(x)
    submission = pd.DataFrame({'Id':test['id'],'Category':x})
    submission.head()
    filename = 'prediction.csv'
    submission.to_csv(filename,index=False)

    print("File %s is ready for submission to Kaggle" % filename)
    
   

if __name__ == "__main__":
    main()



#------------------------------------------------------------------------------
# coding : utf-8
# Goal   : Validation pipeline, using a held out test set
#------------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def split_dataframe_kfold(df, k=5):
    """
    Returns a dictionnary where the keys are the fold indices,
    and the values contain the training dataframe and the test dataframe
    {
        'fold1': { 
            'train': pd.Dataframe(...), 
            'test' : pd.DataFrame(...)
        }
        ...
    }
    """
    # Shuffle dataframe
    data         = df.copy().sample(frac=1).reset_index(drop=True)
    fold_len     = int(len(data.index)/k)
    output       = {}
    cur_fold     = 1
    test_idx_min = 0

    while cur_fold < k:
        test_idx_max = test_idx_min + fold_len

        # train_dataframe
        test_df = data[test_idx_min:test_idx_max]

        if test_idx_min == 0:
            train_df = data[test_idx_max:]
        else:
            # concatenate the remaining slices
            train1 = data[:test_idx_min]
            train2 = data[test_idx_max:]
            train_df = pd.concat([train1, train2], axis=0)

        # Update the dictionary
        output['fold%d'%cur_fold] = {
            'train': train_df,
            'test' : test_df
        }

        # Prepare next fold
        cur_fold     = cur_fold + 1
        test_idx_min = test_idx_max

    # Last fold
    test_df  = data[test_idx_min:]
    train_df = data[:test_idx_min]
    # Update the dictionary
    output['fold%d'%cur_fold] = {
        'train': train_df,
        'test' : test_df
    }
    return output
    

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
    n_folds   = 5
    kfold_dic = split_dataframe_kfold(train, n_folds)

    #------------------------------------------------------------------------------
    # Models to be compared
    #------------------------------------------------------------------------------
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine" : LinearSVC()
    }
    accuracies = {
        "Multinomial Naive Bayes": {},
        "Support Vector Machine" : {}
    }

    for name,clf in models.items():
        print(name)

        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', clf),
        ])

        # Loop through each fold and compute the accuracies
        for fold in kfold_dic.keys():
            print(fold)

            X_train = kfold_dic[fold]['train']['comments']
            X_test  = kfold_dic[fold]['test']['comments']
            y_train = kfold_dic[fold]['train']['subreddits']
            y_test  = kfold_dic[fold]['test']['subreddits']
        
            text_clf.fit(X_train, y_train)
            
            accuracies[name][fold] = np.mean(y_test==text_clf.predict(X_test))

        # Average accuracy for the model
        accuracies[name]['avg'] = sum([acc for acc in accuracies[name].values()])/n_folds


    #------------------------------------------------------------------------------
    # Print the accuracies
    #------------------------------------------------------------------------------
    print("\n\nModel Accuracies\n\n")

    headers = ['Model'] + [ fold for fold in kfold_dic.keys()] + ['Avg Accuracy']
    rows    = []
    for name,acc in accuracies.items():
        row = []
        row.append(name)
        for fold in acc.keys():
            row.append('%.3f' % acc[fold])
        rows.append(row)

    print(tabulate(rows, headers=headers))
    print("\n")
   

if __name__ == "__main__":
    main()



# coding: utf-8

import sys
import numpy as np
import pandas as pd
import re
import tldextract
import contractions
import inflect

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


stemmer     = PorterStemmer()
lemmatizer  = WordNetLemmatizer()
words       = set(nltk.corpus.words.words())
stop        = stopwords.words('english')


def lemmatize(comment):
    comment = comment.split()
    comment = [lemmatizer.lemmatize(word) for word in comment]
    #comment = [ stemmer.stem(w) for w in comment ]
    comment = ' '.join(comment)
    return comment

def replace_link(word):
    if 'http' in word:
        ext = tldextract.extract(word)
        return "%s.%s" % (ext.domain, ext.suffix)
    else:
        return word

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def replace_numbers(comment):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    words = comment.split()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)



def preprocess(df):

    #-------------------------------------------------------------
    # Add columns
    #-------------------------------------------------------------
    df['char_count']    = df['comments'].str.len() 
    df['stop_words']    = df['comments'].apply(lambda x: len([x for x in x.split() if x in stop]))
    df['n_questions']   = df['comments'].apply(lambda x: x.count('?'))
    df['youtube_url']   = df['comments'].apply(lambda x: x.count('youtu'))
    df['numerics']      = df['comments'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    #-------------------------------------------------------------
    # Text preprocessing for the 'comments' column
    #-------------------------------------------------------------
    # Lowercase
    df['comments'] = df['comments'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Replace full links by website only
    df['comments'] = df['comments'].apply(lambda x: " ".join(replace_link(x) for x in x.split()))
    # Remove text bwteen square brackets
    df['comments'] = df['comments'].apply(remove_between_square_brackets)
    # Replace contractions
    df['comments'] = df['comments'].apply(replace_contractions)
    # Replace all other special characters with spaces
    df['comments'] = df['comments'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
    # Replace numbers with their textual equivalent
    df['comments'] = df['comments'].apply(replace_numbers)

    # Delete all single characters
    df['comments'] = df['comments'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

    # Removing stop words
    df['comments'] = df['comments'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # Lemmatization
    df['comments'] = df['comments'].apply(lemmatize)

    #-------------------------------------------------------------
    # Words that are not in the dictionnary
    #-------------------------------------------------------------
    errors = []
    for comment in df['comments']:
        errors = errors + [w for w in comment.split() if w not in words]

    f = open("./spelling_errors.txt", "w")
    for err in errors:
        f.write("%s\n" % err)
    f.close()


    print(df.head())
    
    return df


def main():

    train_data = "../data/reddit_train.csv"
    train = pd.read_csv(train_data)
    train = preprocess(train)
    train.to_csv("train_df_processed.csv", index=False)

    text = open("pp_train.csv", "w")
    for comment in train['comments']:
        text.write("%s\n\n" % comment)
    text.close()
   

if __name__ == "__main__":
    main()



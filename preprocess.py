import os
from typing import List, Tuple
from tqdm import tqdm
import pickle
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class TweetExample:
    def __init__(self, words, tweet_id, label=None):
        self.words = words
        self.tweet_id = tweet_id
        self.label = label
    
    def __repr__(self):
        return f"Tweet id: {self.tweet_id}, words: {repr(self.words)}; class: {repr(self.label)}"

    def __str__(self):
        return self.__repr__()


def clean_tweet(tweet: str) -> str:
    space_pattern = '\s+'
    url_pattern = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_pattern = '@[\w\-]+'

    parsed_text = re.sub(space_pattern, ' ', tweet)
    parsed_text = re.sub(url_pattern, '', parsed_text)
    parsed_text = re.sub(mention_pattern, '', parsed_text)

    return parsed_text


def tweet_preprocess(sentence: str) -> List[str]:

    stop_words = stopwords.words("english")
    

    tweet = clean_tweet(sentence)
    tweet = BeautifulSoup(tweet, 'lxml').get_text()
    tweet = re.sub("[^a-zA-Z]", " ", tweet)
    words = word_tokenize(tweet.lower())

    preprocessed_words = []
    lemmatizer = WordNetLemmatizer()

    for word in words:
        lemma_word = lemmatizer.lemmatize(word)
        if lemma_word not in stop_words:
            preprocessed_words.append(lemma_word)

    return preprocessed_words


def load_examples(file_path: str, split: str) -> List[TweetExample]:
    '''Read tweet examples from raw file. Clean and tokenize the sentence.'''

    if os.path.exists(f"data/{split}_exs.pkl"):
        with open(f"data/{split}_exs.pkl", "rb") as f:
            exs = pickle.load(f)
    else:
        data = pd.read_csv(file_path)
        exs = []

        data = data.rename(columns = {'class': 'label'}, inplace=False)
        for row in tqdm(data.itertuples(), total=len(data), desc=f"Load {split} data"):
            label = getattr(row, "label")
            tweet_id = getattr(row, "Index")
            phrase = getattr(row, "tweet")

            word_list = tweet_preprocess(phrase)
            if len(word_list) > 0:
                exs.append(TweetExample(word_list, tweet_id, label))

        with open(f"data/{split}_exs.pkl", "wb") as f:
            pickle.dump(exs, f)
    return exs
        

def load_examples_transformer(file_path: str, split: str, tokenizer) -> Tuple[dict, np.array]:
    X = []
    y = []
    if os.path.exists(f"data/transformer/{split}_x.pkl") and os.path.exists(f"data/transformer/{split}_y.pkl"):
        with open(f"data/transformer/{split}_x.pkl", "rb") as f:
            X = pickle.load(f)
        with open(f"data/transformer/{split}_y.pkl", "rb") as f:
            y = pickle.load(f)
    else:
        data = pd.read_csv(file_path)
        data = data.rename(columns = {'class': 'label'}, inplace=False)
        tweets = []
        for row in tqdm(data.itertuples(), total=len(data), desc=f"Load {split} data"):
            label = getattr(row, "label")
            tweet = getattr(row, "tweet")
            tweet = clean_tweet(tweet)
            tweet = BeautifulSoup(tweet, 'lxml').get_text()
            tweets.append(tweet)
            y.append(label)

        X = tokenizer(tweets, return_tensors="pt", padding=True, truncation=False)
        y = np.array(y)
        with open(f"data/transformer/{split}_x.pkl", "wb") as f:
            pickle.dump(X, f)
        with open(f"data/transformer/{split}_y.pkl", "wb") as f:
            pickle.dump(y, f)

    return X, y




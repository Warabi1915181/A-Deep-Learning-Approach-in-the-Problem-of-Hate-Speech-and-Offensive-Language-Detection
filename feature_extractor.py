from typing import List
import numpy as np

from preprocess import TweetExample

np.random.seed(42)

class Vocab:
    def __init__(self):
        self.int2str = dict()
        self.str2int = dict()

    def index_of(self, word: str) -> int:
        return self.str2int[word]
    
    def get_word(self, index: int) -> str:
        return self.int2str[index]

    def add_word(self, word: str) -> None:
        index = len(self.int2str)
        self.int2str[index] = word
        self.str2int[word] = index

    def has_word(self, word: str) -> bool:
        return (word in self.str2int)

    def get_vocab_size(self) -> int:
        return len(self.int2str)

    def get_dict(self) -> dict:
        return self.str2int


class RawTextFeatureExtractor:
    def __init__(self, exs: List[TweetExample]) -> None:
        all_words = []
        self.max_len = 0
        for ex in exs:
            self.max_len = max(self.max_len, len(ex.words))
            all_words += ex.words
        distinct_words = sorted(list(set(all_words)))

        self.vocab = Vocab()
        for word in distinct_words:
            self.vocab.add_word(word)
        self.vocab.add_word('<unk>')
        self.vocab.add_word('<pad>')

    def extract_features(self, exs: List[TweetExample]) -> np.array:
        feats = []

        for ex in exs:
            feat = []
            for word in ex.words:
                if self.vocab.has_word(word):
                    feat.append(self.vocab.index_of(word))
                else:
                    feat.append(self.vocab.index_of("<unk>"))

            num_words = len(feat)
            feat += [self.vocab.index_of("<pad>")] * (self.max_len - num_words)
            feats.append(feat)
        
        return np.array(feats, dtype=np.int16)
    
    def get_vocab(self):
        return self.vocab
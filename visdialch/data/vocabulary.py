"""
A Vocabulary maintains a mapping between words and corresponding unique
integers, holds special integers (tokens) for indicating start and end of
sequence, and offers functionality to map out-of-vocabulary words to the
corresponding token.
"""
import json
import os
from typing import List
from anatool import AnaLogger, AnaArgParser


class Vocabulary(object):
    """
    A simple Vocabulary class which maintains a mapping between words and
    integer tokens. Can be initialized either by word counts from the VisDial
    v1.0 train dataset, or a pre-saved vocabulary mapping.

    Parameters
    ----------
    word_counts_path: str
        Path to a json file containing counts of each word across captions,
        questions and answers of the VisDial v1.0 train dataset.
    min_count : int, optional (default=5)
        When initializing the vocabulary from word counts, you can specify a
        minimum count, and every token with a count less than this will be
        excluded from vocabulary.
    """
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"
    UNK_TOKEN = "<UNK>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, word_counts_path, min_count=5, logger: AnaLogger = None):
        self.logger = logger

        if not os.path.exists(word_counts_path):
            self.logger.error('word counts do not exist at %s' % word_counts_path)
            raise FileNotFoundError

        with open(word_counts_path, 'r') as word_counts_file:
            word_counts = json.load(word_counts_file)
            self.logger.debug('word_counts: ' + str(len(word_counts)))

        word_counts = [
            (word, count)
            for word, count in word_counts.items()
            if count >= min_count
        ]
        word_counts = sorted(word_counts, key=lambda wc: -wc[1])

        words, _ = zip(*word_counts)

        self.word2index = {
            self.PAD_TOKEN: self.PAD_INDEX,
            self.SOS_TOKEN: self.SOS_INDEX,
            self.EOS_TOKEN: self.EOS_INDEX,
            self.UNK_TOKEN: self.UNK_INDEX
        }

        for index, word in enumerate(words):
            self.word2index[word] = index + 4

        self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))

        self.logger.debug('index2word length: ' + str(len(self.index2word)))

    def to_indices(self, words: List[str]):
        return [
            self.word2index.get(word, self.UNK_INDEX)
            for word in words
        ]

    def to_words(self, indices: List[int]):
        return [
            self.index2word.get(index, self.UNK_TOKEN)
            for index in indices
        ]

    def __len__(self):
        return len(self.index2word)

    def save(self, save_vocabulary_path):
        with open(save_vocabulary_path, 'w') as save_vocabulary_file:
            json.dump(self.word2index, save_vocabulary_file)

    def load(self, saved_vocabulary_path):
        with open(saved_vocabulary_path, 'r') as saved_vocabulary_file:
            self.word2index = json.load(saved_vocabulary_file)
            self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    Vocabulary(
        word_counts_path=opt.word_counts_json,
        min_count=opt.vocab_min_count,
        logger=logger
    )

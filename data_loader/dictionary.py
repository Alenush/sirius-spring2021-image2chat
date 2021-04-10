from collections import defaultdict
import copy
import numpy as np
import os
import json
import re
import nltk


RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)


class Dictionary():
    def __init__(self, opt):
        self.tokenizer = opt['tokenizer']
        self.filepaths = opt['filepaths']
        
        self.tokenizer_fun = getattr(self, self.tokenizer + '_tokenize')
        if self.tokenizer == 'nltk':
            st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
            try:
                self.sent_tok = nltk.data.load(st_path)
            except LookupError:
                nltk.download('punkt')
                self.sent_tok = nltk.data.load(st_path)
            self.sent_tok = nltk.data.load(st_path)
            self.word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()
        
        self.default_tokens = ['__null__', '__start__', '__end__', '__unk__']
        self._unk_token = '__unk__'
        
        self.freq = defaultdict(int)
        self.tok2ind = {}
        self.ind2tok = {}

        for default_token in self.default_tokens:
            self._add_token(default_token)

        self._unk_token_idx = self.tok2ind.get(self._unk_token)
        self._build()

    def _build(self):
        for path in self.filepaths:
            with open(path) as f:
                data = json.load(f)
                for elem in data:
                    for sent in elem['dialog']:
                        self._add_sentence(sent[1])
        
    def _add_token(self, word):
        if word not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[word] = index
            self.ind2tok[index] = word

    def nltk_tokenize(self, text):
        return [token for sent in self.sent_tok.tokenize(text) for token in self.word_tok.tokenize(sent)]

    def re_tokenize(self, text):
        return RETOK.findall(text)

    def tokenize(self, text):
        text = text.lower()
        word_tokens = self.tokenizer_fun(text)
        return word_tokens

    def _add_sentence(self, sentence):
        for token in self.tokenize(sentence):
            self._add_token(token)
            self.freq[token] += 1

    def remove_tail(self, min_freq):
        to_remove = []
        for token, freq in self.freq.items():
            if freq < min_freq or freq > max_freq:
                to_remove.append(token)

        for token in to_remove:
            del self.freq[token]
            idx = self.tok2ind.pop(token)
            del self.ind2tok[idx]
            
    def _word_lookup(self, key):
        return self.tok2ind.get(key, self._unk_token_idx)

    def _index_lookup(self, key):
        return self.ind2tok.get(key, self._unk_token)

    def txt2vec(self, text):
        return [self.tok2ind.get(token, self._unk_token_idx) for token in self.tokenize(str(text))]

    def vec2txt(self, vector, delimiter=' '):
        return ' '.join([self.ind2tok.get(index, self._unk_token) for index in vector])
      
"""
d = Dictionary({
    'tokenizer': 'nltk',
    'filepaths': ['C://Users//daria.vinogradova//ParlAI//data//image_chat//train.json',
                  'C://Users//daria.vinogradova//ParlAI//data//image_chat//test.json',
                  'C://Users//daria.vinogradova//ParlAI//data//image_chat//valid.json']
})
"""

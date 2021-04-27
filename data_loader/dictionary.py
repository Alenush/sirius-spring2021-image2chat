from collections import defaultdict
import json
import re
import nltk
from tqdm import tqdm

RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)


def unescape(s):
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')


class Dictionary():
    def __init__(self, dict_path):
        self.ind2tok = {}
        self.tok2ind = {}
        self.freq = {}
        self.tokenizer_fun = self.re_tokenize
        self.load(dict_path)

    def __len__(self):
        return len(self.ind2tok)

    def load(self, filename):
        self.null_token = '[PAD]'
        self._unk_token = '[UNK]'
        with open(filename, 'r', encoding='utf-8', errors='ignore') as read:
            for line in read:
                #split = line.strip().split('\t')
                token = unescape(line[:-1])
                #cnt = int(split[1]) if len(split) > 1 else 0
                #self.freq[token] = cnt
                self._add_token(token)
        self._unk_token_idx = self.tok2ind[self._unk_token]

    def _build(self):
        for path in self.filepaths:
            print(f"Update dictionary with {path}")
            with open(path) as f:
                data = json.load(f)
                for i in tqdm(range(len(data))):
                    elem = data[i]
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
            if freq < min_freq or freq > self.max_freq:
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
        return delimiter.join([self.ind2tok.get(index, self._unk_token) for index in vector])
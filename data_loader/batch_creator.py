import json
import os
import numpy as np
import torch

from data_loader.image_loader import ImageLoader
from data_loader.dictionary import Dictionary


class BatchCreator:
    def __init__(self, bs, train_path, images_path, personalities_path):
        self.use_cuda = False
        self.batch_size = bs
        self.img_loader = ImageLoader({
            'image_mode': 'resnet152',
            'image_size': 256,
            'image_cropsize': 224
        })
        self.images_path = images_path
        self._extract_text_and_images(train_path, images_path, personalities_path)
        self._setup_data(train_path)
        self._truncate_len = 32
        self.dictionary = Dictionary({
            'tokenizer': 'nltk',
            'filepaths': ['C://Users//daria.vinogradova//ParlAI//data//image_chat//train.json',
                          'C://Users//daria.vinogradova//ParlAI//data//image_chat//test.json',
                          'C://Users//daria.vinogradova//ParlAI//data//image_chat//valid.json']
        })

    def _setup_data(self, data_path):
        self.idx_to_ep = {}
        self.idx_to_turn = {}

        ep_idx = 0
        for i, d in enumerate(self.data):
            for j in range(len(d['dialog'])):
                self.idx_to_ep[ep_idx] = i
                self.idx_to_turn[ep_idx] = j
                ep_idx += 1

        self.total = ep_idx

    def _extract_text_and_images(self, train_path, images_path, personalities_path):
        raw_data = []
        with open(train_path) as f:
            raw_data = json.load(f)

        with open(personalities_path) as f:
            json_pers = json.load(f)
            personalities = json_pers["positive"] + json_pers["negative"] + json_pers["neutral"]
            self._build_personality_dictionary(personalities)

        self.data = []
        possible_hashes = set(os.listdir(images_path))
        for i, sample in enumerate(raw_data):
            if sample['image_hash'] + '.jpg' not in possible_hashes:
                continue
            self.data.append(sample)

    def personalities_to_index(self, personalities):
        res = []
        for p in personalities:
            if p in self.personality_to_id:
                res.append(self.personality_to_id[p] + 1)
            else:
                res.append(0)
        return res

    def _build_personality_dictionary(self, personalities_list):
        self.personality_to_id = {p: i for i, p in enumerate(personalities_list)}
        self.num_personalities = len(personalities_list) + 1

    def sentences_to_tensor(self, sentences):
        max_length = self._truncate_len
        indexes = []
        for s in sentences:
            vec = self.dictionary.txt2vec(s)
            if len(vec) > max_length:
                vec = vec[:max_length]
            indexes.append(vec)

        longest = max([len(v) for v in indexes])
        res = torch.LongTensor(len(sentences), longest).fill_(
            self.dictionary.tok2ind[self.dictionary.null_token]
        )
        mask = torch.FloatTensor(len(sentences), longest).fill_(0)
        for i, inds in enumerate(indexes):
            res[i, 0: len(inds)] = torch.LongTensor(inds)
            mask[i, 0: len(inds)] = torch.FloatTensor([1] * len(inds))

        if self.use_cuda:
            res = res.cuda()
            mask = mask.cuda()
        return res, mask

    def personalities_to_tensor(self, personalities):
        res = torch.FloatTensor(
            len(personalities), self.num_personalities
        ).fill_(0)
        p_to_i = self.personalities_to_index(personalities)
        for i, index in enumerate(p_to_i):
            res[i, index] = 1  # no personality corresponds to 0
        if self.use_cuda:
            res = res.cuda()
        return res

    def _get_dialogue(self, episode_idx):
        data = self.data[self.idx_to_ep[episode_idx]]
        turn = self.idx_to_turn[episode_idx]

        personality, text = data['dialog'][turn]
        episode_done = (turn == len(data['dialog']) - 1)
        full_dialog = []
        for i, utt in enumerate(data['dialog']):
            if i >= turn:
                break
            full_dialog.append(utt[1])
        return {
            'text': '\n'.join(full_dialog),
            'image_path': os.path.join(self.images_path, data['image_hash'] + '.jpg'),
            'episode_done': episode_done,
            'labels': [text],
            'personality': personality
        }

    def form_batch(self):
        idxs = np.random.randint(0, self.total, size=self.batch_size)
        raw_batch = [self._get_dialogue(idx) for idx in idxs]
        images_tensor = torch.squeeze(torch.stack([self.img_loader.load(data['image_path'])
                                     for data in raw_batch]))
        indexes, mask = self.sentences_to_tensor([data['text'] for data in raw_batch])
        personalities_ohe = self.personalities_to_tensor([data['personality'] for data in raw_batch])
        return images_tensor, (indexes, mask), personalities_ohe

if __name__ == '__main__':
    bc = BatchCreator(
        32,
        'C://Users//daria.vinogradova//ParlAI//data//image_chat//train.json',
        'C://Users//daria.vinogradova//ParlAI//data//yfcc_images',
        'C://Users//daria.vinogradova//ParlAI//data//personality_captions//personalities.json'
    )
    bc.form_batch()
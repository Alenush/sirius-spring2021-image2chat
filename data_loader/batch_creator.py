import json
import os
import numpy as np
from .image_loader import ImageLoader
from .dictionary import Dictionary


class BatchCreator:
    def __init__(self, bs, train_path, images_path):
        self.use_cuda = False
        self.batch_size = bs
        self.img_loader = ImageLoader({
            'image_mode': 'resnet152',
            'image_size': 256,
            'image_cropsize': 224
        })
        self.images_path = images_path
        self._extract_text_and_images(train_path, images_path)
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

    def _extract_text_and_images(self, train_path, images_path):
        raw_data = []
        with open(train_path) as f:
            raw_data = json.load(f)

        self.data = []
        possible_hashes = set(os.listdir(images_path))
        for i, sample in enumerate(raw_data[:50]):
            if sample['image_hash'] + '.jpg' not in possible_hashes:
                continue
            self.data.append(sample)

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

    def _get_dialogue(self, episode_idx):
        data = self.data[self.idx_to_ep[episode_idx]]
        turn = self.idx_to_turn[episode_idx]

        personality, text = data['dialog'][turn]
        episode_done = (turn == len(data['dialog']) - 1)
        full_dialog = [personality]
        for i, utt in enumerate(data['dialog']):
            if i >= turn:
                break
            full_dialog.append(utt[1])
        return {
            'text': '\n'.join(full_dialog),
            'image_path': os.path.join(self.images_path, data['image_hash'] + '.jpg'),
            'episode_done': episode_done,
            'labels': [text],
        }

    def form_batch(self):
        idxs = np.random.randint(0, self.total, size=self.batch_size)
        raw_batch = [self._get_dialogue(idx) for idx in idxs]
        images_tensor = torch.stack([self.img_loader.load(data['image_path'])
                                     for data in raw_batch])
        indexes, mask = self.sentences_to_tensor([data['text'] for data in raw_batch])
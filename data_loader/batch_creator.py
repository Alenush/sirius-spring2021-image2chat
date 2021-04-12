import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from data_loader.image_loader import ImageLoader
from data_loader.dictionary import Dictionary


class ImageChatDataset(Dataset):
    def __init__(self, dialogs_path, images_path, personalities_path):
        self.use_cuda = False
        self.img_loader = ImageLoader({
            'image_mode': 'resnet152',
            'image_size': 256,
            'image_cropsize': 224
        })
        self.images_path = images_path
        self._extract_text_and_images(os.path.join(dialogs_path, 'train.json'), images_path, personalities_path)
        self._setup_data()
        self._truncate_len = 32
        self.dictionary = Dictionary({
            'tokenizer': 'nltk',
            'filepaths': [os.path.join(dialogs_path, 'train.json'),
                          os.path.join(dialogs_path, 'test.json'),
                          os.path.join(dialogs_path, 'valid.json')]
        })

    def _build_personality_dictionary(self, personalities_list):
        self.personality_to_id = {p: i for i, p in enumerate(personalities_list)}
        self.num_personalities = len(personalities_list) + 1

    def _setup_data(self):
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
        for i, sample in enumerate(raw_data[:10]):
            if sample['image_hash'] + '.jpg' not in possible_hashes:
                continue
            self.data.append(sample)

    def _get_dialogue(self, episode_idx):
        data = self.data[self.idx_to_ep[episode_idx]]
        turn = self.idx_to_turn[episode_idx]
        personality, text = data['dialog'][turn]
        full_dialog = []
        for i, utt in enumerate(data['dialog']):
            if i >= turn:
                break
            full_dialog.append(utt[1])
        return {
            'dialogue_history': '\n'.join(full_dialog),
            'image_path': os.path.join(self.images_path, data['image_hash'] + '.jpg'),
            'true_continuation': text,
            'personality': personality
        }

    def personality_to_index(self, personality):
        if personality in self.personality_to_id:
            res = self.personality_to_id[personality] + 1
        else:
            res = 0
        return res

    def sentence_to_tensor(self, sentence):
        max_length = self._truncate_len
        vec = self.dictionary.txt2vec(sentence)
        if len(vec) > max_length:
            vec = vec[:max_length]

        res = torch.LongTensor(max_length).fill_(self.dictionary.tok2ind[self.dictionary.null_token])
        res[0: len(vec)] = torch.LongTensor(vec)
        mask = torch.FloatTensor(max_length).fill_(0)
        mask[0: len(vec)] = torch.FloatTensor([1] * len(vec))

        if self.use_cuda:
            res = res.cuda()
            mask = mask.cuda()
        return res, mask

    def personality_to_tensor(self, personality):
        res = torch.FloatTensor(self.num_personalities).fill_(0)
        index = self.personality_to_index(personality)
        res[index] = 1

        if self.use_cuda:
            res = res.cuda()
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self._get_dialogue(idx)
        images_tensor = torch.squeeze(self.img_loader.load(data['image_path']))
        d_indexes, d_mask = self.sentence_to_tensor(data['dialogue_history'])
        l_indexes, l_mask = self.sentence_to_tensor(data['true_continuation'])
        personality_ohe = self.personality_to_tensor(data['personality'])
        return images_tensor, personality_ohe, (d_indexes, d_mask), (l_indexes, l_mask)


if __name__ == '__main__':
    ds = ImageChatDataset(
        'C://Users//daria.vinogradova//ParlAI//data//image_chat',
        'C://Users//daria.vinogradova//ParlAI//data//yfcc_images',
        'C://Users//daria.vinogradova//ParlAI//data//personality_captions//personalities.json'
    )

    dataloader = DataLoader(ds, batch_size=4, shuffle=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
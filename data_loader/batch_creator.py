import json
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_loader.image_loader import ImageLoader
from data_loader.dictionary import Dictionary
from transformers import BertTokenizer

CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'

class ImageChatDataset(Dataset):
    def __init__(self, args, split='train'):
        self.use_cuda = torch.cuda.is_available()
        self.img_loader = ImageLoader({
            'image_mode': args.backbone, #'efficientnet', #'resnext101_32x48d_wsl'
            'image_size': 256,
            'image_cropsize': 224,
            'split': split
        })
        prefix = split + '.json'
        self.images_path = args.images_path
        self.image_mode = args.backbone
        self._ru_model = args.ru_model
        self._extract_text_and_images(os.path.join(args.dialogues_path, prefix), args.images_path, args.personalities_path)
        self._setup_data()
        self._truncate_len = 32
        self.dictionary = None

        if args.ru_model:
            self._build_tokenizer(args.transformer)
        else:
            self.dictionary = Dictionary(args.dict_path)

    def _build_tokenizer(self, bert_type):
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

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
        possible_hashes = set(os.listdir(images_path + '/' + self.image_mode))
        print("loading data: ")
        for i in tqdm(range(len(raw_data))):
            sample = raw_data[i]
            if sample['image_hash'] not in possible_hashes:
                continue
            self.data.append(sample)

    def _get_dialogue(self, episode_idx):
        data = self.data[self.idx_to_ep[episode_idx]]
        if 'candidates' in data:
            cands = data['candidates']
        else:
            cands = {}
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
            'personality': personality,
            'turn': turn,
            'candidates': cands
        }

    def personality_to_index(self, personality):
        if personality in self.personality_to_id:
            res = self.personality_to_id[personality] + 1
        else:
            res = 0
        return res

    def sentences_to_tensor_with_dict(self, sentences):
        res_all = []
        masks_all = []

        for sentence in sentences:
            max_length = self._truncate_len
            vec = self.dictionary.txt2vec(sentence)
            if len(vec) > max_length:
                vec = vec[:max_length]

            res = torch.LongTensor(max_length).fill_(self.dictionary.tok2ind[self.dictionary.null_token])
            res[0: len(vec)] = torch.LongTensor(vec)
            mask = torch.LongTensor(max_length).fill_(0)
            mask[0: len(vec)] = torch.LongTensor([1] * len(vec))

            if self.use_cuda:
                res = res.cuda()
                mask = mask.cuda()

            res_all.append(res)
            masks_all.append(mask)
        return torch.stack(res_all), torch.stack(masks_all)

    def sentences_to_tensor_with_tokenizer(self, sentences):
        res_all = []
        masks_all = []
        max_length = self._truncate_len

        for sent in sentences:
            tokens = [CLS_TOKEN] + self.tokenizer.tokenize(sent) + [SEP_TOKEN]
            if len(tokens) > self._truncate_len:
                tokens = tokens[:self._truncate_len]
                tokens[-1] = SEP_TOKEN
            res = torch.LongTensor(max_length).fill_(0)
            mask = torch.LongTensor(max_length).fill_(0)
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            res[0: len(tokens)] = torch.LongTensor(tokens_ids)
            mask[0: len(tokens)] = 1

            if self.use_cuda:
                res = res.cuda()
                mask = mask.cuda()

            res_all.append(res)
            masks_all.append(mask)
        return torch.stack(res_all), torch.stack(masks_all)


    def sentences_to_tensor(self, sentences):
        if self._ru_model:
            return self.sentences_to_tensor_with_tokenizer(sentences)
        else:
            return self.sentences_to_tensor_with_dict(sentences)

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
        personality_ohe = self.personality_to_tensor(data['personality'])
        turn = torch.LongTensor([data['turn']])
        return images_tensor, personality_ohe, data['dialogue_history'], data['true_continuation'], turn, data['candidates']

import json
import os
import numpy as np

class BatchCreator:
    def __init__(self, bs, train_path, images_path):
        self.batch_size = bs
        self.img_loader = ImageLoader({
            'image_mode': 'resnet152',
            'image_size': 256,
            'image_cropsize': 224
        })
        self.images_path = images_path
        self._extract_text_and_images(train_path, images_path)
        self._setup_data(train_path)
        
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
            if sample['image_hash']+'.jpg' not in possible_hashes:
                continue
            self.data.append(sample)
        
    def _get_dialogue(self, episode_idx):
        data = self.data[self.idx_to_ep[episode_idx]]
        turn = self.idx_to_turn[episode_idx]
        
        personality, text = data['dialog'][turn]
        episode_done = (turn == len(data['dialog'])-1)
        full_dialog = [personality]
        for i, utt in enumerate(data['dialog']):
            if i >= turn:
                break
            full_dialog.append(utt[1])
        return {
            'text': '\n'.join(full_dialog),
            'image': self.img_loader.load(os.path.join(self.images_path, data['image_hash'] + '.jpg')),
            'episode_done': episode_done,
            'labels': [text],
        }
    
    def form_batch(self):
        idxs = np.random.randint(0, self.total, size=self.batch_size)
        raw_batch = [self._get_dialogue(idx) for idx in idxs]
        return raw_batch

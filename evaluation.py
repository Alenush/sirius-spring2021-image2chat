import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from model import TransresnetMultimodalModel
from data_loader.batch_creator import ImageChatDataset

use_cuda = torch.cuda.is_available()


def rank_candidates(dialogue_encoded, labels_encoded, labels_str, true_label):
    with torch.no_grad():
        dot_products = torch.mm(dialogue_encoded, labels_encoded.t())
        log_prob = log_softmax(dot_products, dim=1)
        order = torch.argsort(log_prob, descending=True)
        ranked = np.array(labels_str)[order]
        top1 = labels_str[true_label][0] in ranked[0][0][0]
        top5 = labels_str[true_label][0] in ranked[0][:5][0]
        top10 = labels_str[true_label][0] in ranked[0][:10][0]
        #print(labels_str[true_label][0])
    return top1, top5, top10


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', default='C://Users//daria.vinogradova//ParlAI//data//yfcc_images', type=str)
    parser.add_argument('--dialogues_path', default='C://Users//daria.vinogradova//ParlAI//data//image_chat', type=str)
    parser.add_argument('--dict_path',
                        default='C://Users//daria.vinogradova//ParlAI//data//models//image_chat//transresnet_multimodal//model.dict',
                        type=str)
    parser.add_argument('--personalities_path',
                        default='C://Users//daria.vinogradova//ParlAI//data//personality_captions//personalities.json',
                        type=str)
    parser.add_argument('--model_path',
                        default='C://Users//daria.vinogradova//ParlAI//data//models//image_chat//model_resnext_1',
                        type=str)

    args = parser.parse_args()
    test_ds = ImageChatDataset(
        args.dialogues_path,
        args.images_path,
        args.personalities_path,
        args.dict_path,
        'test.json'
    )

    model = TransresnetMultimodalModel(test_ds.dictionary)
    model.load_state_dict(torch.load(args.model_path,
                                     map_location=torch.device('cpu') if not use_cuda else None)['model_state_dict'])
    model.eval()

    top1 = {100: 0, 1000: 0}
    top5 = {100: 0, 1000: 0}
    top10 = {100: 0, 1000: 0}
    cnt = 0

    with open(os.path.join(args.dialogues_path, 'test.json')) as f:
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
        for i, batch in enumerate(test_loader):
            print(i)
            image, personality, dialogue_history, true_continuation, turn, candidates = batch
            for num_cands in [100, 1000]:
                labels = candidates[turn][str(num_cands)]
                true_id = labels.index(true_continuation)

                d_indexes, d_masks = test_ds.sentences_to_tensor(dialogue_history)
                l_indexes, l_masks = test_ds.sentences_to_tensor(labels)

                if use_cuda:
                    image = image.cuda()
                    personality = personality.cuda()
                    d_indexes = d_indexes.cuda()
                    d_masks = d_masks.cuda()
                    l_indexes = l_indexes.cuda()
                    l_masks = l_masks.cude()

                samples_encoded, answers_encoded = model(image, personality, (d_indexes, d_masks),
                                                         (l_indexes, l_masks))
                _top1, _top5, _top10 = rank_candidates(samples_encoded, answers_encoded, labels, true_id)
                top1[num_cands] += _top1
                top5[num_cands] += _top5
                top10[num_cands] += _top10
            cnt += 1

    print(f'top1 acc: {top1[100] / cnt} for 100, {top1[1000] / cnt} for 1000')
    print(f'top5 acc: {top5[100] / cnt} for 100, {top5[1000] / cnt} for 1000')
    print(f'top10 acc: {top10[100] / cnt} for 100, {top10[1000] / cnt} for 1000')

import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from model import TransresnetMultimodalModel
from data_loader.batch_creator import ImageChatDataset
import re
import argparse
from nltk.translate import bleu_score as nltkbleu

use_cuda = torch.cuda.is_available()


def rank_output_candidates(dialogue_encoded, labels_encoded, labels_str, true_label):
    with torch.no_grad():
        dot_products = torch.mm(dialogue_encoded, labels_encoded.t())
        log_prob = log_softmax(dot_products, dim=1)
        order = torch.argsort(log_prob, descending=True)
        ranked = np.array(labels_str)[order.cpu().numpy()]
        ranked = np.squeeze(ranked)
        if true_label is None:
            return ranked[0]
        chosen = labels_str[true_label][0]
        top1 = chosen == ranked[0]
        top5 = chosen in ranked[:5]
        top10 = chosen in ranked[:10]
        bleu = BleuMetric().compute(chosen, [ranked[0]], 4)
    return top1, top5, top10, bleu


def apply_model(ds, model, image_tensor, personality_tensor, dialogue_history, labels):
    d_indexes, d_masks = ds.sentences_to_tensor(dialogue_history)
    l_indexes, l_masks = ds.sentences_to_tensor(labels)

    if use_cuda:
        image_tensor = image_tensor.cuda()
        personality_tensor = personality_tensor.cuda()
        d_indexes = d_indexes.cuda()
        d_masks = d_masks.cuda()
        l_indexes = l_indexes.cuda()
        l_masks = l_masks.cuda()

    return model(image_tensor, personality_tensor, (d_indexes, d_masks),
                                             (l_indexes, l_masks))


def create_model_and_dataset(args):
    test_ds = ImageChatDataset(
        args.dialogues_path,
        args.images_path,
        args.personalities_path,
        args.dict_path,
        'test',
        args.backbone,
        'test.json',
    )

    model = TransresnetMultimodalModel(test_ds.dictionary)
    model.load_state_dict(torch.load(args.model_path,
                                     map_location=torch.device('cpu') if not use_cuda else None)['model_state_dict'])
    if use_cuda:
        model = model.cuda()
    model.eval()
    return test_ds, model


class BleuMetric:
    @staticmethod
    def normalize_answer(s):
        re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
        re_art = re.compile(r'\b(a|an|the)\b')
        s = s.lower()
        s = re_punc.sub(' ', s)
        s = re_art.sub(' ', s)
        s = ' '.join(s.split())
        return s

    @staticmethod
    def compute(guess, answers, k):
        if nltkbleu is None:
            # bleu library not installed, just return a default value
            return None

        weights = [1 / k for _ in range(k)]

        score = nltkbleu.sentence_bleu(
            [BleuMetric.normalize_answer(a).split(" ") for a in answers],
            BleuMetric.normalize_answer(guess).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
            weights=weights,
        )
        return (score)


def evaluate(args, model, test_ds):
    model.eval()

    top1 = {100: 0, 1000: 0}
    top5 = {100: 0, 1000: 0}
    top10 = {100: 0, 1000: 0}
    bleu = {100: 0, 1000: 0}
    top1_turn = {100: [0, 0, 0], 1000: [0, 0, 0]}
    top5_turn = {100: [0, 0, 0], 1000: [0, 0, 0]}
    top10_turn = {100: [0, 0, 0], 1000: [0, 0, 0]}
    cnt = 0
    cnt_turns = [0, 0, 0]

    with open(os.path.join(args.dialogues_path, 'test.json')) as f:
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
        for i, batch in enumerate(test_loader):
            if i % 100 == 0:
                print(i)
            image, personality, dialogue_history, true_continuation, turn, candidates = batch
            turn = turn[0].item()
            for num_cands in [100, 1000]:
                labels = candidates[turn][str(num_cands)]
                true_id = labels.index(true_continuation)
                samples_encoded, answers_encoded = apply_model(test_ds, model, image, personality, dialogue_history,
                                                               labels)
                _top1, _top5, _top10, _bleu = rank_output_candidates(samples_encoded, answers_encoded, labels, true_id)
                top1[num_cands] += _top1
                top5[num_cands] += _top5
                top10[num_cands] += _top10
                bleu[num_cands] += _bleu
                top1_turn[num_cands][turn] += _top1
                top5_turn[num_cands][turn] += _top5
                top10_turn[num_cands][turn] += _top10
            cnt_turns[turn] += 1
            cnt += 1

    print(f'top1 acc: {top1[100] / cnt} for 100, {top1[1000] / cnt} for 1000')
    print(f'top5 acc: {top5[100] / cnt} for 100, {top5[1000] / cnt} for 1000')
    print(f'top10 acc: {top10[100] / cnt} for 100, {top10[1000] / cnt} for 1000')
    print(f'bleu: {bleu[100] / cnt} for 100, {bleu[1000] / cnt} for 1000')

    for turn in range(3):
        print(f'turn {turn}:')
        print(f'top1 acc: {top1_turn[100][turn] / cnt_turns[turn]} for 100, {top1_turn[1000][turn] / cnt_turns[turn]} for 1000')
        print(f'top5 acc: {top5_turn[100][turn] / cnt_turns[turn]} for 100, {top5_turn[1000][turn] / cnt_turns[turn]} for 1000')
        print(f'top10 acc: {top10_turn[100][turn] / cnt_turns[turn]} for 100, {top10_turn[1000][turn] / cnt_turns[turn]} for 1000')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', default='data//yfcc_images', type=str)
    parser.add_argument('--dialogues_path', default='data//image_chat', type=str)
    parser.add_argument('--dict_path',
                        default='data//models//image_chat//transresnet_multimodal//model.dict',
                        type=str)
    parser.add_argument('--personalities_path',
                        default='data//personalities.json',
                        type=str)
    parser.add_argument('--model_path',
                        default='C://Users//daria.vinogradova//ParlAI//data//models//image_chat//model_resnext_1',
                        type=str)
    parser.add_argument('--backbone', default="resnet152", help='type of backbone')

    args = parser.parse_args()
    test_ds, model = create_model_and_dataset(args)
    evaluate(args, model, test_ds)

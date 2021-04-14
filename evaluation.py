import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from model import TransresnetMultimodalModel
from data_loader.batch_creator import ImageChatDataset


def rank_candidates(dialogue_encoded, labels_encoded, labels_str, true_label):
    with torch.no_grad():
        dot_products = torch.mm(dialogue_encoded, labels_encoded.t())
        log_prob = log_softmax(dot_products, dim=1)
        order = torch.argsort(log_prob, descending=True)
        ranked = np.array(labels_str)[order]
        print(ranked[0][:5,:])
        print(labels_str[true_label])


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

    args = parser.parse_args()
    test_ds = ImageChatDataset(
        args.dialogues_path,
        args.images_path,
        args.personalities_path,
        args.dict_path,
        'test.json'
    )

    model = TransresnetMultimodalModel(test_ds.dictionary)
    model.load_state_dict(torch.load('C://Users//daria.vinogradova//ParlAI//data//models//image_chat//model_resnext_1',
                                     map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    with open('C://Users//daria.vinogradova//ParlAI//data//image_chat//test.json') as f:
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
        for i, batch in enumerate(test_loader):
            image, personality, dialogue_history, true_continuation, turn, candidates = batch
            for num_cands in ['100', '1000']:
                labels = candidates[turn][num_cands]
                true_id = labels.index(true_continuation)

                d_indexes, d_masks = test_ds.sentences_to_tensor(dialogue_history)
                l_indexes, l_masks = test_ds.sentences_to_tensor(labels)

                samples_encoded, answers_encoded = model(image, personality, (d_indexes, d_masks),
                                                         (l_indexes, l_masks))
                rank_candidates(samples_encoded, answers_encoded, labels, true_id)
                break

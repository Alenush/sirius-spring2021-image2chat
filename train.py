import argparse
from data_loader.batch_creator import ImageChatDataset
from torch.utils.data import DataLoader
from model import TransresnetMultimodalModel
import torch
import os
from torch import optim
from torch.nn.functional import log_softmax, nll_loss

use_cuda = torch.cuda.is_available()

def get_loss(dialogs_encoded, labels_encoded):
    dot_products = torch.mm(dialogs_encoded, labels_encoded.t())
    log_prob = log_softmax(dot_products, dim=1)
    targets = torch.arange(0, len(dialogs_encoded), dtype=torch.long)
    if use_cuda:
        targets = targets.cuda()
    loss = nll_loss(log_prob, targets)
    num_correct = (log_prob.max(dim=1)[1] == targets).float().sum()
    return loss, num_correct


def load_transformers(model, context_encoder_path, label_encoder_path):
    model.context_encoder.load_state_dict(torch.load(context_encoder_path))
    model.label_encoder.load_state_dict(torch.load(label_encoder_path))


def compute_metrics(valid_loader):
    with torch.no_grad():
        model.eval()
        cnt = 0
        total_acc = 0
        for batch in valid_loader:
            images, personalities, (d_indexes, d_masks), (l_indexes, l_masks) = batch
            if use_cuda:
                images = images.cuda()
                personalities = personalities.cuda()
                d_indexes = d_indexes.cuda()
                d_masks = d_masks.cuda()
                l_indexes = l_indexes.cuda()
                l_masks = l_masks.cuda()

            samples_encoded, answers_encoded = model(images, personalities, (d_indexes, d_masks), (l_indexes, l_masks))
            _, n_correct = get_loss(samples_encoded, answers_encoded)
            total_acc += n_correct / images.shape[0]
            cnt += 1

        print('valid accuracy: ', total_acc / cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--batchsize', default=32, type=int, help='batch size')
    parser.add_argument('--images_path', default='C://Users//daria.vinogradova//ParlAI//data//yfcc_images', type=str)
    parser.add_argument('--dialogues_path', default='C://Users//daria.vinogradova//ParlAI//data//image_chat', type=str)
    parser.add_argument('--dict_path',
                        default='C://Users//daria.vinogradova//ParlAI//data//models//image_chat//transresnet_multimodal//model.dict',
                        type=str)
    parser.add_argument('--personalities_path',
                        default='C://Users//daria.vinogradova//ParlAI//data//personality_captions//personalities.json',
                        type=str)
    parser.add_argument('--label_enc',
                        default='', #C://Users//daria.vinogradova//ParlAI//data//image_chat//label_encoder.pt
                        type=str)
    parser.add_argument('--context_enc',
                        default='', #C://Users//daria.vinogradova//ParlAI//data//image_chat//context_encoder.pt
                        type=str)

    args = parser.parse_args()

    train_ds = ImageChatDataset(
        args.dialogues_path,
        args.images_path,
        args.personalities_path,
        args.dict_path
    )
    valid_ds = ImageChatDataset(
        args.dialogues_path,
        args.images_path,
        args.personalities_path,
        args.dict_path,
        'valid.json'
    )

    train_loader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batchsize, shuffle=True)

    model = TransresnetMultimodalModel(train_ds.dictionary)
    context_encoder_path = args.context_enc
    label_encoder_path = args.label_enc
    if context_encoder_path != '' and label_encoder_path != '':
        load_transformers(model, context_encoder_path, label_encoder_path)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.0001)

    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            images, personalities, (d_indexes, d_masks), (l_indexes, l_masks) = batch

            if use_cuda:
                images = images.cuda()
                personalities = personalities.cuda()
                d_indexes = d_indexes.cuda()
                d_masks = d_masks.cuda()
                l_indexes = l_indexes.cuda()
                l_masks = l_masks.cuda()

            samples_encoded, answers_encoded = model(images, personalities, (d_indexes, d_masks), (l_indexes, l_masks))
            loss, ok = get_loss(samples_encoded, answers_encoded)
            if i % 10 == 0:
                print(loss, ok)
            if i % 100 == 0 and i > 0:
                compute_metrics(valid_loader)

            loss.backward()
            optimizer.step()

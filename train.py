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

        return total_acc / cnt


def save_state(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)


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

    parser.add_argument('--valid_after_epoch_fraction', default=0.05, type=float)
    parser.add_argument('--loss_after_n_batches', default=20, type=int)
    parser.add_argument('--save_model_every', type=float, default=0.1, help='save model every fraction of epoch')
    parser.add_argument('--save_model_path', default="./model_state_dict")

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
    n_batches = len(train_loader)
    best_val_acc, no_updates, stopped = -1, 0, False
    valid_after_n_bathes = args.valid_after_epoch_fraction * n_batches

    for epoch in range(args.epochs):
        valid_cnt = 0
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

            loss.backward()
            optimizer.step()

            if i % args.loss_after_n_batches:
                print(loss, ok)

            if i % valid_after_n_bathes == 0 and i > 0:
                val_acc = compute_metrics(valid_loader)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_updates = 0
                else:
                    no_updates += 1
                    if no_updates == args.early_stopping:
                        print(f"No updates of accuracy for {no_updates} steps, stopping training")
                        save_state(model, optimizer, args.save_model_path)
                        stopped = True
                        break
                print("valid accuracy: ", val_acc)

            k = int(n_batches * args.save_model_every)
            if i % k == 0 and i != 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, args.save_model_path)

        print(f'{epoch} epoch passed. Summary:')
        compute_metrics(valid_loader)

import argparse
from data_loader.batch_creator import ImageChatDataset
from torch.utils.data import DataLoader

from evaluation import evaluate
from model import TransresnetMultimodalModel
import torch
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
    print('Loaded pretrained transformers')


def compute_metrics_onesample(dialogs_encoded, labels_encoded, k=5):
    dot_products = torch.mm(dialogs_encoded, labels_encoded.t())
    _, ids5 = torch.topk(dot_products, k=5, dim=1)
    _, ids10 = torch.topk(dot_products, k=10, dim=1)
    targets = torch.arange(0, len(dialogs_encoded), dtype=torch.long)
    if use_cuda:
        targets = targets.cuda()
    top1 = (ids5[:, 0] == targets).int().sum()
    top5 = (ids5 == targets[:, None]).int().sum()
    top10 = (ids10 == targets[:, None]).int().sum()
    return top1, top5, top10


def compute_metrics(valid_loader):
    with torch.no_grad():
        model.eval()
        cnt = 0
        turns_acc1 = [0, 0, 0]
        turns_acc5 = [0, 0, 0]
        turns_acc10 = [0, 0, 0]
        turns_cnt = [0, 0, 0]

        for batch in valid_loader:
            images, personalities, dialogues, labels, turns, _ = batch
            d_indexes, d_masks = train_ds.sentences_to_tensor(dialogues)
            l_indexes, l_masks = train_ds.sentences_to_tensor(labels)
            turns = torch.squeeze(turns)
            if use_cuda:
                images = images.cuda()
                personalities = personalities.cuda()
                d_indexes = d_indexes.cuda()
                d_masks = d_masks.cuda()
                l_indexes = l_indexes.cuda()
                l_masks = l_masks.cuda()
                turns = turns.cuda()

            for turn in range(3):
                mask = (turns == turn)
                samples_encoded, answers_encoded = model(images[mask], personalities[mask],
                                                         (d_indexes[mask], d_masks[mask]),
                                                         (l_indexes[mask], l_masks[mask]))

                #_, n_correct = get_loss(samples_encoded, answers_encoded)
                acc1, acc5, acc10 = compute_metrics_onesample(samples_encoded, answers_encoded, 5)
                turns_acc1[turn] += acc1
                turns_acc5[turn] += acc5
                turns_acc10[turn] += acc10
                turns_cnt[turn] += torch.sum(mask)
                cnt += 1

        for turn in range(3):
            print(f'{turn+1} turn acc1: {turns_acc1[turn] / turns_cnt[turn]}')
        for turn in range(3):
            print(f'{turn + 1} turn acc5: {turns_acc5[turn] / turns_cnt[turn]}')
        for turn in range(3):
            print(f'{turn + 1} turn acc5: {turns_acc10[turn] / turns_cnt[turn]}')

        mean_acc1 = (turns_acc1[0] + turns_acc1[1] + turns_acc1[2]) / (turns_cnt[0] + turns_cnt[1] + turns_cnt[2])
        mean_acc5 = (turns_acc5[0] + turns_acc5[1] + turns_acc5[2]) / (turns_cnt[0] + turns_cnt[1] + turns_cnt[2])
        mean_acc10 = (turns_acc10[0] + turns_acc10[1] + turns_acc10[2]) / (turns_cnt[0] + turns_cnt[1] + turns_cnt[2])
        return mean_acc1, mean_acc5, mean_acc10


def save_state(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--batchsize', default=500, type=int, help='batch size')
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
    parser.add_argument('--save_model_every', type=float, default=0.33, help='save model every fraction of epoch')
    parser.add_argument('--save_model_path', default="./model_state_dict")
    parser.add_argument('--early_stopping', type=int, default=5)

    args = parser.parse_args()
    backbone_type = "efficientnet"
    train_ds = ImageChatDataset(
        args.dialogues_path,
        args.images_path,
        args.personalities_path,
        args.dict_path,
        "train",
        backbone_type
    )
    valid_ds = ImageChatDataset(
        args.dialogues_path,
        args.images_path,
        args.personalities_path,
        args.dict_path,
        "val",
        backbone_type,
        'val.json',
    )

    test_ds = ImageChatDataset(
        args.dialogues_path,
        args.images_path,
        args.personalities_path,
        args.dict_path,
        "test",
        backbone_type,
        'test.json',
    )


    train_loader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batchsize, shuffle=True)

    model = TransresnetMultimodalModel(train_ds.dictionary, backbone_type)
    context_encoder_path = args.context_enc
    label_encoder_path = args.label_enc
    if context_encoder_path != '' and label_encoder_path != '':
        load_transformers(model, context_encoder_path, label_encoder_path)
    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.0005)
    n_batches = len(train_loader)
    best_val_acc, no_updates, stopped = -1, 0, False
    valid_after_n_bathes = int(args.valid_after_epoch_fraction * n_batches)

    for epoch in range(0):
        model.train()
        valid_cnt = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            images, personalities, dialogues, labels, _, _ = batch
            d_indexes, d_masks = train_ds.sentences_to_tensor(dialogues)
            l_indexes, l_masks = train_ds.sentences_to_tensor(labels)

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
                print(f'{loss.item()} loss, {ok.item()} right samples from {args.batchsize}')

            if i % valid_after_n_bathes == 0 and i > 0:
                val_acc1, val_acc5, val_acc10 = compute_metrics(valid_loader)
                if val_acc1 > best_val_acc:
                    best_val_acc = val_acc1
                    no_updates = 0
                else:
                    no_updates += 1
                    if no_updates == args.early_stopping:
                        print(f"No updates of accuracy for {no_updates} steps, stopping training")
                        save_state(model, optimizer, args.save_model_path)
                        stopped = True
                        break
                print("valid accuracy1: ", val_acc1.item())
                print("valid accuracy5: ", val_acc5.item())
                print("valid accuracy10: ", val_acc10.item())

            if i % int(n_batches * args.save_model_every) == 0 and i != 0:
                save_state(model, optimizer, args.save_model_path)
        if stopped:
            break

    evaluate(args, model, valid_ds)

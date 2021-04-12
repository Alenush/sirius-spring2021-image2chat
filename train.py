import argparse
from data_loader.batch_creator import ImageChatDataset
from torch.utils.data import DataLoader
import torch
from torch import optim
from torch.nn.functional import log_softmax, nll_loss


def get_loss(dialogs_encoded, labels_encoded):
    dot_products = torch.mm(dialogs_encoded, labels_encoded.t())
    log_prob = log_softmax(dot_products, dim=1)
    targets = torch.arange(0, len(dialogs_encoded), dtype=torch.long)
    loss = nll_loss(log_prob, targets)
    num_correct = (log_prob.max(dim=1)[1] == targets).float().sum()
    return loss, num_correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--batchsize', default=32, type=int, help='batch size')

    args = parser.parse_args()
    args_dict = vars(args)

    ds = ImageChatDataset(
        'C://Users//daria.vinogradova//ParlAI//data//image_chat',
        'C://Users//daria.vinogradova//ParlAI//data//yfcc_images',
        'C://Users//daria.vinogradova//ParlAI//data//personality_captions//personalities.json'
    )
    loader = DataLoader(ds, batch_size=args_dict['batchsize'], shuffle=True)

    model = TransResNetModel(...)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.0001)

    for epoch in args_dict['epoch']:
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            images, personalities, (d_indexes, d_masks), (l_indexes, l_masks) = batch
            samples_encoded, answers_encoded = model(images, personalities, (d_indexes, d_masks), (l_indexes, l_masks))
            loss = get_loss(samples_encoded, answers_encoded)
            loss.backward()
            optimizer.step()
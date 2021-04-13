import argparse
from data_loader.batch_creator import ImageChatDataset
from torch.utils.data import DataLoader
from model import TransresnetMultimodalModel
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


def load_transformers(model, context_encoder_path, label_encoder_path):
    model.context_encoder.load_state_dict(torch.load(context_encoder_path))
    model.label_encoder.load_state_dict(torch.load(label_encoder_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--batchsize', default=32, type=int, help='batch size')
    parser.add_argument('--impath', default='C://Users//daria.vinogradova//ParlAI//data//yfcc_images', type=str)
    parser.add_argument('--textpath', default='C://Users//daria.vinogradova//ParlAI//data//image_chat', type=str)
    parser.add_argument('--perspath', default='C://Users//daria.vinogradova//ParlAI//data//personality_captions//personalities.json', type=str)
    args = parser.parse_args()
    args_dict = vars(args)

    ds = ImageChatDataset(
        args_dict['textpath'],
        args_dict['impath'],
        args_dict['perspath']
    )
    loader = DataLoader(ds, batch_size=args_dict['batchsize'], shuffle=True)
    model = TransresnetMultimodalModel(ds.dictionary)
    context_encoder_path = # TODO
    label_encoder_path =  # TODO
    load_transformers(model, context_encoder_path, label_encoder_path)
    model = model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.0001)

    for epoch in range(args_dict['epochs']):
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            images, personalities, (d_indexes, d_masks), (l_indexes, l_masks) = batch
            samples_encoded, answers_encoded = model(images.cuda(), personalities.cuda(),
                                                     (d_indexes.cuda(), d_masks.cuda()), (l_indexes.cuda(), l_masks.cuda()))
            loss, ok = get_loss(samples_encoded, answers_encoded)
            print(loss, ok)
            loss.backward()
            optimizer.step()

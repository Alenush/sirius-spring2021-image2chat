import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from scipy.special import softmax
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import math
import argparse


def calc_ppl(phrase,
             tokenizer,
             model,
             model_name):
    if "gpt" in model_name:
        input_ids = torch.tensor([tokenizer(phrase)['input_ids']]).cuda()
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=input_ids).loss
        perplexity = math.exp(loss.item())
        return perplexity

    if "bert" in model_name:
        loss = 0
        input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(phrase) + ['[SEP]'])
        batch = []
        for i in range(len(input_ids)):
            input_ids_copy = input_ids.copy()
            input_ids_copy[i] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            input_tensor = torch.unsqueeze(torch.tensor(input_ids_copy), dim=0)
            batch.append(input_tensor)

        with torch.no_grad():
            predictions = model(torch.squeeze(torch.stack(batch)).cuda())[0]

        for i in range(1, len(input_ids)):
            masked_pred = predictions[i, i]
            probs = softmax(masked_pred.cpu().numpy())
            loss += np.log(probs[input_ids[i]])
        return loss


def count_unigram_frequencies(tok, texts):
  freq = defaultdict(int)
  for sent in texts:
    for token in tok.tokenize(sent):
      freq[token] += 1
  return freq


def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()


def calc_ppl_for_model(model_name):
    lp = []
    mean_lp = []
    pen_lp = []
    norm_lp = []
    slor = []

    if "gpt" in model_name:
        print('loading gpt')
        tok, model = load_tokenizer_and_model(model_name)
    else:
        print('loading bert')
        tok = transformers.AutoTokenizer.from_pretrained(model_name, from_tf=True)
        model = transformers.BertForMaskedLM.from_pretrained(model_name, from_tf=True).cuda()

    freqs = count_unigram_frequencies(tok, texts)
    failed = []
    for n, sentence in enumerate(texts):
        sent_len = len(tok(sentence))
        # try:
        ppl = calc_ppl(phrase=sentence, model=model, tokenizer=tok, model_name=model_name)
        lp.append(ppl)
        mean_lp.append(ppl / sent_len)
        pen_lp.append(ppl / ((5 + sent_len) * (5 + 1) ** 0.8))

        denom = 0.0
        for w in tok(sentence):
            denom += np.log(freqs[w] / tok.vocab_size)
        norm_lp.append(ppl / denom)
        slor.append((ppl - denom) / sent_len)
        # except BaseException:
        #  failed.append(n)
        if n % 100 == 0:
            print(n)

    return lp, mean_lp, pen_lp, norm_lp, slor, failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_path', default="../../../translations.txt", help='path to translations')
    parser.add_argument('--model_name', default='Geotrend/bert-base-ru-cased', help='name of pretrained language model')
    parser.add_argument('--dataset_part', default=1.0, type=float, help='part of translations to process')
    args = parser.parse_args()

    texts = []

    with open(args.tr_path, "rb") as inf:
        trigger = False
        for line in inf.readlines():
            eng, rus = line.decode("cp1251").split("%%##########%%")
            texts.append(rus.strip())

    texts = texts[:int(len(texts)*args.dataset_part)]
    print(len(texts))

    lp, mean_lp, pen_lp, norm_lp, slor, failed = calc_ppl_for_model(args.model_name)
    df = pd.DataFrame()
    df["lp"] = lp
    df["mean_lp"] = mean_lp
    df["pen_lp"] = pen_lp
    df["norm_lp"] = norm_lp
    df["slor"] = slor

    df.to_csv(f"metrics_{args.model_name}.csv")
    with open(f"failed_{args.model_name}.csv", 'w') as f:
        print(failed, file=f)

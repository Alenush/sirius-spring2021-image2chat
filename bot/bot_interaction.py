import torch
import argparse
import random
from evaluation import apply_model, rank_output_candidates, create_model_and_dataset


def process_data(ds, model, image_path, personality, dialogue_history, candidates_list):
    image_tensor = torch.unsqueeze(torch.squeeze(ds.img_loader.load(image_path)), dim=0)
    personality_ohe = torch.unsqueeze(ds.personality_to_tensor(personality), dim=0)
    samples_encoded, answers_encoded = apply_model(ds, model, image_tensor, personality_ohe, dialogue_history, candidates_list)
    top_answer = rank_output_candidates(samples_encoded, answers_encoded, candidates_list, None)
    return top_answer

cands = []
def parse_candidates(path):

    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            if len(line) > 1:
                cands.append(line[:-1])
                
    #random.shuffle(cands)
    #return cands[:1000]

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', default='/Users/isypov/Desktop/data/yfcc_images', type=str)
parser.add_argument('--dialogues_path', default='/Users/isypov/Desktop/Bot/testdata/', type=str)
parser.add_argument('--dict_path',
                        default='/Users/isypov/ParlAI/data/models/image_chat/transresnet_multimodal/model.dict',
                        type=str)
parser.add_argument('--personalities_path',
                        default='/Users/isypov/ParlAI/data/personality_captions/personalities.json',
                        type=str)
parser.add_argument('--model_path',
                        default='/Users/isypov/ParlAI/data/models/image_chat/model_resnext_1',
                        type=str)
parser.add_argument('--candidates_path',
                        default='/Users/isypov/ParlAI/data/models/image_chat/transresnet_multimodal/candidates.txt',
                        type=str)

args = parser.parse_args()
candidates_list = parse_candidates(args.candidates_path)

test_ds, model = create_model_and_dataset(args)


def AskModel(image_path, personality, dialogue_data):

    random.shuffle(cands)
    candidates_list = cands[:1000]
    #image_path = '/Users/isypov/Desktop/Bot/testdata/image.jpg'
    #personality = 'Fanatical'
    dialogue_history = [dialogue_data]
    Answer = process_data(test_ds, model, image_path, personality, dialogue_history, candidates_list)
    print(Answer)
    return(Answer)

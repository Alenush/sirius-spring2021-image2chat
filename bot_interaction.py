import torch
import argparse
from evaluation import apply_model, rank_output_candidates, create_model_and_dataset


def process_data(ds, model, image_path, personality, dialogue_history, candidates_list):
    image_tensor = torch.unsqueeze(torch.squeeze(ds.img_loader.load(image_path)), dim=0)
    personality_ohe = torch.unsqueeze(ds.personality_to_tensor(personality), dim=0)
    samples_encoded, answers_encoded = apply_model(ds, model, image_tensor, personality_ohe, dialogue_history, candidates_list)
    top_answer = rank_output_candidates(samples_encoded, answers_encoded, candidates_list, None)
    return top_answer


def parse_candidates(path):
    cands = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            if len(line) > 1:
                cands.append(line[:-1])
    return cands[:100]


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
    parser.add_argument('--candidates_path',
                        default='C://Users//daria.vinogradova//ParlAI//data//models//image_chat//transresnet_multimodal//candidates.txt',
                        type=str)

    args = parser.parse_args()
    candidates_list = parse_candidates(args.candidates_path)
    test_ds, model = create_model_and_dataset(args)
    image_path = 'C://Users//daria.vinogradova//ParlAI//data//yfcc_images//2923e28b6f588aff2d469ab2cccfac57.jpg'
    personality = 'Fanatical'
    dialogue_history = ['A little heavy on the make-up don\'t ya think.']
    print(process_data(test_ds, model, image_path, personality, dialogue_history, candidates_list))

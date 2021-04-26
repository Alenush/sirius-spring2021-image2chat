import json
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from PIL import Image
import sys
import subprocess

# from model import EfficentNetBackbone
import os

def save_tensor(state_dict, path):
  with open(path, "wb") as f:
    torch.save(state_dict, f)

IMAGE_MODE_SWITCHER = {
    'resnet152': ['resnet152', -1],
    'resnet101': ['resnet101', -1],
    'resnet50': ['resnet50', -1],
    'resnet34': ['resnet34', -1],
    'resnet18': ['resnet18', -1],
    'resnext101_32x8d_wsl': ['resnext101_32x8d_wsl', -1],
    'resnext101_32x16d_wsl': ['resnext101_32x16d_wsl', -1],
    'resnext101_32x32d_wsl': ['resnext101_32x32d_wsl', -1],
    'resnext101_32x48d_wsl': ['resnext101_32x48d_wsl', -1],
}


class ImageLoader:
    """
    Extract image feature using pretrained CNN network.
    """

    def __init__(self, opt):
        self.opt = opt.copy()
        self.netCNN = None
        self.image_mode = opt['image_mode']
        self.image_size = opt['image_size']
        self.crop_size = opt['image_cropsize']
        self.use_cuda = torch.cuda.is_available()
        self._init_transform(opt['split'])
        if 'resnet' in self.image_mode:
            self._init_resnet_cnn()
        elif 'resnext' in self.image_mode:
            self._init_resnext_cnn()
        elif 'efficientnet' in self.image_mode:
            self._init_efficientnet()

    def _init_transform(self, split):
        # initialize the transform function using torch vision.
        try:
            import torchvision
            import torchvision.transforms

            self.torchvision = torchvision
            self.transforms = torchvision.transforms

        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
            import torchvision
            import torchvision.transforms

            self.torchvision = torchvision
            self.transforms = torchvision.transforms
            # raise ImportError('Please install torchvision; see https://pytorch.org/')

        if split == "train":
            self.transform = self.transforms.Compose(
                [
                    self.transforms.Scale(self.image_size),
                    self.transforms.CenterCrop(self.crop_size),
                    self.transforms.ToTensor(),
                    self.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = self.transforms.Compose(
                [
                    self.transforms.Scale(self.crop_size),
                    self.transforms.ToTensor(),
                    self.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _init_resnet_cnn(self):
        """
        Lazily initialize preprocessor model.

        When image_mode is one of the ``resnet`` varieties
        """
        cnn_type, layer_num = self._image_mode_switcher()
        # initialize the pretrained CNN using pytorch.
        CNN = getattr(self.torchvision.models, cnn_type)

        # cut off the additional layer.
        self.netCNN = torch.nn.Sequential(*list(CNN(pretrained=True).children())[:layer_num])

        if self.use_cuda:
            self.netCNN.cuda()

    def _init_resnext_cnn(self):
        """
        Lazily initialize preprocessor model.

        When image_mode is one of the ``resnext101_..._wsl`` varieties
        """
        cnn_type, layer_num = self._image_mode_switcher()
        model = torch.hub.load('facebookresearch/WSL-Images', cnn_type)
        # cut off layer for ImageNet classification
        self.netCNN = torch.nn.Sequential(*list(model.children())[:layer_num])

        if self.use_cuda:
            self.netCNN.cuda()

    def _init_efficientnet(self):
        pass

    #    self.netCNN = EfficentNetBackbone()

    def _image_mode_switcher(self):
        return IMAGE_MODE_SWITCHER.get(self.image_mode)

    def extract(self, image, path=None):
        # check whether initialize CNN network.
        # extract the image feature
        transform = self.transform(image).unsqueeze(0)
        if self.use_cuda:
            transform = transform.cuda()
        with torch.no_grad():
            feature = self.netCNN(transform)
        # save the feature
        if path is not None:
            save_tensor(feature.cpu(), path)
        return feature

    def _load_image(self, path):
        """
        Return the loaded image in the path.
        """
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def _get_prepath(self, path):
        prepath, imagefn = os.path.split(path)
        return prepath, imagefn

    def load(self, path):
        """
        Load from a given path.
        """
        mode = self.opt['image_mode']

        # otherwise, looks for preprocessed version under 'mode' directory
        prepath, imagefn = self._get_prepath(path)
        dpath = os.path.join(prepath, mode)
        if not os.path.exists(dpath):
            os.mkdir(dpath)
        imagefn = imagefn.split('.')[0]
        new_path = os.path.join(prepath, mode, imagefn)
        if not os.path.exists(new_path):
            return self.extract(self._load_image(path), new_path)
        else:
            with open(new_path, 'rb') as f:
                return torch.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', default=0, type=int)
    args = parser.parse_args()
    img_loader = ImageLoader({
        'image_mode': 'resnext101_32x48d_wsl',  # 'efficientnet', #'resnext101_32x48d_wsl'
        'image_size': 256,
        'image_cropsize': 224,
        'split': 'train'
    })
    images_path = 'C://Users//miair//Downloads//yfcc_images//yfcc_images'
    with open('C://Users//miair//Desktop//Учёба//Сириус//Сириус3//image_chat//train.json') as f:
        data = json.load(f)
        data = data[args.part*50000:min(len(data), (args.part+1)*50000)]
        for i in tqdm(range(len(data))):
            elem = data[i]
            img_loader.load(images_path + '/' + elem['image_hash'] + '.jpg')
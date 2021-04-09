import torch
from PIL import Image

class ImageLoader:
    """
    Extract image feature using pretrained CNN network.
    """

    def __init__(self, opt):
        self.opt = opt.copy()
        self.use_cuda = False
        self.netCNN = None
        self.image_mode = opt['image_mode']
        self.image_size = opt['image_size']
        self.crop_size = opt['image_cropsize']
        self.use_cuda = torch.cuda.is_available()
        self._init_transform()
        if 'resnet' in self.image_mode:
            self._init_resnet_cnn()
        elif 'resnext' in self.image_mode:
            self._init_resnext_cnn()
        elif 'faster_r_cnn_152_32x8d' in self.image_mode:
            self._init_faster_r_cnn()

    def _init_transform(self):
        # initialize the transform function using torch vision.
        try:
            import torchvision
            import torchvision.transforms

            self.torchvision = torchvision
            self.transforms = torchvision.transforms

        except ImportError:
            raise ImportError('Please install torchvision; see https://pytorch.org/')

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

    def _image_mode_switcher(self):
        return IMAGE_MODE_SWITCHER.get(self.image_mode)

    def extract(self, image):
        # check whether initialize CNN network.
        # extract the image feature
        transform = self.transform(image).unsqueeze(0)
        if self.use_cuda:
            transform = transform.cuda()
        with torch.no_grad():
            feature = self.netCNN(transform)
        return feature

    def _load_image(self, path):
        """
        Return the loaded image in the path.
        """
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def load(self, path):
        """
        Load from a given path.
        """
        mode = self.opt['image_mode']
        if mode == 'raw':
            return self._load_image(path)

        return self.extract(self._load_image(path))

import torch
from torch import nn
from parlai.agents.transformer.modules import TransformerEncoder
import torch
from parlai.core.opt import Opt
from efficientnet_pytorch import EfficientNet


class TransresnetMultimodalModel(nn.Module):
    def __init__(self, dictionary, backbone_type=None):
        super().__init__()
        self.hidden_dim = 500
        self.image_features_dim = 2560 if backbone_type == "efficientnet" else 2048
        self.embedding_size = 300
        self.dropout = 0.2
        self.additional_layer_dropout = 0.2
        self.dictionary = dictionary
        self.num_personalities = 216
        self.use_personality = True

        self._build_image_encoder()
        self._build_personality_encoder()
        self._build_label_encoder()
        self.context_encoder = self._get_context_encoder()
        self.label_encoder = self._get_context_encoder()
        self.combine_layer = torch.empty(3, 1)
        nn.init.uniform_(self.combine_layer)

    def _build_image_encoder(self):
        image_layers = [
            nn.BatchNorm1d(self.image_features_dim),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.image_features_dim, self.hidden_dim),
        ]
        self.image_encoder = nn.Sequential(*image_layers)

    def _build_personality_encoder(self):
        personality_layers = [
            nn.BatchNorm1d(self.num_personalities),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.num_personalities, self.hidden_dim),
        ]
        self.personality_encoder = nn.Sequential(*personality_layers)

    def _get_context_encoder(self):
        embeddings = nn.Embedding(len(self.dictionary), self.embedding_size)
        return TransformerEncoder(
            #opt = Opt({'embedding_size': self.embedding_size,
            #           'ffn_size': self.embedding_size * 4,
            #           'n_layers': 4,
            #           'n_heads': 6}),
            embedding_size=self.embedding_size,
            ffn_size=self.embedding_size * 4,
            n_layers=4,
            n_heads=6,
            embedding=embeddings,
            vocabulary_size=len(self.dictionary),
            padding_idx=self.dictionary.tok2ind[self.dictionary.null_token],
            embeddings_scale=False,
            output_scaling=1.0,
            n_positions=1000
        )

    def _build_label_encoder(self):
        self.additional_layer = LinearWrapper(self.embedding_size, self.hidden_dim, self.additional_layer_dropout)

    def forward(self, images_tensor, personality_ohe, dialogue, labels):
        d_indexes, d_mask = dialogue
        l_indexes, l_mask = labels
        forward_image = self.image_encoder(images_tensor)
        if self.use_personality:
            forward_personality = self.personality_encoder(personality_ohe)
        else:
            forward_personality = torch.zeros_like(forward_image)
        forward_dialogue = self.additional_layer(self.context_encoder(d_indexes))
        forward_labels = self.additional_layer(self.label_encoder(l_indexes))
        combine = torch.mm(torch.stack((forward_dialogue, forward_image, forward_personality)), self.combine_layer)
        return combine, forward_labels


class LinearWrapper(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(LinearWrapper, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, input):
        return self.lin(self.dp(input))


class EfficentNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, X):
        return self.pool(self.model.extract_features(X))

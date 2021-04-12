from torch import nn
from parlai.agents.transformer.modules import TransformerEncoder


class TransresnetMultimodalModel(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.hidden_dim = 500
        self.image_features_dim = 2048
        self.embedding_size = 300
        self.dropout = 0.4
        self.dictionary = dictionary

        self._build_image_encoder()
        self._build_personality_encoder()
        self._build_context_encoder()
        self._build_label_encoder()

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

    def _build_context_encoder(self):
        embeddings = nn.Embedding(len(self.dictionary), self.embedding_size)
        self.context_encoder = TransformerEncoder(
            opt={
                'embedding_size': self.embedding_size,
                'ffn_size': self.embedding_size * 4,
                'n_layers': 4,
                'n_heads': 6
            },
            embedding=embeddings,
            vocabulary_size=len(self.dictionary),
            padding_idx=self.dictionary.tok2ind[self.dictionary.null_token],
            embeddings_scale=False,
            output_scaling=1.0,
        )

    def _build_label_encoder(self):
        pass

    def forward(self, images_tensor, personality_ohe, dialogue, labels):
        d_indexes, d_mask = dialogue
        l_indexes, l_mask = labels
        forward_image = self.image_encoder(images_tensor)
        forward_personality = self.personality_encoder(personality_ohe)
        forward_dialogue = self.context_encoder(d_indexes)
        forward_labels = self.label_encoder(l_indexes)
        return forward_dialogue + forward_image + forward_personality, forward_labels

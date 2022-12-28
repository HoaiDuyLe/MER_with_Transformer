import torch
from torch import nn
from transformers import AlbertModel

class MME2E_T(nn.Module):
    def __init__(self, feature_dim, num_classes=6, size='base'):
        super(MME2E_T, self).__init__()
        self.albert = AlbertModel.from_pretrained(f'albert-{size}-v2')
        self.text_cls_affine = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        self.text_tokens_affine = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, text, get_cls=False):
        last_hidden_state, _ = self.albert(**text)
        text_cls = self.text_cls_affine(last_hidden_state[:, 0])
        text_tokens = self.text_tokens_affine(last_hidden_state[:,1:])
        return text_cls, text_tokens
import torch
from torch import nn, Tensor
from src.models.e2e_t import MME2E_T
from src.utils import padTensor
from functools import partial
import math
from src.models.transformer_encoder import WrappedTransformerEncoder, BuildTransformerDecoder
from torchvision import transforms
from facenet_pytorch import MTCNN
from src.models.vgg_block import VggBasicBlock
from typing import Optional, Tuple
import torch.nn.functional as F

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class MME2E(nn.Module):
    def __init__(self, args, device):
        super(MME2E, self).__init__()        
        self.args = args
        self.mod = args['modalities'].lower()
        self.num_classes = args['num_emotions']
        self.device = device
        self.dim = args['trans_dim']
        self.num_modal = 3

        nlayers = args['trans_nlayers']
        nheads = args['trans_nheads']
        trans_dim = args['trans_dim']        

        self.T = MME2E_T(feature_dim=trans_dim, size=args['text_model_size'])

        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        self.V = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.A = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.v_flatten = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.a_flatten = nn.Sequential(
            nn.Linear(512 * 8 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, 
                                                num_layers=int(0.5*nlayers), num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, 
                                                num_layers=int(0.5*nlayers), num_heads=nheads)
        self.f_transformer = WrappedTransformerEncoder(dim=trans_dim,
                                                num_layers=int(0.5*nlayers), num_heads=nheads)
        
        # self.out = nn.Linear(trans_dim, self.num_classes)

        self.query_embed = nn.Embedding(self.num_classes, trans_dim)   
        self.f_decoder = BuildTransformerDecoder(dim=trans_dim,
                                                num_layers=int(0.5*nlayers), num_heads=nheads)
        self.fc = GroupWiseLinear(self.num_classes, trans_dim, bias=True)

    def forward(self, imgs, imgs_lens, specs, spec_lens, text):

        if 't' in self.mod:
            text_cls, text_tokens = self.T(text)            
            text_tokens_mask = ~text["attention_mask"].bool()[:,1:]
        # return self.out(text_cls)

        if 'v' in self.mod:
            faces = self.mtcnn(imgs)
            for i, face in enumerate(faces):
                if face is None:
                    center = self.crop_img_center(torch.tensor(imgs[i]).permute(2, 0, 1))
                    faces[i] = center
            faces = [self.normalize(face) for face in faces]
            faces = torch.stack(faces, dim=0).to(device=self.device)
            faces = self.V(faces)
            faces = self.v_flatten(faces.flatten(start_dim=1))
            face_cls, face_tokens, face_tokens_mask = self.v_transformer(faces,imgs_lens,
                                                                get_cls=True,pos_en=False)
        # return self.out(face_cls)

        if 'a' in self.mod:
            for a_module in self.A:
                specs = a_module(specs)
            specs = self.a_flatten(specs.flatten(start_dim=1))
            spec_cls, spec_tokens, spec_tokens_mask = self.a_transformer(specs,spec_lens, 
                                                                get_cls=True,pos_en=False)     
        # return self.out(spec_cls)  

        # return self.out(torch.cat((text_cls,face_cls,spec_cls), dim=1)), torch.cat((text_cls,face_cls,spec_cls), dim=1)
        
        tokens = torch.cat((text_tokens,face_tokens,spec_tokens), dim=1)        
        tokens_mask = torch.cat((text_tokens_mask,face_tokens_mask,spec_tokens_mask), dim=1)
        tokens_no_padding = tokens[~tokens_mask]
        tokens_mask_no_padding_lens = torch.count_nonzero(~tokens_mask, dim=1).tolist()
        fusion_cls, fusion_tokens, fusion_tokens_mask = self.f_transformer(
                                                tokens_no_padding,tokens_mask_no_padding_lens,
                                                get_cls=True,pos_en=False)  

        # return self.out(fusion_cls), fusion_cls
        # return self.out(torch.cat((text_cls,face_cls,spec_cls,fusion_cls), dim=1)), torch.cat((text_cls,face_cls,spec_cls,fusion_cls), dim=1)

        query_input = self.query_embed.weight
        hs = self.f_decoder(query_input, fusion_tokens.permute(1, 0, 2),
                            memory_key_padding_mask=fusion_tokens_mask)
        return self.fc(hs.permute(1, 0, 2)), hs

    def crop_img_center(self, img: torch.tensor, target_size=48):
        '''
        Some images have un-detectable faces, to make the training goes normally,
        for those images, we crop the center part,
        which highly likely contains the face or part of the face.

        @img - (channel, height, width)
        '''
        current_size = img.size(1)
        off = (current_size - target_size) // 2 # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped
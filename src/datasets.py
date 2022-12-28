import os
import glob
from random import sample
import torch
import torch.nn.functional as F
import torchaudio
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from typing import List, Dict, Tuple, Optional
from src.utils import load, padTensor, get_max_len
from PIL import Image

def getEmotionDict() -> Dict[str, int]:
    return {'ang': 0, 'exc': 1, 'fru': 2, 'hap': 3, 'neu': 4, 'sad': 5}

def get_dataset_iemocap(data_folder: str, phase: str, img_interval: int, hand_crafted_features: Optional[bool] = False):
    main_folder = os.path.join(data_folder, 'IEMOCAP_RAW_PROCESSED')
    meta = load(os.path.join(main_folder, 'meta.pkl'))

    emoDict = getEmotionDict()
    uttr_ids = open(os.path.join(data_folder, 'IEMOCAP_SPLIT', f'{phase}_split.txt'), 'r').read().splitlines()
    texts = [meta[uttr_id]['text'] for uttr_id in uttr_ids]
    labels = [emoDict[meta[uttr_id]['label']] for uttr_id in uttr_ids]

    this_dataset = IEMOCAP(
        main_folder=main_folder,
        utterance_ids=uttr_ids,
        texts=texts,
        labels=labels,
        label_annotations=list(emoDict.keys()),
        img_interval=img_interval
    )
    return this_dataset

class IEMOCAP(Dataset):
    def __init__(self, main_folder: str, utterance_ids: List[str], texts: List[List[int]], labels: List[int],
                 label_annotations: List[str], img_interval: int):
        super(IEMOCAP, self).__init__()
        self.utterance_ids = utterance_ids
        self.texts = texts
        self.labels = F.one_hot(torch.tensor(labels)).numpy()
        self.label_annotations = label_annotations

        self.utteranceFolders = {
            folder.split('/')[-1]: folder
            for folder in glob.glob(os.path.join(main_folder, '**/*'))
        }
        self.img_interval = img_interval

    def get_annotations(self) -> List[str]:
        return self.label_annotations

    def use_left(self, utteranceFolder: str) -> bool:
        entries = utteranceFolder.split('_')
        return entries[0][-1] == entries[-1][0]

    def sample_imgs_by_interval(self, folder: str, imgNamePrefix: str, fps: Optional[int] = 30) -> List[str]:
        '''
        Arguments:
            @folder - utterance folder
            @imgNamePrefix - prefix of the image name (determines L/R)
            @interval - how many ms per image frame
            @fps - fps of the original video
        '''
        files = sorted(glob.glob(f'{folder}/*.jpg'))
        nums = len(files)// 2
        
        if nums<=50:
            step=1
        elif nums>50 and nums<=100:
            step=2
        else:
            step=3
        sampled = [os.path.join(folder, f'{imgNamePrefix}{i}.jpg') for i in list(range(0,nums,step))]
        return sampled

    def cutSpecToPieces(self, spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))
        return specs

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def __len__(self):
        return len(self.utterance_ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        uttrId = self.utterance_ids[ind]
        uttrFolder = self.utteranceFolders[uttrId]
        use_left = self.use_left(uttrId)
        suffix = 'L' if use_left else 'R'
        audio_suffix = 'L' if use_left else 'R'
        imgNamePrefix = f'image_{suffix}_'

        sampledImgs = np.array([
            np.float32(Image.open(imgPath)) # .transpose()
            for imgPath in self.sample_imgs_by_interval(uttrFolder, imgNamePrefix)
        ])

        waveform, sr = torchaudio.load(os.path.join(uttrFolder, f'audio_{audio_suffix}.wav'))

        # Cut Spec
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=int(float(sr) / 16000 * 400))(waveform).unsqueeze(0)
        specgrams = self.cutSpecToPieces(specgram)

        return uttrId, sampledImgs, specgrams, self.texts[ind], self.labels[ind]


def get_dataset_mosei(data_folder: str, phase: str, img_interval: int):
    main_folder = os.path.join(data_folder, 'MOSEI_RAW_PROCESSED')
    meta = load(os.path.join(main_folder, 'meta.pkl'))

    ids = open(os.path.join(data_folder, 'MOSEI_SPLIT', f'{phase}_split.txt'), 'r').read().splitlines()
    texts = [meta[id]['text'] for id in ids]
    labels = [meta[id]['label'] for id in ids]

    return MOSEI(
        main_folder=main_folder,
        ids=ids,
        texts=texts,
        labels=labels,
        img_interval=img_interval
    )

class MOSEI(Dataset):
    def __init__(self, main_folder: str, ids: List[str], texts: List[List[int]], labels: List[int], img_interval: int):
        super(MOSEI, self).__init__()
        self.ids = ids
        self.texts = texts
        self.labels = np.array(labels)
        self.main_folder = main_folder
        self.img_interval = img_interval
        self.crop = transforms.CenterCrop(360)

    def get_annotations(self) -> List[str]:
        return ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']

    def getPosWeight(self):
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def sample_imgs_by_interval(self, folder: str, fps: Optional[int] = 30) -> List[str]:
        files = sorted(glob.glob(f'{folder}/*.jpg'),
                        key=lambda x: int(os.path.splitext(x)[0].split(' ')[0].split('_')[-1]))

        sampled = [os.path.join(folder, f'image_{i}.jpg') for i in list(range(0, len(files), 2))]
        return sampled

    def cutSpecToPieces(self, spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))
        return specs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        this_id = self.ids[ind]
        sample_folder = os.path.join(self.main_folder, this_id)

        sampledImgs = []
        for imgPath in self.sample_imgs_by_interval(sample_folder):
            this_img = Image.open(imgPath)
            H = np.float32(this_img).shape[0]
            W = np.float32(this_img).shape[1]
            if H > 360:
                resize = transforms.Resize([H // 2, W // 2])
                this_img = resize(this_img)
            this_img = self.crop(this_img)
            sampledImgs.append(np.float32(this_img))
        sampledImgs = np.array(sampledImgs)

        waveform, sr = torchaudio.load(os.path.join(sample_folder, f'audio.wav'))

        # Cut Spec
        specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            win_length=int(float(sr) / 16000 * 400),
            n_fft=int(float(sr) / 16000 * 400)
        )(waveform).unsqueeze(0)
        specgrams = self.cutSpecToPieces(specgram)

        return this_id, sampledImgs, specgrams, self.texts[ind], self.labels[ind]


def collate_fn(batch):
    utterance_ids = []
    texts = []
    labels = []

    newSampledImgs = None
    imgSeqLens = []

    specgrams = []
    specgramSeqLens = []

    for dp in batch:
        utteranceId, sampledImgs, specgram, text, label = dp
        if sampledImgs.shape[0] == 0:
            continue
        utterance_ids.append(utteranceId)
        texts.append(text)
        labels.append(label)

        imgSeqLens.append(sampledImgs.shape[0])
        newSampledImgs = sampledImgs if newSampledImgs is None else np.concatenate((newSampledImgs, sampledImgs), axis=0)

        specgramSeqLens.append(len(specgram))
        specgrams.append(torch.cat(specgram, dim=0))

    imgs = newSampledImgs

    return (
        utterance_ids,
        imgs,
        imgSeqLens,
        torch.cat(specgrams, dim=0),
        specgramSeqLens,
        texts,
        torch.tensor(labels, dtype=torch.float32)
    )

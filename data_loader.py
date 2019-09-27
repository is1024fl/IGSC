import torch

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

import os
from os.path import join as PJ
import numpy as np
import yaml
import random


def ConceptSets(state, concepts):

    split_list = state['split_list'] + ['general']
    data_dir = PJ('./dataset', state['dataset'], 'list')
    concept_file = pd.read_csv(PJ(data_dir, 'concepts', 'concepts_' + concepts + '.txt'))
    val_times = None if state['mode'] == 'train_test' else state['val_times']

    def _concept_split():
        if state['mode'] == "train_val":
            return yaml.load(open(PJ(data_dir, 'train_val', 'id_split' + str(val_times) + '.txt')))

        else:
            return yaml.load(open(PJ(data_dir, "train_test", 'id_split' + '.txt')))

    def _concept(split_mode, concept_split):

        if split_mode in ['train', 'trainval', 'test_seen']:
            concept_label = concept_split['train_id']

        elif split_mode in ['val', 'test', 'test_unseen']:
            concept_label = concept_split['test_id']

        elif split_mode in ['general']:
            concept_label = list(range(concept_file.shape[1]))
        else:
            assert "Split Mode Error"

        concept_vector = [torch.cuda.FloatTensor(concept_file.iloc[:, i].values) for i in concept_label]

        return {'label': concept_label, 'vector': torch.stack(concept_vector)}

    concept_split = _concept_split()
    return {s: _concept(s, concept_split) for s in split_list}


def ClassDatasets(state):

    split_list = state['split_list']

    class ClassDataset(Dataset):

        def __init__(self, split_mode, state):

            self.dataset = state['dataset']

            self.split_mode = split_mode
            self.root = PJ('./dataset', state['dataset'])

            self.file_name = str(state['val_times']) + '.txt' if state.get('val_times') else '.txt'
            self.csv_file = PJ(self.root, 'list', state['mode'], self.split_mode.strip("_g") + self.file_name)
            self.data = pd.read_csv(self.csv_file, header=None)

            self.img_transform = self.img_transform()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):

            image = Image.open(PJ(self.root, 'img', self.data.iloc[idx, 0])).convert('RGB')
            label = torch.FloatTensor(self.data.iloc[idx, 1:].tolist())
            sample = {'image': self.img_transform(image), 'label': label}
            return sample

        def img_transform(self):

            if self.split_mode.find("train") != -1:
                img_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])
            else:
                img_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])

            return img_transform

    return {s: ClassDataset(s, state) for s in split_list}

import torch
import numpy as np
import random

torch.manual_seed(64)
torch.cuda.manual_seed(64)
np.random.seed(64)
random.seed(64)

torch.backends.cudnn.deterministic = True

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader

import torch.optim as optim
from model import visual_semantic_model, model_epoch

import utils

import numpy as np
from os.path import join as PJ
import yaml
import json
from tensorboardX import SummaryWriter

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    # setting
    CONFIG = yaml.load(open("train_val_config.yaml"))

    EXP_NAME = CONFIG['exp_name']

    DATASET = CONFIG['dataset']
    CONCEPTS = CONFIG['concepts']

    SAVE_PATH = PJ('.', 'runs', DATASET, EXP_NAME)

    L_RATE = np.float64(CONFIG['l_rate'])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = None

    ##########################################################################################

    # random validation for three times
    for val_times in range(3):

        # state
        STATE = {
            'dataset': DATASET,
            'mode': 'train_val',
            'split_list': ['train', 'val'],
            'val_times': val_times + 1
        }

        # build model
        model = visual_semantic_model(backbone=CONFIG['model'],
                                      k=CONFIG['k'], d=CONFIG['d'][DATASET],
                                      pretrained=CONFIG['pretrained'], freeze=CONFIG['freeze'])
        model = model.to(DEVICE)

        # data setting
        concepts = ConceptSets(STATE, CONCEPTS)
        datasets = ClassDatasets(STATE)

        train_loader = DataLoader(datasets['train'], batch_size=CONFIG['train_batch_size'], shuffle=True)
        val_loader = DataLoader(datasets['val'], batch_size=CONFIG['val_batch_size'], shuffle=False)

        ##########################################################################################

        # optim setting
        params = [{
            'params': model.transform.parameters() if CONFIG['freeze'] else model.parameters()
        }]

        if CONFIG['optim'] == 'SGD':
            optimizer = optim.SGD(params, np.float64(CONFIG['l_rate']), momentum=CONFIG['momentum'])

        elif CONFIG['optim'] == 'Adam':
            optimizer = optim.Adam(params, np.float64(CONFIG['l_rate']))

        for epoch in range(1, CONFIG['end_epoch']):

            writer = SummaryWriter(PJ(SAVE_PATH, 'val' + str(val_times)))

            # training
            train_metrics = model_epoch(loss_name="train", epoch=epoch, model=model,
                                        data_loader=train_loader, concepts=concepts,
                                        optimizer=optimizer, writer=writer)

            for g in [False, True]:
                record_name = 'train_g' if g else 'train'
                train_class, train_acc = utils.cal_acc(train_metrics, g)
                writer.add_scalar(record_name + '_acc', train_acc * 100, epoch)

            ######################################################################################

            # val
            record = {'val': {'acc': 0.0, 'class': None}}
            record.update({'val_g': {'acc': 0.0, 'class': None}})

            val_metric = model_epoch(loss_name="val", epoch=epoch, model=model,
                                     data_loader=val_loader, concepts=concepts,
                                     optimizer=optimizer, writer=writer)

            for g in [False, True]:
                record_name = 'val_g' if g else 'val'
                val_class, val_acc = utils.cal_acc(val_metric, g)
                record[record_name]['acc'] = val_acc
                record[record_name]['class'] = val_class

                writer.add_scalar(record_name + '_acc', val_acc * 100, epoch)

                with open(PJ(SAVE_PATH, 'val' + str(val_times), "test_table_" + str(epoch) + ".txt"), "w") as f:
                    table = json.dump({record_name: val_class}, f)

            writer.close()

import torch

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader
import torch.optim as optim
from model import classifier, transformer, DeVise, model_epoch
import utils

from os.path import join as PJ
import yaml
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
from collections import deque

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(1)

if __name__ == '__main__':

    # load config
    CONFIG = yaml.load(open("test.yaml"))

    print(CONFIG['exp_name'])

    for DATASET in ['sun', 'cub', 'awa2', 'apy']:

        EXP_NAME = CONFIG['exp_name'] + '_' + CONFIG['type']
        CONCEPTS = CONFIG['concepts']

        # state
        STATE = {
            'dataset': DATASET,
            'mode': 'train_test',
            'split_list': ['trainval',
                           'test_seen', 'test_unseen']
        }

        concepts = ConceptSets(STATE, CONCEPTS)
        datasets = ClassDatasets(STATE)

        # if CONFIG['skewness']:
        #     print(DATASET + " skewness:")
        #     for tn in STATE['split_list']:
        #         df = datasets[tn].data.iloc[:, 1:].sum(axis=0)
        #         print(tn + ": " + str(df[df > 0].skew()))

        train_loader = DataLoader(datasets['trainval'],
                                  batch_size=CONFIG['train_batch_size'], shuffle=True)

        test_loaders = {tn: DataLoader(datasets[tn],
                                       batch_size=CONFIG['test_batch_size'], shuffle=False)
                        for tn in STATE['split_list'][1:]}

        ##########################################################################################
        # experiment for n times
        for exp_times in range(1):

            SAVE_PATH = PJ('.', 'runs_test', DATASET, EXP_NAME, str(exp_times))

            # set experiment type: classifier / transformer
            if CONFIG['type'] == "classifier":
                model = classifier(backbone=CONFIG['model'],
                                   k=CONFIG['k'], d=CONFIG['d'][CONFIG['concepts']][DATASET],
                                   pretrained=CONFIG['pretrained'], freeze=CONFIG['freeze'])

            elif CONFIG['type'] == "transformer":
                model = DeVise(backbone=CONFIG['model'], d=CONFIG['d'][CONFIG['concepts']][DATASET],
                               pretrained=CONFIG['pretrained'], freeze=CONFIG['freeze'])
            else:
                assert False, "Must Assign the model type: classifier or transformer"

            # load model weight
            if CONFIG['load_model']:
                state = torch.load(PJ(SAVE_PATH, 'best_result.pkl'))
                epoch = state['epoch']

                model.load_state_dict(state['state_dict'])

            model = model.to(DEVICE)

            # test
            # record = {i: {'acc': 0.0, 'class': None} for i in ['conv', 'general']}
            test_metrics = {
                'total': deque(),
                'correct': deque(),
                'total_g': deque(),
                'correct_g': deque()
            }

            for tn in STATE['split_list'][1:]:

                test_metric = model_epoch(loss_name=tn, epoch=epoch,
                                          model=model, type=CONFIG['type'],
                                          data_loader=test_loaders[tn], concepts=concepts,
                                          optimizer=None, writer=None, debug=CONFIG['debug'])

                for k in test_metrics.keys():
                    v = np.asarray(test_metric[k]).argmax(axis=1)
                    test_metrics[k].extend(v)

            print(DATASET + ' skewness:')

            for g in [False, True]:
                test_skew = utils.skewness(test_metrics, g)
                print(test_skew)

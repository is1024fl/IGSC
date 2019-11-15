import torch

from data_loader import ClassDatasets, ConceptSets
from torch.utils.data import DataLoader
import torch.optim as optim
from model import DeVise, model_epoch
import utils

from os.path import join as PJ
import yaml
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # load config
    CONFIG = yaml.load(open("devise.yaml"))

    EXP_NAME = CONFIG['exp_name'] + '_' + CONFIG['type']
    DATASET = CONFIG['dataset']
    CONCEPTS = CONFIG['concepts']

    # state
    STATE = {
        'dataset': DATASET,
        'mode': 'train_test',
        'split_list': ['trainval',
                       'test_seen', 'test_unseen']
    }

    # load
    print("load data")

    concepts = ConceptSets(STATE, CONCEPTS)
    datasets = ClassDatasets(STATE)

    if CONFIG['skewness']:
        print(DATASET + " skewness:")
        for tn in STATE['split_list']:
            df = datasets[tn].data.iloc[:, 1:].sum(axis=0)
            print(tn + ": " + str(df[df > 0].skew()))

    train_loader = DataLoader(datasets['trainval'],
                              batch_size=CONFIG['train_batch_size'], shuffle=True)

    test_loaders = {tn: DataLoader(datasets[tn],
                                   batch_size=CONFIG['test_batch_size'], shuffle=False)
                    for tn in STATE['split_list'][1:]}

    ##########################################################################################
    # experiment for n times
    for exp_times in range(CONFIG['exp_times']):

        SAVE_PATH = PJ('.', 'runs_test', DATASET, EXP_NAME, str(exp_times))
        writer = SummaryWriter(PJ(SAVE_PATH))

        # set experiment type: classifier / transformer
        model = DeVise(backbone=CONFIG['model'], d=CONFIG['d'][CONFIG['concepts']][DATASET],
                       pretrained=CONFIG['pretrained'], freeze=CONFIG['freeze'])

        # load model weight
        if CONFIG['load_model']:
            print("Loading pretrained model")
            state = torch.load(PJ(SAVE_PATH, 'best_result.pkl'))

            # load model epoch
            CONFIG['start_epoch'] = state['epoch']
            assert CONFIG['end_epoch'] > CONFIG['start_epoch'], \
                ("The start epoch is {}, and the end epoch is smaller than start epoch.", state['epoch'])

            # load model parameter
            model.load_state_dict(state['state_dict'])

        model = model.to(DEVICE)

        # optim setting
        params = [{
            'params': model.transform.parameters() if CONFIG['freeze'] else model.parameters()
        }]

        if CONFIG['optim'] == 'SGD':
            optimizer = optim.SGD(params, float(CONFIG['l_rate']), momentum=CONFIG['momentum'])

        elif CONFIG['optim'] == 'Adam':
            optimizer = optim.Adam(params, float(CONFIG['l_rate']))

        else:
            assert False, "Must Assign the optimizer type: SGD or Adam."

        if CONFIG['load_model']:
            optimizer.load_state_dict(state['optimizer'])

        # optim scheduler
        scheduler = None
        if CONFIG['type'] == "transformer":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # record best result
        BEST_RESULT = {
            "h_acc": 0,
            "epoch": 0
        }

        for epoch in range(CONFIG['start_epoch'], CONFIG['end_epoch']):

            # train
            train_metrics = model_epoch(loss_name="trainval", epoch=epoch,
                                        model=model, type=CONFIG['type'], neg_sample=CONFIG['neg_sample'],
                                        data_loader=train_loader, concepts=concepts, use_smooth=False, margin=0.1,
                                        optimizer=optimizer, writer=writer, debug=CONFIG['debug'])

            for g in [False, True]:
                record_name = 'train_g' if g else 'train'
                train_class, train_acc = utils.cal_acc(train_metrics, g)
                writer.add_scalar(record_name + '_acc', train_acc * 100, epoch)

                if CONFIG['skewness']:
                    train_skew = utils.skewness(train_metrics, g)
                    writer.add_scalar(record_name + '_skewness', train_skew, epoch)

            ######################################################################################
            # test
            record = {tn: {'acc': 0.0, 'class': None} for tn in STATE['split_list'][1:]}
            record.update({tn + '_g': {'acc': 0.0, 'class': None} for tn in STATE['split_list'][1:]})

            for tn in STATE['split_list'][1:]:

                test_metric = model_epoch(loss_name=tn, epoch=epoch,
                                          model=model, type=CONFIG['type'],
                                          data_loader=test_loaders[tn], concepts=concepts,
                                          optimizer=optimizer, writer=writer, debug=CONFIG['debug'])

                for g in [False, True]:
                    test_class, test_acc = utils.cal_acc(test_metric, g)
                    record_name = tn + '_g' if g else tn
                    record[record_name]['acc'] = test_acc
                    record[record_name]['class'] = test_class

                    writer.add_scalar(record_name + '_acc', test_acc * 100, epoch)

                    if CONFIG['skewness']:
                        test_skew = utils.skewness(test_metric, g)
                        writer.add_scalar(record_name + '_skewness', test_skew, epoch)

            tmp_h_acc = utils.cal_h_acc(record, True)
            writer.add_scalar('H_acc', 100 * tmp_h_acc, epoch)

            ######################################################################################
            # record best result
            if tmp_h_acc > BEST_RESULT['h_acc']:
                BEST_RESULT['h_acc'] = tmp_h_acc
                BEST_RESULT['epcoh'] = epoch

                save_state = {
                    'epoch': epoch,
                    'h_acc': tmp_h_acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(save_state, PJ(SAVE_PATH, 'best_result.pkl'))

            if scheduler:
                scheduler.step()


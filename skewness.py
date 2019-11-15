import os
import yaml
import numpy as np
import pandas as pd
import re
from collections import deque
import torch
from os.path import join as PJ
import tensorflow as tf




if __name__ == '__main__':

    torch.cuda.set_device(1)

    DATASET = 'cub'
    EXP_NAME ='nonlinear_resnet101'
    # EXP_NAME = 'devise_resnet101'
    TYPE = 'classifier'
    # TYPE = 'transformer'

    exp_times = 1

    records = dict()
    for i in range(exp_times):
        SAVE_PATH = PJ('.', 'runs_test', DATASET, EXP_NAME + '_' + TYPE, str(i))
        state = torch.load(PJ(SAVE_PATH, 'best_result.pkl'), torch.device('cuda:1'))
        print(f"time {i} => epoch {state['epoch']} has max H_acc: {state['h_acc']:.3f}")

        event_file = [d for d in os.listdir(SAVE_PATH) if re.search('event', d)][0]

        for e in tf.train.summary_iterator(PJ(SAVE_PATH, event_file)):
            for v in e.summary.value:
                if re.search('skewness', v.tag) and e.step == state['epoch']:
                    if v.tag not in records:
                        records[v.tag] = v.simple_value
                    else:
                        records[v.tag] += v.simple_value

    records = {r: records[r] / exp_times for r in records}
    records = pd.DataFrame.from_dict(records, orient='index', columns=['skewness'])
    print(records)
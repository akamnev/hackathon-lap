"""Обучение простой модели предсказания"""
import os
import argparse
import json
import numpy as np
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from hack_lap.utils.dataset import get_eval_dataset, get_gcn_dataset
from hack_lap.utils.evaluate import calculate_metrics, evaluate_pred
from hack_lap.models.gcn import GCNModel

DIR = os.path.split(os.path.abspath(__file__))
DIR = os.path.split(DIR[0])[0]
DIR_MODULE = os.path.split(DIR)[0]
DIR_DATA = os.path.join(DIR, 'data')
DIR_MODEL = os.path.join(DIR_DATA, 'model')
DIR_PREDICT = os.path.join(DIR_DATA, 'predict')


def calculate_threshold(
        model, filename, dev_frac, seed, positive_multiplier, batch_size,
        device, repeat
):
    filename = os.path.join(DIR_DATA, filename)
    _, dataset_dev = get_gcn_dataset(
        filename,
        dev_frac=dev_frac,
        positive_multiplier=positive_multiplier,
        seed=seed
    )
    loader = DataLoader(dataset_dev, batch_size=batch_size)
    _, _, (pr, re, f1), th = calculate_metrics(model, loader, device, repeat)
    return th, (pr, re, f1)


def main(
    exp_name,
    batch_size,
    device,
    filename,
    dev
):
    filename = os.path.join(DIR_DATA, filename)
    dataset = get_eval_dataset(filename)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with open(os.path.join(DIR_MODEL, exp_name + '.json')) as fp:
        cfg = json.load(fp)

    cfg_model = {
        'num_hidden_layers': cfg['num_hidden_layers'],
        'hidden_size': cfg['hidden_size'],
        'intermediate_size': cfg['intermediate_size'],
        'pooling': cfg['pooling'],
        'normalize': cfg['normalize'],
        'bias': cfg['bias'],
    }
    model = GCNModel(**cfg_model)
    model.load_state_dict(
        torch.load(
            os.path.join(DIR_MODEL, exp_name + '.pt'),
            map_location='cpu'
        )
    )
    model.train()
    model.to(device)

    th, stat = calculate_threshold(
        model,
        filename=dev,
        dev_frac=cfg['dev_frac'],
        seed=cfg['seed'],
        positive_multiplier=cfg['positive_multiplier'],
        batch_size=batch_size,
        device=device,
        repeat=32
    )
    print("pr: %s, re: %s, f1: %s" % stat)

    yp = evaluate_pred(model, loader, device, repeat=32)
    yp = np.mean(yp, axis=1).ravel()
    yp = (yp > th).astype(int)
    df = pd.read_csv(os.path.join(DIR_DATA, 'test.csv'))
    df['Active'] = yp
    df.to_csv(os.path.join(DIR_PREDICT, exp_name + '.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', dest='exp_name', required=True)
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
    parser.add_argument('--device', dest='device', default='cpu')
    parser.add_argument('--filename', dest='filename', default='test.pkl')
    parser.add_argument('--dev', dest='dev', default='train.pkl')

    params = parser.parse_args()

    main(
        exp_name=params.exp_name,
        batch_size=params.batch_size,
        filename=params.filename,
        dev=params.dev,
        device=params.device,
    )

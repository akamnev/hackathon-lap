"""Обучение простой модели предсказания"""
import os
import argparse
import json
from tqdm.auto import trange
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from hack_lap.models.gcn import GCNModel
from hack_lap.utils.dataset import get_gcn_dataset
from hack_lap.utils.train import kld_loss

DIR = os.path.split(os.path.abspath(__file__))
DIR = os.path.split(DIR[0])[0]
DIR_MODULE = os.path.split(DIR)[0]
DIR_DATA = os.path.join(DIR, 'data')
DIR_LOG = os.path.join(DIR_DATA, 'log')
DIR_MODEL = os.path.join(DIR_DATA, 'model')


@torch.no_grad()
def evaluate(model, loader, device, eps=1e-6):
    fn_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    sigmoid = torch.nn.Sigmoid()
    bce_loss = []
    yt, yp = [], []
    for batch in loader:
        batch.to(device)
        out = model(batch)
        log_proba = out[0].view(-1)
        loss = fn_loss(log_proba, batch.y)
        bce_loss.append(loss)

        proba = sigmoid(log_proba)
        yp.append((proba > 0.5).to(torch.float32))
        yt.append(batch.y.to(torch.float32))
    bce_loss = torch.cat(bce_loss)
    bce_loss = torch.mean(bce_loss)

    yt = torch.cat(yt)
    yp = torch.cat(yp)
    tp = (yt * yp).sum()
    # tn = ((1.0 - yt) * (1.0 - yp)).sum()
    fp = ((1.0 - yt) * yp).sum()
    fn = (yt * (1.0 - yp)).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return bce_loss, (precision, recall, f1)


def main(
    tb_writer=None,
    exp_name=None,
    positive_multiplier=20,
    dev_frac=0.1,
    batch_size=64,
    num_epoch=2,
    lr=1e-4,
    wd=1e-1,
    weight_kld=1e-5,
    device='cpu',
    filename='train.pkl',
    seed=None,
):
    filename = os.path.join(DIR_DATA, filename)
    dataset_train, dataset_dev = get_gcn_dataset(
        filename,
        dev_frac=dev_frac,
        positive_multiplier=positive_multiplier,
        seed=seed
    )
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_dev = DataLoader(dataset_dev, batch_size=batch_size)

    cfg_model = {
        'num_hidden_layers': 4,
        'hidden_size': 32,
        'intermediate_size': 128,
        'pooling': 'mean',
        'normalize': False,
        'bias': False,
    }
    model = GCNModel(**cfg_model)
    model.train()
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )
    fn_loss = torch.nn.BCEWithLogitsLoss()

    cnt = 0
    for n_epoch in trange(num_epoch):
        for batch in loader_train:
            batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            log_proba = out[0].view(-1)
            loss_pred = fn_loss(log_proba, batch.y)

            kld = kld_loss(model, 'kld', {'nu': 0.0, 'rho': 1.0})
            loss = loss_pred + weight_kld * kld
            loss.backward()

            opt.step()

            if cnt % 5 == 0:
                tb_writer.add_scalar(
                    f'{exp_name}/loss-train', loss_pred.item(), cnt)
                tb_writer.add_scalar(
                    f'{exp_name}/kld', kld.item(), cnt)
            cnt += 1
        loss_pos, (pr, re, f1) = evaluate(model, loader_dev, device)
        tb_writer.add_scalar(
            f'{exp_name}/loss-eval', loss_pos.item(), n_epoch)
        tb_writer.add_scalar(
            f'{exp_name}/precision', pr.item(), n_epoch)
        tb_writer.add_scalar(
            f'{exp_name}/recall', re.item(), n_epoch)
        tb_writer.add_scalar(
            f'{exp_name}/f1', f1.item(), n_epoch)

    torch.save(
        model.state_dict(),
        os.path.join(DIR_MODEL, f'mdl-{exp_name}-{num_epoch}.pt')
    )
    cfg = {
        'positive_multiplier': positive_multiplier,
        'dev_frac': dev_frac,
        'batch_size': batch_size,
        'num_epoch': num_epoch,
        'lr': lr,
        'wd': wd,
        'weight_kld': weight_kld,
        'seed': seed,
    }
    cfg.update(cfg_model)
    with open(os.path.join(DIR_MODEL, f'mdl-{exp_name}-{num_epoch}.json'), 'w') as fp:
        json.dump(cfg, fp, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', dest='exp_name', required=True)
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', default=1, type=int)
    parser.add_argument('--device', dest='device', default='cpu')
    parser.add_argument('--positive_multiplier', dest='positive_multiplier', default=20, type=int)
    parser.add_argument('--dev_frac', dest='dev_frac', default=0.1, type=float)
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float)
    parser.add_argument('--wd', dest='wd', default=1e-1, type=float)
    parser.add_argument('--weight_kld', dest='weight_kld', default=1e-6, type=float)
    parser.add_argument('--filename', dest='filename', default='train.pkl')
    parser.add_argument('--seed', dest='seed', default=None, type=int)

    params = parser.parse_args()

    with SummaryWriter(log_dir=DIR_LOG) as tb_writer_:
        main(
            tb_writer=tb_writer_,
            exp_name=params.exp_name,
            batch_size=params.batch_size,
            num_epoch=params.num_epoch,
            positive_multiplier=params.positive_multiplier,
            dev_frac=params.dev_frac,
            lr=params.lr,
            wd=params.wd,
            weight_kld=params.weight_kld,
            filename=params.filename,
            seed=params.seed,
            device=params.device
        )

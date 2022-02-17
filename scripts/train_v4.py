"""Схема обучения: учим модель, каждые 500 эпох измеряем ее точность, сохраняем
модель и результат, если модель не улучшает точность 5000 эпох, начинаем обучение
заново. Если на измерении модель показала точность хуже порога, учим еще раз.
"""
import os
import argparse
import json
import pickle
from tqdm.auto import trange
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from hack_lap.models.gcn import GCNModel
from hack_lap.utils.dataset import get_gcn_dataset, get_eval_dataset
from hack_lap.utils.train import kld_loss
from hack_lap.utils.evaluate import calculate_metrics_one_vs_rest, \
    evaluate_pred, evaluate, calculate_metrics_one_vs_rest_

DIR = os.path.split(os.path.abspath(__file__))
DIR = os.path.split(DIR[0])[0]
DIR_MODULE = os.path.split(DIR)[0]
DIR_DATA = os.path.join(DIR, 'data')
DIR_LOG = os.path.join(DIR_DATA, 'log')
DIR_MODEL = os.path.join(DIR_DATA, 'model')


def save_train_data(cfg, model, yt_dev, yp_dev, yp_test, output_filename):
    torch.save(
        model.state_dict(),
        os.path.join(DIR_MODEL, output_filename + '.pt')
    )
    with open(os.path.join(DIR_MODEL, output_filename + '.json'), 'w') as fp:
        json.dump(cfg, fp, indent=2)
    output = {
        'yt_dev': yt_dev,
        'yp_dev': yp_dev,
        'yp_test': yp_test
    }
    with open(os.path.join(DIR_MODEL, output_filename + '.pkl'), 'wb') as fp:
        pickle.dump(output, fp, protocol=4)


def main(
    n_try,
    tb_writer,
    exp_name,
    positive_multiplier,
    dev_frac,
    dev_reverse,
    batch_size,
    num_epoch,
    wait_epoch,
    repeat,
    lr,
    wd,
    weight_kld,
    weight_entropy,
    seed,
    device,
    filename_train,
    filename_test,
):
    exp_name += f'-seed-{seed}-r-{dev_reverse}-n-{n_try}'
    dataset_train, dataset_dev = get_gcn_dataset(
        filename=os.path.join(DIR_DATA, filename_train),
        dev_frac=dev_frac,
        dev_reverse=bool(dev_reverse),
        positive_multiplier=positive_multiplier,
        seed=seed
    )
    dataset_test = get_eval_dataset(
        filename=os.path.join(DIR_DATA, filename_test)
    )
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # model name: Layer-N Hidden-N Pooling-type, Normalize bool bias bool
    cfg_model = {
        'num_hidden_layers': 4,
        'hidden_size': 32,
        'intermediate_size': 128,
        'pooling': 'attention',
        'normalize': False,
        'bias': False,
    }
    cfg = {
        'positive_multiplier': positive_multiplier,
        'dev_frac': dev_frac,
        'dev_reverse': dev_reverse,
        'batch_size': batch_size,
        'num_epoch': num_epoch,
        'repeat': repeat,
        'lr': lr,
        'wd': wd,
        'weight_kld': weight_kld,
        'weight_entropy': weight_entropy,
        'seed': seed,
    }
    cfg.update(cfg_model)

    model = GCNModel(**cfg_model)
    model.train()
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )
    fn_loss = torch.nn.BCEWithLogitsLoss()
    fn_proba = torch.nn.Sigmoid()

    best_f1_coarse, best_f1_smooth = 0.0, 0.0
    best_n_epoch = 0
    evaluate_score = False
    for n_epoch in trange(num_epoch):
        for batch in loader_train:
            batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            log_proba = out[0].view(-1)
            loss_pred = fn_loss(log_proba, batch.y)

            kld = kld_loss(model, 'kld', {'nu': 0.0, 'rho': 1.0})
            loss = loss_pred + weight_kld * kld
            loss_norm = [1.0 + weight_kld]

            if weight_entropy > 0.0:
                proba = fn_proba(log_proba)
                entropy = torch.mean(proba * log_proba)
                loss = loss + weight_entropy * entropy
                loss_norm.append(weight_entropy)

            loss = loss / sum(loss_norm)
            loss.backward()

            opt.step()

        if n_epoch % 10 == 0:
            tb_writer.add_scalar(f'{exp_name}/loss-train', loss_pred, n_epoch)
            tb_writer.add_scalar(f'{exp_name}/kld', kld, n_epoch)
            bce_loss, (precision_0, recall_0, f1_0), (precision_1, recall_1, f1_1) \
                = calculate_metrics_one_vs_rest(model, loader_dev, device)
            tb_writer.add_scalar(
                f'{exp_name}/loss-eval', bce_loss, n_epoch)
            tb_writer.add_scalar(
                f'{exp_name}/precision-0', precision_0, n_epoch)
            tb_writer.add_scalar(
                f'{exp_name}/recall-0', recall_0, n_epoch)
            tb_writer.add_scalar(
                f'{exp_name}/f1-0', f1_0, n_epoch)
            tb_writer.add_scalar(
                f'{exp_name}/precision-1', precision_1, n_epoch)
            tb_writer.add_scalar(
                f'{exp_name}/recall-1', recall_1, n_epoch)
            tb_writer.add_scalar(
                f'{exp_name}/f1-1', f1_1, n_epoch)

            if f1_1 > best_f1_coarse:
                best_f1_coarse = f1_1
                evaluate_score = True

        if (n_epoch + 1) % 500 == 0 or evaluate_score:

            evaluate_score = False
            # evaluate on dev
            _, yt_dev, yp_dev = evaluate(model, loader_dev, device, repeat=repeat)
            # evaluate on test
            yp_test = evaluate_pred(model, loader_test, device, repeat=repeat)
            (_, recall_0, _), (_, recall_1, f1_1) = calculate_metrics_one_vs_rest_(yt_dev, yp_dev)
            if f1_1 > best_f1_smooth:
                best_f1_smooth = f1_1
                best_n_epoch = n_epoch
                output_filename = f'{exp_name}-best'
                cfg['best_epoch'] = n_epoch
                cfg['best_f1'] = best_f1_smooth
                cfg['best_recall_0'] = recall_0
                cfg['best_recall_1'] = recall_1
                save_train_data(cfg, model, yt_dev, yp_dev, yp_test, output_filename)
            tb_writer.add_scalar(
                f'{exp_name}/f1-best', best_f1_smooth, n_epoch)

        if best_n_epoch + wait_epoch <= n_epoch:
            break

    output_filename = f'{exp_name}-Nepoch-{n_epoch + 1}'
    # evaluate on dev
    _, yt_dev, yp_dev = evaluate(model, loader_dev, device, repeat=repeat)
    # evaluate on test
    yp_test = evaluate_pred(model, loader_test, device, repeat=repeat)

    save_train_data(cfg, model, yt_dev, yp_dev, yp_test, output_filename)
    return best_f1_smooth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', dest='exp_name', required=True)
    parser.add_argument('--batch_size', dest='batch_size', default=256, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', default=20000, type=int)
    parser.add_argument('--wait_epoch', dest='wait_epoch', default=5000, type=int)
    parser.add_argument('--repeat', dest='repeat', default=32, type=int)
    parser.add_argument('--device', dest='device', default='cpu')
    parser.add_argument('--positive_multiplier', dest='positive_multiplier', default=20, type=int)
    parser.add_argument('--dev_frac', dest='dev_frac', default=0.25, type=float)
    parser.add_argument('--dev_reverse', dest='dev_reverse', default=0, type=int)
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float)
    parser.add_argument('--wd', dest='wd', default=1e-1, type=float)
    parser.add_argument('--weight_kld', dest='weight_kld', default=2e-6, type=float)
    parser.add_argument('--weight_entropy', dest='weight_entropy', default=0.0, type=float)
    parser.add_argument('--threshold_f1', dest='threshold_f1', default=0.4, type=float)
    parser.add_argument('--filename_train', dest='filename_train', default='train.pkl')
    parser.add_argument('--filename_test', dest='filename_test', default='test.pkl')
    parser.add_argument('--seed', dest='seed', default=None, type=int)

    params = parser.parse_args()

    with SummaryWriter(log_dir=DIR_LOG) as tb_writer_:
        for n in range(3):
            best_f1 = main(
                n_try=n,
                tb_writer=tb_writer_,
                exp_name=params.exp_name,
                batch_size=params.batch_size,
                num_epoch=params.num_epoch,
                wait_epoch=params.wait_epoch,
                repeat=params.repeat,
                positive_multiplier=params.positive_multiplier,
                dev_frac=params.dev_frac,
                dev_reverse=params.dev_reverse,
                lr=params.lr,
                wd=params.wd,
                weight_kld=params.weight_kld,
                weight_entropy=params.weight_entropy,
                filename_train=params.filename_train,
                filename_test=params.filename_test,
                seed=params.seed,
                device=params.device
            )
            if best_f1 >= params.threshold_f1:
                break

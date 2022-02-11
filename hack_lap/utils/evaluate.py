import torch
import numpy as np


def _binary_clf_curve(y_true, y_score, pos_label=None):
    if pos_label is None:
        pos_label = 1.0
    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    threshold_idxs = np.arange(y_true.size)

    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def recall_both_curve(y_true, proba, eps=1e-8):
    fps0, tps0, thresholds0 = _binary_clf_curve(
        y_true, proba, pos_label=0
    )
    fps1, tps1, thresholds1 = _binary_clf_curve(
        y_true, proba, pos_label=1
    )
    assert np.all(thresholds0 == thresholds1)

    recall0 = 1 - tps0 / (tps0[-1] + eps)
    recall1 = tps1 / (tps1[-1] + eps)

    return recall0, recall1, thresholds1


def precision_recall(y_true, y_proba, cls_count_1=200, cls_count_2=5000):
    recall0, recall1, thresholds = recall_both_curve(y_true, y_proba)
    precision1 = recall1 * cls_count_1 / (recall1 * cls_count_1 + (1.0 - recall0) * cls_count_2)
    return recall1, precision1, thresholds


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
        yp.append(proba)
        yt.append(batch.y.to(torch.float32))
    bce_loss = torch.cat(bce_loss)
    bce_loss = torch.mean(bce_loss)

    yt = torch.cat(yt).detach().cpu().numpy()
    yp = torch.cat(yp).detach().cpu().numpy()

    recall, precision, thresholds = precision_recall(yt, yp)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    ii = np.argmax(f1)
    f1 = f1[ii]
    recall = recall[ii]
    precision = precision[ii]
    return bce_loss, (precision, recall, f1)

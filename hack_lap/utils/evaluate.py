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


def precision_recall(y_true, y_proba, cls_count_0=5000, cls_count_1=200):
    recall0, recall1, thresholds = recall_both_curve(y_true, y_proba)
    precision1 = recall1 * cls_count_1 / (recall1 * cls_count_1 + (1.0 - recall0) * cls_count_0)
    precision0 = recall0 * cls_count_0 / (recall0 * cls_count_0 + (1.0 - recall1) * cls_count_1)
    return (recall0, precision0), (recall1, precision1), thresholds


@torch.no_grad()
def evaluate(model, loader, device, repeat=1):
    assert isinstance(repeat, int) and repeat > 0
    fn_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    sigmoid = torch.nn.Sigmoid()
    bce_loss = []
    yt, yp = [], []
    for batch in loader:
        batch.to(device)
        ypb = []
        for _ in range(repeat):
            out = model(batch)
            log_proba = out[0].view(-1)
            proba = sigmoid(log_proba).view(-1, 1)
            ypb.append(proba)
        loss = fn_loss(log_proba, batch.y)
        bce_loss.append(loss)

        proba = torch.cat(ypb, dim=1)
        yp.append(proba)
        yt.append(batch.y.to(torch.float32))
    bce_loss = torch.cat(bce_loss)
    bce_loss = bce_loss.detach().cpu().numpy()

    yt = torch.cat(yt).detach().cpu().numpy()
    yp = torch.cat(yp).detach().cpu().numpy()
    return bce_loss, yt, yp


@torch.no_grad()
def evaluate_pred(model, loader, device, repeat=1):
    assert isinstance(repeat, int) and repeat > 0
    sigmoid = torch.nn.Sigmoid()
    yp = []
    for batch in loader:
        batch.to(device)
        ypb = []
        for _ in range(repeat):
            out = model(batch)
            log_proba = out[0].view(-1)
            proba = sigmoid(log_proba).view(-1, 1)
            ypb.append(proba)
        proba = torch.cat(ypb, dim=1)
        yp.append(proba)
    yp = torch.cat(yp).detach().cpu().numpy()
    return yp


def calculate_metrics(model, loader, device, repeat=1, eps=1e-6):
    bce_loss, yt, yp = evaluate(model, loader, device, repeat)
    bce_loss = np.mean(bce_loss)
    yp = np.mean(yp, axis=1).ravel()
    rp0, rp1, thresholds = precision_recall(yt, yp)
    f1 = 2 * rp1[0] * rp1[1] / (rp1[0] + rp1[1] + eps)
    ii = np.argmax(f1)
    f1 = f1[ii]
    recall_1 = rp1[0][ii]
    precision_1 = rp1[1][ii]
    recall_0 = rp0[0][ii]
    precision_0 = rp0[1][ii]
    thresholds = thresholds[ii]
    return bce_loss, (precision_0, recall_0), (precision_1, recall_1, f1), \
            thresholds


def estimate_prediction(yt, yp, cls_count_0, cls_count_1, dump_factor=1.0):
    yt, yp = yt.tolist(), yp.tolist()
    new_yt, new_yp = [], []
    for i in range(len(yt)):
        rp0, rp1, th = precision_recall(
            np.array(yt[:i] + yt[i + 1:]),
            np.array(yp[:i] + yp[i + 1:]),
            cls_count_0=cls_count_0,
            cls_count_1=cls_count_1
        )
        f1 = 2.0 * rp1[0] * rp1[1] / (rp1[0] + rp1[1] + 1e-5)
        ii = np.argmax(f1)
        new_yt.append(yt[i])
        new_yp.append(int(yp[i] > th[ii] * dump_factor))
    new_yt = np.array(new_yt)
    new_yp = np.array(new_yp)
    return new_yt, new_yp


def calculate_metrics_one_vs_rest_(yt, yp, eps=1e-6, cls_count_0=5000, cls_count_1=200, dump_factor=1.0):
    yp = np.mean(yp, axis=1).ravel()
    new_yt, new_yp = estimate_prediction(yt, yp, cls_count_0, cls_count_1, dump_factor)

    tp = (new_yp * new_yt).sum()
    tn = ((1 - new_yp) * (1 - new_yt)).sum()
    fp = (new_yp * (1 - new_yt)).sum()
    fn = ((1 - new_yp) * new_yt).sum()

    recall_1 = tp / (tp + fn)
    recall_0 = tn / (tn + fp)
    precision_1 = recall_1 * cls_count_1 / (recall_1 * cls_count_1 + (1.0 - recall_0) * cls_count_0)
    precision_0 = recall_0 * cls_count_0 / (recall_0 * cls_count_0 + (1.0 - recall_1) * cls_count_1)
    f1_1 = 2 * recall_1 * precision_1 / (recall_1 + precision_1 + eps)
    f1_0 = 2 * recall_0 * precision_0 / (recall_0 + precision_0 + eps)

    return (precision_0, recall_0, f1_0), (precision_1, recall_1, f1_1)


def calculate_metrics_one_vs_rest(model, loader, device, repeat=1, eps=1e-6,
                                  cls_count_0=5000, cls_count_1=200):
    bce_loss, yt, yp = evaluate(model, loader, device, repeat)
    bce_loss = np.mean(bce_loss)
    yp = np.mean(yp, axis=1).ravel()
    yt, yp = yt.tolist(), yp.tolist()
    new_yt, new_yp = [], []
    for i in range(len(yt)):
        rp0, rp1, th = precision_recall(
            np.array(yt[:i] + yt[i + 1:]),
            np.array(yp[:i] + yp[i + 1:]),
            cls_count_0=cls_count_0,
            cls_count_1=cls_count_1
        )
        f1 = 2.0 * rp1[0] * rp1[1] / (rp1[0] + rp1[1] + 1e-5)
        ii = np.argmax(f1)
        new_yt.append(yt[i])
        new_yp.append(int(yp[i] > th[ii]))
    new_yt = np.array(new_yt)
    new_yp = np.array(new_yp)

    tp = (new_yp * new_yt).sum()
    tn = ((1 - new_yp) * (1 - new_yt)).sum()
    fp = (new_yp * (1 - new_yt)).sum()
    fn = ((1 - new_yp) * new_yt).sum()

    recall_1 = tp / (tp + fn)
    recall_0 = tn / (tn + fp)
    precision_1 = recall_1 * cls_count_1 / (recall_1 * cls_count_1 + (1.0 - recall_0) * cls_count_0)
    precision_0 = recall_0 * cls_count_0 / (recall_0 * cls_count_0 + (1.0 - recall_1) * cls_count_1)
    f1_1 = 2 * recall_1 * precision_1 / (recall_1 + precision_1 + eps)
    f1_0 = 2 * recall_0 * precision_0 / (recall_0 + precision_0 + eps)

    return bce_loss, (precision_0, recall_0, f1_0), (precision_1, recall_1, f1_1)

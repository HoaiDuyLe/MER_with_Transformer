from copy import deepcopy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score

def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def weighted_acc(preds, truths, verbose):
    preds = preds.view(-1)
    truths = truths.view(-1)

    total = len(preds)
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(total):
        if truths[i] == 0:
            n += 1
            if preds[i] == 0:
                tn += 1
        elif truths[i] == 1:
            p += 1
            if preds[i] == 1:
                tp += 1

    w_acc = (tp * n / p + tn) / (2 * n)

    if verbose:
        fp = n - tn
        fn = p - tp
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N', n, 'Recall', recall, "f1", f1)

    return w_acc


def eval_iemocap(preds, truths, dataset_name, best_thresholds=None):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    '''
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    preds = torch.sigmoid(preds)

    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(np.average(aucs))

    if best_thresholds is None:
        # select the best threshold for each emotion category, based on F1 score
        thresholds = np.arange(0.05, 1, 0.05)
        _f1s = []
        for t in thresholds:
            _preds = deepcopy(preds)
            _preds[_preds > t] = 1
            _preds[_preds <= t] = 0

            this_f1s = []

            for i in range(num_emo):
                pred_i = _preds[:, i]
                truth_i = truths[:, i]
                this_f1s.append(f1_score(truth_i, pred_i))

            _f1s.append(this_f1s)
        _f1s = np.array(_f1s)
        best_thresholds = (np.argmax(_f1s, axis=0) + 1) * 0.05

    # th = [0.5] * truths.size(1)
    for i in range(num_emo):
        pred = preds[:, i]
        pred[pred > best_thresholds[i]] = 1
        pred[pred <= best_thresholds[i]] = 0
        preds[:, i] = pred

    accs = []
    recalls = []
    precisions = []
    f1s = []
    for i in range(num_emo):
        pred_i = preds[:, i]
        truth_i = truths[:, i]

        # mosei
        acc = weighted_acc(pred_i, truth_i, verbose=False) if dataset_name=="mosei" else accuracy_score(truth_i, pred_i)

        # iemocap
        # acc = accuracy_score(truth_i, pred_i)    

        f1 = f1_score(truth_i, pred_i)
        recall = recall_score(truth_i, pred_i)
        precision = precision_score(truth_i, pred_i)

        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    accs.append(np.average(accs))
    recalls.append(np.average(recalls))
    precisions.append(np.average(precisions))
    f1s.append(np.average(f1s))

    return (accs, recalls, precisions, f1s, aucs), best_thresholds

def eval_iemocap_ce(preds, truths):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    '''
    preds: (num_of_data, 4)
    truths: (num_of_data,)
    '''
    preds = preds.argmax(-1)
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='macro')
    r = recall_score(truths, preds, average='macro')
    p = precision_score(truths, preds, average='macro')
    return acc, r, p, f1

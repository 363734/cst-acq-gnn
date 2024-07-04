import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score


def print_metrics(metrics, keys, tk="\t- "):
    for key in keys:
        if key in metrics:
            print("{}{} score: [{}]".format(tk, key, metrics[key]))

def print_metrics_all(metrics):
    print_metrics(metrics, metrics.keys())


def compute_metrics_logits(logits, labels):
    _, indices = torch.max(logits, dim=1)  # gets the class predicted with majority
    return compute_metrics(indices, labels)


def compute_metrics(predicted, labels):
    pos = sum(labels)
    neg = len(labels) - pos
    val_accuracy = accuracy_score(labels, predicted, normalize=True)
    val_balanced_accuracy = balanced_accuracy_score(labels, predicted)
    val_recall = recall_score(labels, predicted, average='binary', pos_label=1)
    val_nb_tp = int(val_recall * pos)
    val_nb_fn = int(pos - val_nb_tp)
    val_tn = recall_score(labels, predicted, average='binary', pos_label=0)
    val_nb_tn = int(val_tn * neg)
    val_nb_fp = int(neg - val_nb_tn)
    val_precision = precision_score(labels, predicted, average='binary', pos_label=1)
    val_f1_score = f1_score(labels, predicted, average='binary', pos_label=1)
    return {"accuracy": val_accuracy, "balanced_accuracy": val_balanced_accuracy, "recall": val_recall, "tn": val_tn,
            "precision": val_precision, "f1": val_f1_score, 'nb_pos': pos, "nb_tp": val_nb_tp, "nb_fn": val_nb_fn,
            'nb_neg': neg, "nb_tn": val_nb_tn, "nb_fp": val_nb_fp}

def compute_f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def compute_precision(tp, fp):
    denom = tp + fp
    if denom == 0:
        return -1
    return tp / denom


def compute_recall(tp, fn):
    denom = tp + fn
    if denom == 0:
        return -1
    return tp / denom

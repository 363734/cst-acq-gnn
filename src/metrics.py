import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score


def print_metrics(metrics, keys, tk="   - "):
    for key in keys:
        if key in metrics:
            print("{}{} score: {}".format(tk, key, metrics[key]))

def print_metrics_all(metrics):
    print_metrics(metrics,metrics.keys())


def compute_metrics(logits, labels):
    # nb_item = len(labels)
    # nb_pos = sum(labels)
    # nb_neg = nb_item - nb_pos
    _, indices = torch.max(logits, dim=1)  # gets the class predicted with majority
    val_accuracy = accuracy_score(labels, indices, normalize=True)
    val_balanced_accuracy = balanced_accuracy_score(labels, indices)
    val_recall = recall_score(labels, indices, average='binary', pos_label=1)
    val_tn = recall_score(labels, indices, average='binary', pos_label=0)
    val_precision = precision_score(labels, indices, average='binary', pos_label=1)
    val_f1_score = f1_score(labels, indices, average='binary', pos_label=1)
    return {"accuracy": val_accuracy, "balanced_accuracy": val_balanced_accuracy, "recall": val_recall, "tn": val_tn,
            "precision": val_precision, "f1": val_f1_score}


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

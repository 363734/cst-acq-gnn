import random

import torch


# take input tensor 1D of ground truth, build masking split to
def create_mask(gt):
    # idx of pos and neg
    idx_pos = gt.nonzero().squeeze()
    idx_neg = (gt == 0).nonzero().squeeze()
    # nb of each class
    count_pos = sum(gt)
    count_neg = len(gt) - count_pos
    # shuffle the idx of neg
    perm = [i for i in range(count_neg)]
    random.shuffle(perm)
    idx_neg_perm = idx_neg[perm]
    # compute mask rotation for under-sampling
    nb_mask = count_neg // count_pos
    res = count_neg % count_pos
    siz = [count_pos + 1 if i < res else count_pos for i in range(nb_mask)]
    # split the negative
    split = torch.split(idx_neg_perm, siz)
    # add again the positives
    split = [torch.cat((idx_pos, s)) for s in split]
    return split


if __name__ == "__main__":

    gt = [1] * 10 + [0] * 92
    gt = torch.tensor(gt)

    sample = torch.tensor([-i for i in range(102)])

    print(gt)

    mask_split = create_mask(gt)
    print(mask_split)
    for mask in mask_split:
        print(sample[mask])

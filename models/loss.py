import torch
import numpy as np
from torch.nn.modules import loss

class WARPLoss(loss.Module):
    def __init__(self, num_labels=204):
        super(WARPLoss, self).__init__()
        self.rank_weights = [1.0 / 1]
        for i in range(1, num_labels):
            self.rank_weights.append(self.rank_weights[i - 1] + (1.0 / i + 1))

    def forward(self, input, target):
        """
        :rtype:
        :param input: Deep features tensor Variable of size batch x n_attrs.
        :param target: Ground truth tensor Variable of size batch x n_attrs.
        :return:
        """
        batch_size = target.size()[0]
        n_labels = target.size()[1]
        max_num_trials = n_labels - 1
        loss = 0.0

        for i in range(batch_size):

            for j in range(n_labels):
                if target[i, j] == 1:
                    neg_labels_idx = np.array([idx for idx, v in enumerate(target[i, :]) if v == 0])
                    neg_idx = np.random.choice(neg_labels_idx, replace=False)
                    sample_score_margin = 1 - input[i, j] + input[i, neg_idx]
                    num_trials = 0

                    while sample_score_margin < 0 and num_trials < max_num_trials:
                        neg_idx = np.random.choice(neg_labels_idx, replace=False)
                        num_trials += 1
                        sample_score_margin = 1 - input[i, j] + input[i, neg_idx]

                    r_j = np.floor(max_num_trials / num_trials)
                    weight = self.rank_weights[r_j]

                    for k in range(n_labels):
                        if target[i, k] == 0:
                            score_margin = 1 - input[i, j] + input[i, k]
                            loss += (weight * torch.clamp(score_margin, min=0.0))
        return loss

class FocalLoss(loss.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, input, target, eps=1e-10, gamma=2):
        probs = torch.clamp(input, eps, 1-eps)
        loss = - (torch.pow((1 - probs),gamma) * target * torch.log(probs) + torch.pow(probs, gamma) * (1 - target) * torch.log(1 - probs))
        loss = loss.sum(1)
        return loss.mean()

class MultiLabelSoftmaxRegressionLoss(loss.Module):
    def __init__(self):
        super(MultiLabelSoftmaxRegressionLoss, self).__init__()

    def forward(self, input, target, eps=1e-10):
        probs = torch.clamp(input, eps, 1-eps)
        target = target / torch.sum(target, dim = 1, keepdim = True)
        return -1 * torch.sum(torch.log(input) * target) / input.shape[0]
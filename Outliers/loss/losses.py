import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 0.1
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        null_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        null_loss = null_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * null_loss + self.smoothing * smooth_loss
        return loss.mean()

class SoftTargetCrossEntropy(nn.module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
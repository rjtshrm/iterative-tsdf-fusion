import torch
import torch.nn.functional as F


class RoutingLoss(torch.nn.Module):
    def __init__(self):
        super(RoutingLoss, self).__init__()

    def forward(self, depth_output, confidence_output, target):
        confidence_loss = torch.log(confidence_output + 1e-16)
        depth_loss = F.smooth_l1_loss(depth_output, target)
        net_loss = confidence_output * depth_loss - confidence_loss
        return torch.mean(net_loss)

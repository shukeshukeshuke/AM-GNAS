import torch
import torch.nn as nn

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # helper selecting normalization layer
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return


class MLP(nn.Module):
    def __init__(self, channel_sequence):
        super().__init__()
        nb_layers = len(channel_sequence) - 1
        self.seq = nn.Sequential()
        for i in range(nb_layers):
            self.seq.add_module(f"fc{i}", nn.Linear(channel_sequence[i], channel_sequence[i + 1]))
            if i != nb_layers - 1:
                self.seq.add_module(f"ReLU{i}", nn.ReLU(inplace=True))
        
    def forward(self, x):
        out = self.seq(x)
        return out



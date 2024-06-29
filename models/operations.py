import torch
import dgl.function as fn
import torch.nn as nn
import numpy as np
from models.networks import MLP

# from models.networks import *


OPS = {
    'V_None': lambda args: V_None(args),
    'V_I': lambda args: V_I(args),
    'V_Max': lambda args: V_Max(args),
    'V_Mean': lambda args: V_Mean(args),
    'V_Min': lambda args: V_Min(args),
    'V_Sum': lambda args: V_Sum(args),
    'V_Coarse': lambda args: V_Coarse(args),
    'V_Fine': lambda args: V_Fine(args),
    'V_HOP1': lambda args: V_HOP1(args),
    'V_HOP2': lambda args: V_HOP2(args),
    'V_HOP3': lambda args: V_HOP3(args),
}

First_Stage = ['V_None', 'V_I', 'V_Coarse', 'V_Fine']
Second_Stage = ['V_I', 'V_Mean', 'V_Sum', 'V_Max']
Third_Stage = ['V_None', 'V_I', 'V_Coarse', 'V_Fine']
K_Message = ['V_HOP1', 'V_HOP2', 'V_HOP3']


class V_Package(nn.Module):

    def __init__(self, args, operation):

        super().__init__()
        self.args = args
        self.operation = operation
        if type(operation) in [V_None, V_I]:
            self.seq = None
        else:
            self.seq = nn.Sequential()
            self.seq.add_module('fc_bn', nn.Linear(args.node_dim, args.node_dim, bias=True))
            if args.batchnorm_op:
                self.seq.add_module('bn', nn.BatchNorm1d(self.args.node_dim))
            self.seq.add_module('act', nn.ReLU())

    def forward(self, input):
        V = self.operation(input)
        if self.seq:
            V = self.seq(V)
        return V


class NodePooling(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.A = nn.Linear(args.node_dim, args.node_dim)
        # self.B        = nn.Linear(args.node_dim, args.node_dim)
        self.activate = nn.ReLU()
        # self.activate = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, V):
        V = self.A(V)
        V = self.activate(V)
        # V = self.B(V)
        return V


class V_None(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, input):
        V = input['V']
        return 0. * V


class V_I(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, input):
        V = input['V']
        return V


class V_Max(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)

    def forward(self, input):
        G, V = input['G'], input['V']
        # G.ndata['V'] = V
        G.ndata['V'] = self.pooling(V)
        G.update_all(fn.copy_u('V', 'M'), fn.max('M', 'V'))
        return G.ndata['V']


class V_Mean(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)

    def forward(self, input):
        G, V = input['G'], input['V']
        # G.ndata['V'] = V
        G.ndata['V'] = self.pooling(V)
        G.update_all(fn.copy_u('V', 'M'), fn.mean('M', 'V'))
        return G.ndata['V']


class V_Sum(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)

    def forward(self, input):
        G, V = input['G'], input['V']
        # G.ndata['V'] = self.pooling(V)
        G.ndata['V'] = V
        G.update_all(fn.copy_u('V', 'M'), fn.sum('M', 'V'))
        return G.ndata['V']


class V_Min(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)

    def forward(self, input):
        G, V = input['G'], input['V']
        G.ndata['V'] = self.pooling(V)
        G.update_all(fn.copy_u('V', 'M'), fn.min('M', 'V'))
        return G.ndata['V']


class V_Coarse(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.W = nn.Linear(args.node_dim * 2, args.node_dim, bias=True)

    def forward(self, input):
        V, V_in = input['V'], input['V_in']
        gates = torch.cat([V, V_in], dim=1)
        gates = self.W(gates)
        return torch.sigmoid(gates) * V


class V_Fine(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.W = nn.Linear(args.node_dim * 2, args.node_dim, bias=True)
        self.a = nn.Linear(args.node_dim, 1, bias=False)

    def forward(self, input):
        V, V_in = input['V'], input['V_in']
        gates = torch.cat([V, V_in], dim=1)
        # gates = self.W(gates)
        gates = torch.relu(self.W(gates))
        gates = self.a(gates)
        return torch.sigmoid(gates) * V


class V_HOP(nn.Module):

    def __init__(self, args, k=1):
        super().__init__()
        self.args = args
        self.fn_agg = args.fn_agg
        self.k = k
        self.pooling = NodePooling(args)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.W = nn.Linear(args.node_dim * 2, args.node_dim, bias=True)
        self.M = nn.Linear(args.edge_dim, args.node_dim, bias=True)
        if args.fn_agg == 'mean':
            self.fn_agg = fn.mean
        elif args.fn_agg == 'sum':
            self.fn_agg = fn.sum
        elif args.fn_agg == 'max':
            self.fn_agg = fn.max
        elif args.fn_agg == 'min':
            self.fn_agg = fn.min

    def messages(self, edges):
        M = self.message_fn(edges.src['V'], edges.dst['V'], edges.data['KHOP'])
        return {'M': M}

    def forward(self, input):
        G, V, V_in, E = input['G'], input['V'], input['V_in'], input['E']
        V = self.act(V)
        G.ndata['V'] = self.pooling(V)
        G.edata['E'] = self.M(E)
        G.edata['KHOP'] = G.edata['edge_attr'][:, self.k - 1].reshape(G.num_edges(), 1)
        G.update_all(
            lambda edge: {'X': (edge.src['V'] + edge.data['E']) * edge.data['KHOP']},
            self.fn_agg('X', 'V'))
        V = torch.cat([G.ndata['V'], V_in], dim=-1)
        G.ndata['V'] = torch.relu(self.W(V))
        return G.ndata['V']


class V_HOP1(V_HOP):

    def __init__(self, args):
        super().__init__(args, 1)


class V_HOP2(V_HOP):

    def __init__(self, args):
        super().__init__(args, 2)


class V_HOP3(V_HOP):

    def __init__(self, args):
        super().__init__(args, 3)

class V_HOP4(V_HOP):

    def __init__(self, args):
        super().__init__(args, 4)

class V_HOP5(V_HOP):

    def __init__(self, args):
        super().__init__(args, 5)


if __name__ == '__main__':
    print("test")

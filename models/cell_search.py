import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import OPS
from models.mixed import Mixed
from models.networks import MLP


'''
cell_arch : 
    topology: list
        (src, dst, weights, ops)
'''


class Cell(nn.Module):

    def __init__(self, args, cell_arch, last = False):
        super().__init__()
        self.args = args
        self.nb_nodes = args.nb_nodes * 2  # ! warning
        self.cell_arch = cell_arch
        self.trans_concat_V = nn.Linear(self.nb_nodes * args.node_dim, args.node_dim, bias = True)
        self.batchnorm_V = nn.BatchNorm1d(args.node_dim)
        self.batchnorm_E = nn.BatchNorm1d(args.edge_dim)
        self.activate = nn.LeakyReLU(args.leaky_slope)
        self.S = nn.Linear(args.node_dim * 2 + args.edge_dim, args.edge_dim, bias = True)
        self.load_arch()

    def load_arch(self):
        link_para = {}
        link_dict = {}
        for src, dst, w, ops in self.cell_arch:
            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append((src, w))
            link_para[str((src, dst))] = Mixed(self.args, ops)

        self.link_dict = link_dict
        self.link_para = nn.ModuleDict(link_para)

    def trans_edges(self, edges):
        E = self.S(torch.concat([edges.src['V'], edges.data['E'], edges.dst['V']], dim = -1))
        return {'E': E}

    def forward(self, input, weight):
        G, V_in, E_in = input['G'], input['V'], input['E']
        link_para = self.link_para
        link_dict = self.link_dict
        states = [V_in]
        for dst in range(1, self.nb_nodes + 1):
            tmp_states = []
            for src, w in link_dict[dst]:
                sub_input = {'G': G, 'V': states[src], 'V_in': V_in, 'E': E_in}
                tmp_states.append(link_para[str((src, dst))](sub_input, weight[w]))
            states.append(sum(tmp_states))


        V = self.trans_concat_V(torch.cat(states[1:], dim=1))
        G.edata['E'] = self.activate(E_in)
        G.ndata['V'] = V
        G.apply_edges(self.trans_edges)
        E = G.edata['E']

        V = self.batchnorm_V(V)
        E = self.batchnorm_E(E)
        V = self.activate(V)
        E = self.activate(E)
        V = F.dropout(V, self.args.dropout, training = self.training)
        E = F.dropout(E, self.args.dropout, training = self.training)
        V = V + V_in
        E = E + E_in
        return {'G': G, 'V': V, 'E': E}


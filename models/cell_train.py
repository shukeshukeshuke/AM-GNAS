import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import V_Package, OPS
from models.networks import MLP
from ogb.nodeproppred import DglNodePropPredDataset

class Cell(nn.Module):

    def __init__(self, args, genotype, last=False):

        super().__init__()
        self.args = args
        self.nb_nodes = args.nb_nodes
        self.genotype = genotype
        self.trans_concat_V = nn.Linear(self.nb_nodes * args.node_dim, args.node_dim, bias=True)
        self.batchnorm_V = nn.BatchNorm1d(args.node_dim)
        self.batchnorm_E = nn.BatchNorm1d(args.edge_dim)
        self.activate = nn.LeakyReLU(args.leaky_slope)
        self.S = nn.Linear(args.node_dim * 2 + args.edge_dim, args.edge_dim, bias=True)
        self.load_genotype()

    def load_genotype(self):
        geno = self.genotype
        link_dict = {}
        module_dict = {}
        for edge in geno['topology']:
            src, dst, ops = edge['src'], edge['dst'], edge['ops']
            dst = f'{dst}'

            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append(src)

            if dst not in module_dict:
                module_dict[dst] = nn.ModuleList([])
            module_dict[dst].append(V_Package(self.args, OPS[ops](self.args)))

        self.link_dict = link_dict
        self.module_dict = nn.ModuleDict(module_dict)

    def trans_edges(self, edges):
        E = self.S(torch.concat([edges.src['V'], edges.data['E'], edges.dst['V']], dim = -1))
        return {'E': E}

    def forward(self, input):

        G, V_in, E_in = input['G'], input['V'], input['E']
        states = [V_in]
        for dst in range(1, self.nb_nodes + 1):
            dst = f'{dst}'
            agg = []
            for i, src in enumerate(self.link_dict[dst]):
                sub_input = {'G': G, 'V': states[src], 'V_in': V_in, 'E': E_in}
                agg.append(self.module_dict[dst][i](sub_input))
            states.append(sum(agg))


        V = self.trans_concat_V(torch.cat(states[1:], dim=1))
        G.edata['E'] = self.activate(E_in)
        G.ndata['V'] = V
        G.apply_edges(self.trans_edges)
        E = G.edata['E']

        V = self.batchnorm_V(V)
        E = self.batchnorm_E(E)
        V = self.activate(V)
        E = self.activate(E)
        V = F.dropout(V, self.args.dropout, training=self.training)
        E = F.dropout(E, self.args.dropout, training=self.training)
        V = V + V_in
        E = E + E_in
        return {'G': G, 'V': V, 'E': E}


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict

    geno = yaml.load(open('example_geno.yaml', 'r'))
    geno = geno['Genotype'][0]
    args = edict({
        'nb_nodes': 4,
        'node_dim': 50,
        'leaky_slope': 0.2,
        'batchnorm_op': True,
    })
    cell = Cell(args, geno)

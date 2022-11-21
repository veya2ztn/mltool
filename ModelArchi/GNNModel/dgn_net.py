import torch.nn as nn
import torch
import dgl
from .dgn_layer import DGNLayer
from .mlp_readout_layer import MLPReadout
import numpy as np
DefaultNetParams={
                    "L": 8,
                    "hidden_dim": 45,
                    "out_dim": 45,
                    "type_net": "complex",
                    "residual": True,
                    "edge_feat": False,
                    "readout": "mean",
                    "in_feat_dropout": 0.0,
                    "dropout": 0.0,
                    "graph_norm": True,
                    "batch_norm": True,
                    "aggregators": "mean dir1-dx dir1-av",
                    "scalers": "identity amplification attenuation",
                    "towers": 5,
                    "divide_input_first": False,
                    "divide_input_last": True,
                    "edge_dim": 0,
                    "pretrans_layers" : 1,
                    "posttrans_layers" : 1,
                    "pos_enc_dim":1,
                    }
def print_dict(_dict):
    _str=""
    for k, v in _dict.items():
            v=str(v)
            _str+=k+' = '+v+'\n'
    print(_str)
class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_layer = None
        self.optimizer  = None
        self.usefocal_loss= False
        self.focal_lossQ  = False
        pass

    def _loss(self):
        raise NotImplementedError

    def _accu(self):
        raise NotImplementedError

    def forward(self,x,target=None):
        raise NotImplementedError

    def test(self,x,target=None):
        with torch.no_grad():
            return self(x,target)

    def fit(self,X_train,y_train):
        def closure():
            self.optimizer.zero_grad()
            loss,outputs = self(X_train,y_train)
            loss.backward()
            return loss
        loss=self.optimizer.step(closure)
        if torch.isnan(loss):
            raise NanValueError
        return loss

    def save_to(self,path):
        checkpoint=self.all_state_dict()
        torch.save(checkpoint,path)

    def load_from(self,path):
        checkpoint = torch.load(path)
        if ('state_dict' not in checkpoint):
            self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'use_focal_loss' in checkpoint:self.focal_lossQ=checkpoint['use_focal_loss']

    def reset(self):
        weights_init = kaiming_init
        self.apply(weights_init)

    def all_state_dict(self,epoch=None,mode="light"):
        checkpoint={}
        checkpoint['epoch']  =  epoch
        checkpoint['state_dict']    = self.state_dict()
        checkpoint['use_focal_loss']= self.focal_lossQ
        if mode != "light":
            checkpoint['optimizer']     = self.optimizer.state_dict()
        return checkpoint
class DGNNet(_Model):
    def __init__(self, image_type,curve_type,**kargs):
        super().__init__()
        ### parameter define
        self.model_class= "GNN"
        self.output_type  = curve_type
        self.output_dim   = np.prod(self.output_type.shape)  # so the output data is (...,256) and will be divide to (...,128,2) for complex num
        self.final_shape  = [-1]+list(self.output_type.data_shape)
        net_params = DefaultNetParams
        for key,val in kargs:net_params[key]=val
        net_params['num_atom_type'] = num_atom_type = 16
        net_params['num_bond_type'] = num_bond_type = 1
        D = torch.from_numpy(np.load("/media/tianning/DATA/metasurface/Compressed/randomly_data_valid_3000/B1NES128/test_imagedata_dgl_node_out_degree.npy"))
        net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))

        hidden_dim       = net_params['hidden_dim']
        out_dim          = net_params['out_dim']
        in_feat_dropout  = net_params['in_feat_dropout']
        dropout          = net_params['dropout']
        n_layers         = net_params['L']
        edge_dim         = net_params['edge_dim']
        pretrans_layers  = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']

        self.type_net    = net_params['type_net']
        self.pos_enc_dim = net_params['pos_enc_dim']
        self.readout     = net_params['readout']
        self.graph_norm  = net_params['graph_norm']
        self.batch_norm  = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers     = net_params['scalers']
        self.avg_d       = net_params['avg_d']
        self.residual    = net_params['residual']
        self.edge_feat   = net_params['edge_feat']
        self.device      = "cuda"

        self.net_params  = net_params

        print_dict(self.net_params)
        if self.pos_enc_dim > 0:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        ### model structure
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h     = nn.Embedding(num_atom_type, hidden_dim)
        if self.edge_feat:self.embedding_e = nn.Embedding(num_bond_type, edge_dim)

        self.layers = nn.ModuleList([DGNLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                                  batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                                  scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                                  edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model for _
                         in range(n_layers - 1)])
        self.layers.append(DGNLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat, edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model)

        if self.readout == "directional" or self.readout == "directional_abs":
            self.MLP_layer = MLPReadout(2 * out_dim, self.output_dim)
        else:
            self.MLP_layer = MLPReadout(out_dim, self.output_dim)  # 1 out dim since regression problem

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc_dim > 0:
            h_pos_enc = self.embedding_pos_enc(g.ndata['pos_enc'].to(self.device))
            h = h + h_pos_enc
        if self.edge_feat:
            e = self.embedding_e(e)

        for i, conv in enumerate(self.layers):
            h = conv(g, h, e, snorm_n)
        g.ndata['h'] = h

        if   self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == "directional_abs":
            g.ndata['dir'] = h * torch.abs(g.ndata['eig'][:, 1:2].to(self.device)) / torch.sum(
                torch.abs(g.ndata['eig'][:, 1:2].to(self.device)), dim=1, keepdim=True)
            hg = torch.cat([dgl.mean_nodes(g, 'dir'), dgl.mean_nodes(g, 'h')], dim=1)
        elif self.readout == "directional":
            g.ndata['dir'] = h * g.ndata['eig'][:, 1:2].to(self.device) / torch.sum(
                torch.abs(g.ndata['eig'][:, 1:2].to(self.device)), dim=1, keepdim=True)
            hg = torch.cat([torch.abs(dgl.mean_nodes(g, 'dir')), dgl.mean_nodes(g, 'h')], dim=1)
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        out = self.MLP_layer(hg)
        out = out.reshape(self.final_shape)
        return out
    def all_state_dict(self,epoch=None,mode="light"):
        checkpoint={}
        checkpoint['epoch']  =  epoch
        checkpoint['state_dict']    = self.state_dict()
        if mode != "light":
            checkpoint['optimizer']     = self.optimizer.state_dict()
        return checkpoint

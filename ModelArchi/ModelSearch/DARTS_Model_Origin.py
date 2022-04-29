import torch
import torch.nn as nn
from .operations_define import *
from torch.autograd import Variable


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = Variable(
            torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        )
        x=x.div(keep_prob) ## do not use inplace operation x.div_(keep_prob)
        x=x.mul(mask)
    return x

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev,padding_mode='zeros'):
        super(Cell, self).__init__()
        #print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0,padding_mode=padding_mode)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0,padding_mode=padding_mode)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction,padding_mode=padding_mode)

    def _compile(self, C, op_names, indices, concat, reduction,padding_mode='zeros'):
        assert len(op_names) == len(indices)
        def restructure(_list):
            return [[_list[2*i],_list[2*i+1]] for i in range(len(_list)//2)]
        op_names  = restructure(op_names)
        indices   = restructure(indices)

        self._steps     = len(op_names)
        self._concat    = concat

        self._ops       = nn.ModuleList()
        self.in_nodes   = []
        for names, indexes in zip(op_names, indices):
            in_node=[]
            for name,index in zip(names, indexes):
                stride = 2 if reduction and index < 2 else 1
                if name == "deleted" or name == "none":continue
                if index >= len(self.in_nodes)+2:continue # check previous node is generated.
                op = OPS[name](C, stride, True,padding_mode=padding_mode)
                self._ops += [op]
                in_node.append(index)
            if len(in_node)>0:self.in_nodes.append(in_node)
        self._indices   = indices
        self.multiplier = len(self.in_nodes)
    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        edge_id= 0
        for in_nodes in self.in_nodes:
            s=0
            for in_node in in_nodes:
                h  = states[in_node]
                op = self._ops[edge_id]
                h  = op(h)
                if self.training and drop_prob > 0.0 and (not isinstance(op, Identity)):h = drop_path(h, drop_prob)
                s+=h
                edge_id+=1
            if type(s) != int:states += [s]
        return torch.cat(states[-self.multiplier:], dim=1)

class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

#Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat init_channels num_classes layers nodes")
class Network(nn.Module):
    def __init__(self, C=None, num_classes=None,
                        nodes=None,layers=None, auxiliary=False, genotype=None,
                        padding_mode='zeros',virtual_bond_dim=5,
                        tnend=False,**kargs):
        super(Network, self).__init__()
        assert genotype is not None
        if isinstance(genotype,str):
            #read from file
            pass

        if C is None: C=genotype.init_channels
        else:assert (not hasattr(genotype,'init_channels')) or C==genotype.init_channels

        if num_classes is None: num_classes=genotype.classes
        else:
            pass
            #assert (not hasattr(genotype,'classes')) or num_classes==genotype.classes

        if nodes is None: nodes=genotype.nodes
        else:assert (not hasattr(genotype,'nodes')) or  nodes==genotype.nodes

        if layers is None: layers=genotype.layers
        else:assert (not hasattr(genotype,'layers')) or  layers==genotype.layers


        self._layers    = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0 ### NOTE: the origin train code will active use dropout
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False,padding_mode=padding_mode), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,padding_mode=padding_mode
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        if tnend:
            from .tnmodel.extend_model import Patch2NetworkInput
            from .tnmodel.two_dim_model import PEPS_uniform_shape_symmetry_any
            self.global_pooling = nn.Sequential(

                                      nn.Tanh(),
                                      #nn.BatchNorm2d(256),
                                      Patch2NetworkInput(1),
                                      PEPS_uniform_shape_symmetry_any(W=4,H=4,
                                          in_physics_bond=256,
                                          init_std=1e-5,
                                          out_features=256,init_method="Expecatation_Normalization",init_set_var=0.01,
                                          virtual_bond_dim=virtual_bond_dim,symmetry='P4Z2'),
                                  )
        else:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_save_states(self):
        return {
            "state_dict": self.state_dict(),
        }

    def load_states(self, save_states):
        self.load_state_dict(save_states["state_dict"])

    def forward(self, input, **kwargs):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

from .operations_define import MSOPS,MSOPS_NO,NORMALSIZE,REDUCESIZE,get_OPs_for
from .operations_layers import ReLUConvBN,FactorizedReduce

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def node_map(num):
    if num == 0: return "a"
    if num == 1: return "b"
    return num-2
class MixedOp(nn.Module):

    def __init__(self, C, stride, input_dim,operation_choose=None,operation_candidate=None,operation_weight   =None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)
        self.k = 2
        self._op_ids=[]
        self.op_names=operation_choose
        #print(operation_choose)
        self.op_candidate=list(operation_candidate.keys())
        for primitive in operation_choose:

            if C <  self.k:C_in = self.k
            else:C_in = C // self.k
            op = operation_candidate[primitive](C_in, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C_in, affine=False))
            if operation_weight is not None:
                # print(f"assign layer {primitive} in"),
                # print(operation_weight.keys())
                # print()
                op.load_state_dict(operation_weight[primitive])
            self._ops.append(op)
            self._op_ids.append(list(operation_candidate.keys()).index(primitive))

    def forward(self, x, weights_full=None):
        weights = weights_full[weights_full>0] if weights_full is not None else None
        assert  (weights is None) or (len(weights) == len(self._ops))

        # channel proportion k=4
        # channel shuffle is believed can accelerate the search speed (PC-DARTS) for large size
        # don't know its performance on small size
        dim_2 = x.shape[1]
        if dim_2 < self.k:
            temp1=0
            if weights is not None:
                for w, op in zip(weights, self._ops):temp1 += w.to(x.device) * op(x)
            else:
                for op in self._ops:temp1 += op(x)
            return temp1
        else:
            xtemp = x[:, :  dim_2 // self.k, :, :]
            xtemp2 = x[:,  dim_2 // self.k:, :, :]
            temp1=0
            if weights is not None:
                for w, op in zip(weights, self._ops):temp1 += w.to(x.device) * op(xtemp)
            else:
                for op in self._ops:temp1 += op(xtemp)

            # reduction cell needs pooling before concat
            if temp1.shape[2] == x.shape[2]:
                ans = torch.cat([temp1, xtemp2], dim=1)
            else:
                ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
            ans = channel_shuffle(ans, self.k)
            #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
            # except channe shuffle, channel shift also works
            return ans


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, input_dim,
                 operation_candidate,operation_config=None,operation_weight=None,withbasic=True):
        super(Cell, self).__init__()
        self.reduction   = reduction
        self.input_dim   = input_dim
        self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False) if reduction_prev else \
                           ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        if operation_weight is not None:
            self.preprocess0.load_state_dict(operation_weight['preprocess0'])
            self.preprocess1.load_state_dict(operation_weight['preprocess1'])
        self._steps      = steps
        self._nodes      = steps
        self._multiplier = multiplier
        self._ops        = nn.ModuleList()
        self._map        = []
        self.in_index    = []
        ## below is search mode:
        ### for each edge, we get #op options.
        self.betas_mask  = nn.Parameter(torch.FloatTensor(self._nodes, self._nodes+2).fill_(-np.inf),requires_grad=False)

        if operation_config is None:
            # initialize a default search config
            self.in_index        = [list(range(2 + node_order)) for node_order in range(self._nodes)]
            self.operation_config = [[None for i in in_index_of_this_node] for in_index_of_this_node in self.in_index]
            # if #node==4, default operation [[None,None],[None,None,None],[None,None,None,None],[None,None,None,None,None]]
        else:
            self.in_index        = operation_config[0]
            self.operation_config = operation_config[1]
        #print(self.reduction)

        for current_node,(in_nodes,edge_types) in enumerate(zip(self.in_index,self.operation_config)):
            # print(f"--layer{current_node}")
            # print(edge_types)
            sub_map = []
            for in_node,edge_type in zip(in_nodes,edge_types):
                stride = 2 if reduction and in_node < 2 else 1 #when do reduction, only the first two edge "a->x" or "b->x" do.
                if edge_type is None:
                    operation_choose = get_OPs_for(input_dim,stride,withbasic=withbasic)
                    operation_weight_for_this_choose = None
                else:
                    if isinstance(edge_type,list):operation_choose = edge_type
                    else:operation_choose = [edge_type]
                    if operation_weight is not None:
                        operation_weight_for_this_choose = operation_weight[current_node][in_node]
                    else:
                        operation_weight_for_this_choose = None
                #print(f'----node_{node_map(in_node)} to node_{current_node}')
                #print(f"{in_node}->{current_node} s:{stride} type:{edge_type}")
                # edge_type == None mean we do search job
                # when we search the arch, this operation is a mixed opteration with extrernal distribution input (shared memory stratagy)
                # we can also give the operation after we finish the search by only give one choose
                op = MixedOp(C, stride, input_dim,operation_choose   =operation_choose,
                                                  operation_weight   =operation_weight_for_this_choose,
                                                  operation_candidate=operation_candidate[stride])

                self._ops.append(op)
                self.betas_mask[current_node,in_node]=0
                sub_map.append(op._op_ids)
            self._map.append(sub_map)
        # not every operation will be available due to the image size,
        # for example Conv2d(k=16) is not available for (8,8) image
        ## if we use search mode, the alphas_mask is a tensor [#edge, #options]
        ## if we use fix model mode, the alphas_mask should be 0 or [#edge,1]
        operation_num_per_edge = REDUCESIZE if reduction else NORMALSIZE
        self.alphas_mask = nn.Parameter(torch.FloatTensor(len(self._ops), operation_num_per_edge).fill_(-np.inf),requires_grad=False)
        for edge_id,op in enumerate(self._ops):self.alphas_mask[edge_id,op._op_ids]=0
    def forward(self, s0, s1, weights1, weights2):
        s0 = self.preprocess0(s0)# node 0 AKA node 'a'
        s1 = self.preprocess1(s1)# node 1 AKA node 'b'
        states = [s0, s1]
        offset = 0
        for in_index_of_this_node in self.in_index:# default:[[0,1],[0,1,2],[0,1,2,3],[0,1,2,3,4]]
            s=0
            for j,in_index in enumerate(in_index_of_this_node):
                edge_id = offset+j # from the serialize weights to the right one
                h = states[in_index]
                if weights1 is not None:
                    s += weights2[edge_id].to(h.device)*self._ops[edge_id](h, weights1[edge_id].to(h.device))
                else:
                    s += self._ops[edge_id](states[in_index])
            offset += len(in_index_of_this_node)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, input_dim=16, shrink_layer_index=None,
                       steps=4, multiplier=4, stem_multiplier=2, circularQ = True,withbasic=True,
                        operation_config=None,operation_weight=None):
        super(Network, self).__init__()
        self._C           = C
        self._num_classes = num_classes
        self._layers      = layers
        self._steps       = steps
        self._multiplier  = multiplier
        self.input_dim    = input_dim
        self._shrink_layer_index = [layers // 3, 2 * layers // 3] if shrink_layer_index is None else shrink_layer_index

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),  # (B,C,16,16)
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        self._map      = []
        self.arch_searchQ = operation_config is None
        operation_candidate = MSOPS if circularQ else MSOPS_NO
        for i in range(layers):
            # for this layer we shrink the size and double the chennels
            if i in self._shrink_layer_index:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            operation_config_for_this_cell = None if operation_config is None else operation_config[i]
            cell_name=f"cell_{i}(reduce)" if reduction else f"cell_{i}(normal)"
            operation_weight_for_this_cell = None if operation_weight is None else operation_weight[cell_name]
            # cell_operation
            # normal layer: the dim for image will not change (B,C,X,X) -> (B,C,X,X)
            # reduce layer: the dim for image will be half    (B,C,X,X) -> (B,2C,X/2,X/2)
            #print(f"-{cell_name}")
            cell = Cell(steps, multiplier, C_prev_prev, C_prev,C_curr, reduction, reduction_prev, input_dim,
                        operation_candidate,operation_config=operation_config_for_this_cell,
                        operation_weight=operation_weight_for_this_cell,withbasic=withbasic)

            self._map.append(cell._map)
            if reduction:
                self.alpha_shape_reduce    = cell.alphas_mask.shape
                self.betas_shape_reduce    = cell.betas_mask.shape
            else:
                self.alpha_shape_normal    = cell.alphas_mask.shape
                self.betas_shape_normal    = cell.betas_mask.shape
            input_dim = input_dim // 2 if reduction else input_dim
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        if self.arch_searchQ:self._initialize_alphas()

    def forward(self, input):
        # here we share weight for every cell which mean the cell structure is fixed for a given architect
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if self.arch_searchQ:
                # if (not cell.alphas_mask.is_cuda) and (self.alphas_reduce.is_cuda):cell.alphas_mask = cell.alphas_mask.cuda()
                # if (not cell.betas_mask.is_cuda) and (self.betas_reduce.is_cuda):cell.betas_mask = cell.betas_mask.cuda()
                weights1 = F.softmax(self.alphas_reduce+ cell.alphas_mask, dim=-1) if cell.reduction else \
                           F.softmax(self.alphas_normal+ cell.alphas_mask, dim=-1)
                weights2 = F.softmax( self.betas_reduce+  cell.betas_mask, dim=-1) if cell.reduction else \
                           F.softmax( self.betas_normal+  cell.betas_mask, dim=-1)
                weights2 = weights2[weights2>0]
                s0, s1 = s1, cell(s0, s1, weights1, weights2)
            else:
                s0, s1 = s1, cell(s0, s1, None, None)
        out    = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


    def _initialize_alphas(self):
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(*self.alpha_shape_normal))
        self.betas_normal  = nn.Parameter(1e-3 * torch.randn(*self.betas_shape_normal))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(*self.alpha_shape_reduce))
        self.betas_reduce  = nn.Parameter(1e-3 * torch.randn(*self.betas_shape_reduce))

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self,use_Zero_layer=False):
        assert self.arch_searchQ
        gene =OrderedDict()
        gen_weight=OrderedDict()
        structure_config = []
        gen_weight['stem']= self.stem.state_dict()
        for i, cell in enumerate(self.cells):
            cell_name=f"cell_{i}(reduce)" if cell.reduction else f"cell_{i}(normal)"
            gene[cell_name]=OrderedDict()
            gen_weight[cell_name]=OrderedDict()
            gen_weight[cell_name]['preprocess0'] = cell.preprocess0.state_dict()
            gen_weight[cell_name]['preprocess1'] = cell.preprocess1.state_dict()
            weights1_full = F.softmax(self.alphas_reduce+ cell.alphas_mask, dim=-1) if cell.reduction else \
                            F.softmax(self.alphas_normal+ cell.alphas_mask, dim=-1)
            weights2_full = F.softmax( self.betas_reduce+ cell.betas_mask, dim=-1) if cell.reduction else \
                            F.softmax( self.betas_normal+  cell.betas_mask, dim=-1)
            weights2_full = weights2_full[weights2_full>0]
            start_id = 0
            the_in_nodes = []
            the_in_edges = []
            for current_node,in_nodes in enumerate(cell.in_index):
                gene[cell_name][current_node]=[]
                gen_weight[cell_name][current_node]={}
                weights1  = weights1_full[start_id:start_id+len(in_nodes)]#(x,36)
                weights2  = weights2_full[start_id:start_id+len(in_nodes)]#(x,)
                weights2  = weights2.unsqueeze(1) #[x,1]
                W         = weights1*weights2 #[x,36]
                all_ops        = [op for op in cell._ops[start_id:start_id+len(in_nodes)]]
                all_ops_cd     = [op.op_candidate for op in cell._ops[start_id:start_id+len(in_nodes)]]
                start_id += len(in_nodes)

                none_ops_idx = [_cd.op_candidate.index("none") for i,_cd in enumerate(all_ops)]
                if not use_Zero_layer:W[range(len(W)),none_ops_idx] = 0 #dont choose none operation

                max_val_per_edge,   max_idx_per_edge =  W.max(-1)
                max_val_idx            = max_val_per_edge.argsort()[-2:]
                in_node = []
                in_edge = []
                for edge_idx in max_val_idx:
                    candidate_ops   = all_ops[edge_idx]
                    candidate       = candidate_ops.op_candidate
                    op_idx_of_edge  = max_idx_per_edge[edge_idx]
                    ops_of_the_edge = candidate[op_idx_of_edge]
                    weight_the_ops  = candidate_ops._ops[op_idx_of_edge].state_dict()
                    in_node.append(edge_idx.item())
                    in_edge.append(ops_of_the_edge)
                    gene[cell_name][current_node].append((ops_of_the_edge, edge_idx))
                    gen_weight[cell_name][current_node][edge_idx.item()]={ops_of_the_edge:weight_the_ops}
                the_in_nodes.append(in_node)
                the_in_edges.append(in_edge)
            structure_config.append([the_in_nodes,the_in_edges])

        gene_str = ""
        for cell_name,cell_dict in gen_weight.items():
            gene_str+='\n'
            gene_str+=f'{cell_name}:'
            prefix   ="  -"
            for current_node,in_nodes_dict in cell_dict.items():
                if isinstance(in_nodes_dict,torch.Tensor):continue
                for in_node,ops_dict in in_nodes_dict.items():
                    if isinstance(ops_dict,torch.Tensor):continue
                    for op_name,op_weight in ops_dict.items():
                        gene_str+='\n'+prefix+f'node_{node_map(in_node)} to node_{current_node}:{op_name}'
        return structure_config,gene_str,gen_weight

from torch.nn.modules.utils import _pair,_single,_triple
from . import complex_layer   as semi_complex_layers
from . import complex_layer16 as full_complex_layers
import torch

def _force_triple(size,end=None):
    if len(size)==1 or len(size)==3:return _triple(size)
    else:
        assert size[0]==size[1]
        if end is None:
            return _triple(size[0])
        else:
            return tuple(list(size)+[end])

def _force_pair(size):
    if len(size)==1 or len(size)==2:return _pair(size)
    elif len(size) ==3:
        assert size[2]==1 or size[2]==2
        return size[:2]
    else:
        raise NotImplementedError

def _force_single(size):
    if len(size)==1:return _single(size)
    elif len(size) ==2:
        assert size[1]==1 or size[1]==2
        return size[:1]
    else:
        raise NotImplementedError

def SemiToFull_Conv2d(module):
    module_output = module
    if isinstance(module, semi_complex_layers.ComplexConv2d):
        module_output = full_complex_layers.ComplexConv2d(in_channels =module.in_channels,
                                                          out_channels=module.out_channels,
                                                          kernel_size =_force_pair(module.kernel_size),
                                                          stride      =_force_pair(module.stride     ),
                                                          padding     =_force_pair(module.padding    ),
                                                          dilation    =_force_pair(module.dilation   ),
                                                          bias        =False if module.bias is None else True)
    for name, child in module.named_children():
        module_output.add_module(name, SemiToFull_Conv2d(child))
    del module
    return module_output
def SemiToFull_Linear(module):
    module_output = module
    if isinstance(module, semi_complex_layers.ComplexLinear):
        module_output = full_complex_layers.ComplexLinear(in_features =module.in_features ,
                                                          out_features=module.out_features,
                                                          bias        =False if module.bias is None else True)
    for name, child in module.named_children():
        module_output.add_module(name, SemiToFull_Linear(child))
    del module
    return module_output
def SemiToFull_FunctionLayer(module):
    module_output = module
    if   isinstance(module, semi_complex_layers.ComplexBatchNorm2d):
        module_output = full_complex_layers.ComplexBatchNorm2d(module.num_features) #num_features is int
    elif isinstance(module, semi_complex_layers.ComplexBatchNorm1d):
        module_output = full_complex_layers.ComplexBatchNorm2d(module.num_features) #num_features is int
    elif isinstance(module, semi_complex_layers.ComplexAvgPool1d):
        module_output = full_complex_layers.ComplexAvgPool1d(_force_single(module.kernel_size))
    elif isinstance(module, semi_complex_layers.ComplexAvgPool2d):
        module_output = full_complex_layers.ComplexAvgPool2d(_force_pair(module.kernel_size))
    elif isinstance(module, semi_complex_layers.ComplexAdaptiveAvgPool1d):
        module_output = full_complex_layers.ComplexAdaptiveAvgPool1d(_force_single(module.output_size))
    elif isinstance(module, semi_complex_layers.ComplexAdaptiveAvgPool2d):
        module_output = full_complex_layers.ComplexAdaptiveAvgPool2d(_force_pair(module.output_size))
    for name, child in module.named_children():
        module_output.add_module(name, SemiToFull_FunctionLayer(child))
    del module
    return module_output
def ConvertAllActive_Tanh(module):
    module_output = module
    if isinstance(module, torch.nn.Sigmoid) or \
       isinstance(module, torch.nn.ReLU)    or \
       isinstance(module, torch.nn.LeakyReLU) or \
       isinstance(module, torch.nn.Tanh):
        module_output = full_complex_layers.ComplexTanh()
    for name, child in module.named_children():
        module_output.add_module(name, ConvertAllActive_Tanh(child))
    del module
    return module_output
def ConvertBNtoReImBN(module):
    module_output = module
    if   isinstance(module, semi_complex_layers.ComplexBatchNorm2d):
        module_output = semi_complex_layers.ComplexReImNorm2d(module.num_features)
    elif isinstance(module, full_complex_layers.ComplexBatchNorm2d):
        module_output = full_complex_layers.ComplexReImNorm2d(module.real_layer.num_features)
    for name, child in module.named_children():
        module_output.add_module(name, ConvertBNtoReImBN(child))
    del module
    return module_output
def SemiToReal(module):
    module_output = module
    if isinstance(module, semi_complex_layers.ComplexConv2d):
        assert (module.bias is None) or (module.bias == False)
        module_output = torch.nn.Conv3d(in_channels =module.in_channels ,
                                        out_channels=module.out_channels,
                                        kernel_size =_force_triple(module.kernel_size ,2),
                                        stride      =_force_triple(module.stride      ,1),
                                        padding     =_force_triple(module.padding     ,1),
                                        dilation    =_force_triple(module.dilation    ,2),
                                        bias        =False)
        # module_output = RealizedConv2d(in_channels =module.in_channels ,
        #                                 out_channels=module.out_channels,
        #                                 kernel_size =_force_triple(module.kernel_size ,2),
        #                                 stride      =_force_triple(module.stride      ,1),
        #                                 padding     =_force_triple(module.padding     ,1),
        #                                 dilation    =_force_triple(module.dilation    ,1),
        #                                 bias        =False)
    elif isinstance(module, semi_complex_layers.ComplexLinear):
        module_output = semi_complex_layers.GroupedLinear(in_features =module.in_features ,
                                       out_features=module.out_features,
                                       bias        =False if module.bias is None else True        ,)
        # module_output = RealizedLinear(in_features =module.in_features*2 ,
        #                                out_features=module.out_features*2,
        #                                bias        =False if module.bias is None else True        ,)
    for name, child in module.named_children():
        module_output.add_module(name, SemiToReal(child))
    del module
    return module_output
def SemiToReal_Conv2d_first_layer(module):
    module_output = module
    if isinstance(module, semi_complex_layers.ComplexConv2d):
        assert (module.bias is None) or (module.bias == False)
        module_output = semi_complex_layers.RealizedConv2d(in_channels =module.in_channels ,
                                        out_channels=module.out_channels,
                                        kernel_size =_force_triple(module.kernel_size ,2),
                                        stride      =_force_triple(module.stride      ,1),
                                        padding     =_force_triple(module.padding     ,1),
                                        dilation    =_force_triple(module.dilation    ,1),
                                        bias        =False)
    for name, child in module.named_children():
        module_output.add_module(name, SemiToReal_Conv2d_first_layer(child))
    del module
    return module_output

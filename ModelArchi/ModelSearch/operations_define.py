from .real_operation_module import *
from .complex_operation_module import *
from ..SymmetryCNN import P4_Conv2d,P4Z2_Conv2d,V2_Conv2d,H2_Conv2d,Z2_Conv2d
from ..TensorNetworkLayer.Conv2dMPS import embedConv2d
def config_list(size,stride):
    # generate all conv type for
    record=[]
    s = stride
    for p in range(1,size//2):
        for d in range(1,2*p+s-1+size):
            if (2*p+s-1)%d !=0:continue
            k = (2*p+s-1)//d+1
            record.append([p//1,d//1,k])
    return  record

def get_OPs_for(input_dim,stride,withbasic=True,circularQ=True):
    if input_dim is None:input_dim  = MAXSIZE
    s      = stride
    cl     = config_list(input_dim,stride)
    if withbasic:
        names  = [n for n in BasicOPS.keys()]
    else:
        names  = ['none', 'avg_pool_3x3','max_pool_3x3','skip_connect',]
    for p,d,k in config_list(input_dim,stride):names.append(MSNAMERULE(k,d,s,p,circularQ))
    return names

### all permitted operation list here

OPS = {
    "none"        : lambda C, stride, affine,padding_mode='zeros': Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine,padding_mode='zeros': nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    "max_pool_3x3": lambda C, stride, affine,padding_mode='zeros': nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride, affine,padding_mode='zeros': Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine,padding_mode=padding_mode),
    "sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, affine=affine,padding_mode=padding_mode),
    "sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, affine=affine,padding_mode=padding_mode),
    "sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, affine=affine,padding_mode=padding_mode),
    "dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, 2, affine=affine,padding_mode=padding_mode),
    "dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, 2, affine=affine,padding_mode=padding_mode),
    "conv_7x1_1x7": lambda C, stride, affine,padding_mode='zeros': nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine,padding_mode=padding_mode),
    ),
    "[symmetry_keep]avg_pool_3x3": lambda C, stride, affine,padding_mode='zeros': nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False) if stride == 1 else nn.AvgPool2d(2, stride=2, padding=0),
    "[symmetry_keep]max_pool_3x3": lambda C, stride, affine,padding_mode='zeros': nn.MaxPool2d(3, stride=1, padding=1) if stride == 1 else nn.MaxPool2d(2, stride=2, padding=0),
    "[symmetry_keep]skip_connect": lambda C, stride, affine,padding_mode='zeros': Identity() if stride == 1 else nn.AvgPool2d(2, stride=2, padding=0),

    "[symP4]sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symP4]sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symP4]sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symP4]dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symP4]dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode),

    "[symZ2]sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symZ2]sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symZ2]sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symZ2]dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symZ2]dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode),

    "[symP4Z2]sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symP4Z2]sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symP4Z2]sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symP4Z2]dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode),
    "[symP4Z2]dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode),

    "[auto][symP4]sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symP4]sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symP4]sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symP4]dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symP4]dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, CNNModule=P4_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),

    "[auto][symZ2]sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symZ2]sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symZ2]sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symZ2]dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symZ2]dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, CNNModule=Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),

    "[auto][symP4Z2]sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symP4Z2]sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symP4Z2]sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symP4Z2]dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),
    "[auto][symP4Z2]dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, CNNModule=P4Z2_Conv2d, affine=affine,padding_mode=padding_mode,active_symmetry_fix=True),


    "[eMPSCNN]sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, CNNModule=embedConv2d, affine=affine,padding_mode=padding_mode),
    "[eMPSCNN]sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, CNNModule=embedConv2d, affine=affine,padding_mode=padding_mode),
    "[eMPSCNN]sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, CNNModule=embedConv2d, affine=affine,padding_mode=padding_mode),
    "[eMPSCNN]dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, CNNModule=embedConv2d, affine=affine,padding_mode=padding_mode),
    "[eMPSCNN]dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, CNNModule=embedConv2d, affine=affine,padding_mode=padding_mode),

    "[cplx]sep_conv_3x3": lambda C, stride, affine,padding_mode='zeros': CplxSepConv(C, C, 3, stride, 1, affine=affine,padding_mode=padding_mode),
    "[cplx]sep_conv_5x5": lambda C, stride, affine,padding_mode='zeros': CplxSepConv(C, C, 5, stride, 2, affine=affine,padding_mode=padding_mode),
    "[cplx]sep_conv_7x7": lambda C, stride, affine,padding_mode='zeros': CplxSepConv(C, C, 7, stride, 3, affine=affine,padding_mode=padding_mode),
    "[cplx]dil_conv_3x3": lambda C, stride, affine,padding_mode='zeros': CplxDilConv(C, C, 3, stride, 2, 2, affine=affine,padding_mode=padding_mode),
    "[cplx]dil_conv_5x5": lambda C, stride, affine,padding_mode='zeros': CplxDilConv(C, C, 5, stride, 4, 2, affine=affine,padding_mode=padding_mode),

}



BasicOPS =  {
  'none'         : lambda C, stride, affine,padding_mode='zeros': Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine,padding_mode='zeros': nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine,padding_mode='zeros': nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine,padding_mode='zeros': Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine,padding_mode=padding_mode),
  'sep_conv_3x3' : lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 3, stride, 1, affine=affine,padding_mode=padding_mode),
  'sep_conv_5x5' : lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 5, stride, 2, affine=affine,padding_mode=padding_mode),
  'sep_conv_7x7' : lambda C, stride, affine,padding_mode='zeros': SepConv(C, C, 7, stride, 3, affine=affine,padding_mode=padding_mode),
  'dil_conv_3x3' : lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 3, stride, 2, 2, affine=affine,padding_mode=padding_mode),
  'dil_conv_5x5' : lambda C, stride, affine,padding_mode='zeros': DilConv(C, C, 5, stride, 4, 2, affine=affine,padding_mode=padding_mode),
}

MAXSIZE = 16
MSNAMERULE=lambda k,d,s,p,c:f"circular_conv_k{k}_s{s}_d{d}_p{p}" if c else f"simple_conv_k{k}_s{s}_d{d}_p{p}"
MSOPS={}
for s in [1,2]:
    MSOPS[s]=BasicOPS.copy()
    for p,d,k in config_list(16,s):
        MSOPS[s][MSNAMERULE(k,d,s,p,True)]=ReLUcclConvBNWrapper(k,s,d,True)

MSOPS_NO={}
for s in [1,2]:
    MSOPS_NO[s]=BasicOPS.copy()
    for p,d,k in config_list(16,s):
        MSOPS_NO[s][MSNAMERULE(k,d,s,p,False)]=ReLUcclConvBNWrapper(k,s,d,False)

NORMALSIZE = len(MSOPS[1])
REDUCESIZE = max(len(MSOPS[1]),len(MSOPS[2]))

BasicOPSNAME=list(BasicOPS.keys())

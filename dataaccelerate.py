#pip install prefetch_generator

# 新建DataLoaderX类
from torch.utils.data import DataLoader

import torch

def sendall2gpu(listinlist,device):
    out=[]
    if isinstance(listinlist,list):
        for _list in listinlist:
            out.append(sendall2gpu(_list,device))
        return out
    else:
        return listinlist.to(device=device, non_blocking=True)
try:
    from prefetch_generator import BackgroundGenerator
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
except:
    DataLoaderX = DataLoader
class DataSimfetcher():
    def __init__(self, loader, device='auto'):
        if device == 'auto':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.loader = iter(loader)

    def next(self):
        try:
            self.batch = next(self.loader)
            self.batch = sendall2gpu(self.batch,self.device)
        except StopIteration:
            self.batch = None
        return self.batch
class DataPrefetcher():
    def __init__(self, loader, device='auto'):
        if device == 'auto':self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:raise NotImplementedError
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = sendall2gpu(self.batch,self.device)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

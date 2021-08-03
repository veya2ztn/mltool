
class Genotype:
    def __init__(self,normal,normal_concat,reduce,reduce_concat,init_channels=16,classes=2,nodes=4,layers=8,**kargs):
        self.normal        = normal
        self.normal_concat = normal_concat
        self.reduce        = reduce
        self.reduce_concat = reduce_concat
        self.init_channels = init_channels
        self.classes       = classes
        self.nodes         = nodes
        self.layers        = layers
        self.class_name    = "Genotype"
    def __repr__(self):
        string=','.join(f"{k}={v}" for k, v in vars(self).items())
        string=f"{self.class_name}({string})"
        return string

from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.fft
from .utils import conv_engines,transposeconv_engines
from .blocks import Block

class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=13, embed_dim=768,conv_simple=True):
        super().__init__()

        if img_size is None:raise KeyError('img is None')
        patch_size   = [patch_size]*len(img_size) if isinstance(patch_size,int) else patch_size

        num_patches=1
        out_size=[]
        for i_size,p_size in zip(img_size,patch_size):
            if not i_size%p_size:
                num_patches*=i_size// p_size
                out_size.append(i_size// p_size)
            else:
                raise NotImplementedError(f"the patch size ({patch_size}) cannot divide the img size {img_size}")
        self.img_size    = tuple(img_size)
        self.patch_size  = tuple(patch_size)
        self.num_patches = num_patches
        self.out_size    = tuple(out_size)
        #conv_engine = [nn.Conv1d,nn.Conv2d,nn.Conv3d]
        conv_engine = conv_engines(len(img_size),conv_simple=conv_simple)
        self.proj   = conv_engine(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, = x.shape[:2]
        inp_size = x.shape[2:]
        assert tuple(inp_size) == self.img_size, f"Input image size ({inp_size}) doesn't match model set size ({self.img_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class AFNONet(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=20, out_chans=20, embed_dim=768, depth=12, mlp_ratio=4.,
                 uniform_drop=False, drop_rate=0., drop_path_rate=0., unique_up_sample_channel=0,
                 dropcls=0, checkpoint_activations=False, fno_blocks=3,double_skip=False,
                 fno_bias=False, fno_softshrink=False,debug_mode=False,history_length=1,
                 conv_simple=True,patch_embedding_layer=PatchEmbed, build_head=True ,**kargs):
        super().__init__()

        assert img_size is not None
        patch_size   = [patch_size]*len(img_size) if isinstance(patch_size,int) else patch_size
        self.in_chans = in_chans
        self.out_chans=out_chans
        if history_length > 1:
            img_size = (history_length,*img_size)
            patch_size = (1,*patch_size)
        # print("============model:AFNONet================")
        # print(f"img_size:{img_size}")
        # print(f"patch_size:{patch_size}")
        # print(f"in_chans:{in_chans}")
        # print(f"out_chans:{out_chans}")
        # print("========================================")
        self.history_length = history_length
        self.checkpoint_activations=checkpoint_activations
        self.embed_dim   = embed_dim
        norm_layer       = partial(nn.LayerNorm, eps=1e-6)
        self.img_size    = img_size
        self.patch_embed = patch_embedding_layer(img_size=img_size, patch_size=patch_size, 
                    in_chans=in_chans, embed_dim=embed_dim,conv_simple=conv_simple)
        num_patches      = self.patch_embed.num_patches
        patch_size       = self.patch_embed.patch_size
        self.patch_size  = patch_size
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop    = nn.Dropout(p=drop_rate)
        self.unique_up_sample_channel =unique_up_sample_channel= out_chans if unique_up_sample_channel == 0 else unique_up_sample_channel
        self.final_shape = self.patch_embed.out_size
        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                                           drop_path=dpr[i],
                                           norm_layer=norm_layer,
                                           region_shape=self.final_shape,
                                           double_skip=double_skip,
                                           fno_blocks=fno_blocks,
                                           fno_bias=fno_bias,
                                           fno_softshrink=fno_softshrink) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        # self.num_features = out_chans * img_size[0] * img_size[1]
        # self.representation_size = self.num_features * 8
        # self.pre_logits = nn.Sequential(OrderedDict([
        #     ('fc', nn.Linear(embed_dim, self.representation_size)),
        #     ('act', nn.Tanh())
        # ]))
        self.transposeconv_engine =transposeconv_engine= transposeconv_engines(len(img_size),conv_simple=conv_simple)

        if build_head:
            conf_list = self.build_upsampler(patch_size, embed_dim, unique_up_sample_channel)

            # Generator head
            # self.head = nn.Linear(self.representation_size, self.num_features)
            self.head = transposeconv_engine(unique_up_sample_channel*4, out_chans,  **conf_list[2])
        else:
            self.head = self.pre_logits = None
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        #torch.nn.init.normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.debug_mode=debug_mode
        if self.history_length >1:
            self.last_Linear_layer = nn.Linear(out_chans*self.history_length,out_chans)

    def build_upsampler(self, patch_size, embed_dim, unique_up_sample_channel):
        transposeconv_engine = self.transposeconv_engine
        conf_list = [{'kernel_size':[],'stride':[],'padding':[]},
                     {'kernel_size':[],'stride':[],'padding':[]},
                     {'kernel_size':[],'stride':[],'padding':[]}]
        conv_set = {8:[[2,2,0],[2,2,0],[2,2,0]],
                    4:[[2,2,0],[3,1,1],[2,2,0]],
                    2:[[3,1,1],[3,1,1],[2,2,0]],
                    1:[[3,1,1],[3,1,1],[3,1,1]],
                    3:[[3,1,1],[3,1,1],[3,3,0]],
                   }
        for patch in patch_size:
            for slot in range(len(conf_list)):
                conf_list[slot]['kernel_size'].append(conv_set[patch][slot][0])
                conf_list[slot]['stride'].append(conv_set[patch][slot][1])
                conf_list[slot]['padding'].append(conv_set[patch][slot][2])
        self.conf_list = conf_list
        #transposeconv_engine = [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d][len(img_size)-1]
        
        self.pre_logits = nn.Sequential(
        OrderedDict([
            ('conv1', transposeconv_engine(embed_dim, unique_up_sample_channel*16, **conf_list[0])),
            ('act1', nn.Tanh()),
            ('conv2', transposeconv_engine(unique_up_sample_channel*16, unique_up_sample_channel*4, **conf_list[1])),
            ('act2', nn.Tanh())
        ]))

        return conf_list


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.pos_drop(x)
        return x

    def kernel_features(self, x):
        for blk in self.blocks:x = blk(x)
        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, *self.final_shape])
        return x
    
    def forward_head(self, x):
        #timer.record('forward_features',level=0)
        x = self.final_dropout(x)
        #timer.record('final_dropout',level=0)
        x = self.pre_logits(x);#print(torch.std_mean(x))
        #timer.record('pre_logits',level=0)
        x = self.head(x)  # print(torch.std_mean(x))
        return x
    
    def forward(self, x, return_feature=False):
        ### we assume always feed the tensor (B, p*z, h, w)
        #print(x.shape)
        B = x.shape[0]
        ot_shape = x.shape[2:]
        x = x.reshape(B,-1,*self.img_size)# (B, p, z, h, w) or (B, p, h, w)
        #timer.restart(level=0)
        #print(torch.std_mean(x))
        x   = self.forward_features(x);#print(torch.std_mean(x))
        fea = self.kernel_features(x)
        x   = self.forward_head(fea)
        if self.history_length >1:
            x = x.flatten(1,2).transpose(1,-1)
            x = self.last_Linear_layer(x)
            x = x.transpose(1,-1)
            ot_shape=ot_shape[1:]
        #timer.record('head',level=0)
        x = x.reshape(B,-1,*ot_shape)
        #timer.show_stat()
        #print("============================")
        if return_feature:return x,fea
        return x
    

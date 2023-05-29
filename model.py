import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformer import BertConfig, BertModel


class ModelConfig(object):
    def __init__(self,configdict):
        super().__init__()
        self.dims = configdict['dims']
        self.view_num = configdict['view_num']
        self.hidden_size = configdict['hidden_size']
        self.projectors = [[self.dims[i]]+[1024,1024,1024]+[self.hidden_size] for i in range(self.view_num)]
        self.generators = [[self.hidden_size]+[1024,1024,1024]+[self.dims[i]] for i in range(self.view_num)]
        self.bertconfig = BertConfig(hidden_size=self.hidden_size,
                                     num_hidden_layers = 3,
                                     num_attention_heads = 4,
                                     intermediate_size = 4*self.hidden_size,
                                     hidden_dropout_prob=0.1)
        self.class_num = configdict['class_num']


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config         = config
        self.cls = nn.Parameter(torch.zeros(1,1,self.config.hidden_size))

        self.projectors     = nn.ModuleList([])
        self.generators     = nn.ModuleList([])
        self.fusion         = BertModel(self.config.bertconfig)
        self.classifier     = nn.Linear(self.config.hidden_size, self.config.class_num)


        for i in range(self.config.view_num):
            self.projectors.append(MLP(self.config.projectors[i],batchnorm=True))
            self.generators.append(MLP(self.config.generators[i],batchnorm=True))

    def forward(self, x, s):
        bsz = s.shape[0]
        device = next(self.parameters()).device
        x_init = x
        #1.project
        x = [self.projectors[i](x[i]).unsqueeze(1) for i in range(self.config.view_num)]
        x = torch.cat(x, dim=1)     #[N,V,d]
        
        #2.pool information
        c = [x[i][s[i].to(dtype=torch.bool)].mean(dim=0,keepdim=True) for i in range(bsz)]
        c = torch.cat(c, dim=0)     #[N,  d]
        
        #3.reconstruct
        x_r = [self.generators[i](c) for i in range(self.config.view_num)]
        
        #4.impute
        x = [torch.zeros_like(x_r[i]) for i in range(self.config.view_num)]
        for i in range(self.config.view_num):
            x[i][s[:,i].to(dtype=torch.bool)]  = x_init[i][s[:,i].to(dtype=torch.bool)]
            x[i][(1-s[:,i]).to(dtype=torch.bool)] = x_r[i][(1-s[:,i]).to(dtype=torch.bool)]
        xm = [F.normalize(e).unsqueeze(1) for e in x]
        
        #5.extract information
        x = [self.projectors[i](x[i]).unsqueeze(1) for i in range(self.config.view_num)]
        
        #6.fuse
        c = self.cls.expand(bsz, 1, self.config.hidden_size)
        x = torch.cat([c]+x, dim = 1)
        features = F.normalize(x[:,1:],dim=2)

        ones = torch.ones(bsz,1).to(dtype=torch.long,device=device)
        attention_mask = torch.cat([ones,s],dim=1)

        token_type_ids = torch.arange(self.config.view_num+1).to(dtype=torch.long, device=device)
        bert_output = self.fusion(x,token_type_ids = token_type_ids,attention_mask=attention_mask)
        x = bert_output[1]
        all_attentions = bert_output[2]
        #features = F.normalize(x).unsqueeze(1)


        #7.classify
        x = self.classifier(x)

        #8.adversarial
        xr = [ReverseLayerF.apply(x_r[i], 1.0) for i in range(self.config.view_num)]

        return x, xr, xm, features


class MLP(nn.Module):
    activation_dict = {'sigmoid':nn.Sigmoid(),
                       'relu'   :torch.relu,
                       'tanh'   :nn.Tanh(),
                       'leakyrelu':nn.LeakyReLU(0.2, inplace=True)}

    def __init__(self,dims,activation='leakyrelu',batchnorm=False):
        super().__init__()
        self.dims = dims
        self.layer_num = len(dims)-1
        self.layers    = []
        self.activation = self.activation_dict[activation]
        for i in range(self.layer_num):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(self.dims[i + 1]))
            if i+1 < self.layer_num:
                self.layers.append(self.activation)
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.layers(x)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

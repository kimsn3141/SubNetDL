import torch
import torch.nn as nn
from torch.nn import init


class GATlayer(nn.Module):

    def __init__(self, input_size, output_size,act_func = 'relu',leaky_relu_alpha=0.2):
        super(GATlayer, self).__init__()
        self.act_func = act_func

        self.layer_weight = nn.Parameter(init.xavier_normal_(torch.empty(input_size,output_size,dtype=torch.float32)))
        self.layer_weight.requires_grad = True
        self.layer_bias = nn.Parameter(torch.randn(output_size,dtype=torch.float32))
        self.layer_bias.requires_grad = True

        self.attention_weight_a = nn.Parameter(torch.randn(output_size,dtype=torch.float32))
        self.attention_weight_a.requires_grad = True
        self.attention_weight_b = nn.Parameter(torch.randn(output_size,dtype=torch.float32))
        self.attention_weight_b.requires_grad = True
        self.attention_bias = nn.Parameter(torch.randn(1,dtype=torch.float32))
        self.attention_bias.requires_grad = True

        self.batch_norm = nn.BatchNorm1d(output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_alpha)
        self.soft_max = nn.Softmax(dim=-1)
        if(self.act_func == 'relu'):
            self.act = nn.ReLU()
        elif(self.act_func == 'sigmoid'):
            self.act = nn.Sigmoid()
        elif(self.act_func == 'prelu'):
            self.act = nn.PReLU()

        self.dropout = nn.Dropout(0.2)

    def forward(self,x,mask):
        x = torch.matmul(x,self.layer_weight)+self.layer_bias
        a = torch.matmul(x,self.attention_weight_a)
        b = torch.matmul(x,self.attention_weight_b)
        e = []
        for i,j in zip(a,b):
            e.append((i+torch.reshape(j,(-1,1))+self.attention_bias)*mask)
        e = torch.stack(e)
        e = self.leaky_relu(e)
        e = self.soft_max(e)
        e = self.dropout(e)
        x = torch.matmul(e,x)
        if(self.act_func == 'non'):
            return x
        x = self.act(x)
        return x


class Graph_classification(nn.Module):
    def __init__(self,number_of_genes,k):
        super(Graph_classification, self).__init__()
        self.gat_layer = GATlayer(k,1,act_func = 'prelu')
        self.linear = nn.Linear(number_of_genes,1)
        self.softnax = nn.Sigmoid()

    def forward(self, x, mask):
        x = self.gat_layer(x, mask)
        x = torch.min(x,0).values
        x = x.squeeze()
        x = self.linear(x)
        x = self.softnax(x)
        return x
    

class SubNetDL(nn.Module):
    def __init__(self,graph_mask,network_form, number_of_genes,act_func = 'relu'):
        super(SubNetDL, self).__init__()
        self.mask = nn.Parameter(graph_mask)
        self.mask.requires_grad = False
        self.form = network_form
        if(act_func == 'relu'):
            self.act = nn.ReLU()
        elif(act_func == 'sigmoid'):
            self.act = nn.Sigmoid()
        elif(act_func == 'prelu'):
            self.act = nn.PReLU()
        n = 0
        for (in_size,out_size,k) in network_form:
            if n < len(self.form)-1:
                for i in range(k):
                    self.add_module("l_"+str(n)+'_'+str(i), GATlayer(in_size,out_size,act_func = act_func))
            else:
                for i in range(k):
                    self.add_module("l_"+str(n)+'_'+str(i), GATlayer(in_size,out_size,act_func='non'))
            n = n+1
        self.graph_classification = Graph_classification(number_of_genes,out_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        
        n = 0
        for (_,_,k) in self.form:
            x = self.dropout(x)
            if n < len(self.form)-1:
                for i in range(k):
                    if(i == 0):
                        y = self._modules["l_"+str(n)+'_'+str(i)](x,self.mask)
                    else:
                        y = torch.cat((y,self._modules["l_"+str(n)+'_'+str(i)](x,self.mask)),dim = -1)
            else:
                for i in range(k):
                    if(i == 0):
                        y = self._modules["l_"+str(n)+'_'+str(i)](x,self.mask)
                    else:
                        y = y+self._modules["l_"+str(n)+'_'+str(i)](x,self.mask)
                y = y/k
                y = self.act(y)
            x = y
            n = n+1
        
        x = self.graph_classification(x, self.mask)
        return x

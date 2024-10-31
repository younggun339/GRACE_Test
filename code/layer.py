import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha ##LeakyReLU 활성화 함수의 negative slope로, 신경망이 음의 입력값을 처리하는 방식에 영향
        self.concat = concat ##True인 경우 출력을 ELU(Exponential Linear Unit)로 활성화

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))) ##입력 피처(in_features)와 출력 피처(out_features) 간의 가중치 행렬
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))) ##각 노드의 특징 정보를 결합하여 attention 스코어를 계산하는 벡터
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh) ## attention score.

        zero_vec = -9e15*torch.ones_like(e).cuda() ##인접 노드가 아닌 경우에 매우 작은 값(-9e15)을 할당하여 softmax에서 무시
        attention = torch.where(adj > 0, e, zero_vec) ## 인접 노드에 대해서만 softmax를 통해 가중치를 적용
        attention = F.softmax(attention, dim=1) ## 즉, 오직 연결된 adj만 강조되도록 나머지를 전부 엄청 큰 음수로 만들어서, softmax에서 무시하게하도록 만듦.....(attention score 자체가 엣지가 없느 곳에서는 무의미하게함.)
        attention = F.dropout(attention, self.dropout, training=self.training) ## dropout을 적용하여 모델이 특정 특징에 과적합하지 않도록
        h_prime = torch.matmul(attention, Wh) ## attention과 Wh의 행렬 곱을 통해 출력 피처 행렬을 생성

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

## 노드 특징 Wh를 이용해 attention 스코어를 계산하기 위해 각 노드 쌍에 대한 값 e를 생성.
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) ##각 노드의 주어진 피처에 대해 weight를 적용한 벡터
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T ## 두 노드 간의 유사도를 통해 attention 스코어를 만들기 위해 broadcasting을 사용하여 더한다.
        return self.leakyrelu(e) ##최종적으로 LeakyReLU를 통해 e에 활성화함수를 적용하여 attention 스코어

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
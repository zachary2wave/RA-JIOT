import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import random
from torch.nn import init
import torch.nn.functional as F


class Mean_layer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, feature_dim, sample_model,
            embed_dim, attention_matrix=None,
            num_sample=10, sample_order="first", gcn=False, cuda=False):
        super(Mean_layer, self).__init__()
        self.num_sample = num_sample
        self.attention_matrix = attention_matrix
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.sample_model = sample_model

        self.sample_order = sample_order
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, feature_dim))
        init.xavier_uniform_(self.weight)
        if self.cuda:
            self.weight.cuda()
        # self.bias = nn.Parameter(torch.FloatTensor(embed_dim,))


    def forward(self, features_ue, features_ap, adj_lists_ue, adj_lists_ap, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        if self.sample_model == "ap2ue":
            other_feature = features_ue
            self_feature = features_ap
            one_ord_adj = adj_lists_ap
            sec_ord_adj = adj_lists_ue
        elif self.sample_model == "ap2ap":
            other_feature = features_ap
            self_feature = features_ap
            one_ord_adj = adj_lists_ap
            sec_ord_adj = adj_lists_ue
        elif self.sample_model == "ue2ap":
            other_feature = features_ap
            self_feature = features_ue
            one_ord_adj = adj_lists_ue
            sec_ord_adj = adj_lists_ap
        elif self.sample_model == "ue2ue":
            other_feature = features_ue
            self_feature = features_ue
            one_ord_adj = adj_lists_ue
            sec_ord_adj = adj_lists_ap

        if self.sample_order =="first":
            to_neighs = [one_ord_adj[int(node)] for node in nodes]
        elif self.sample_order == "second":
            to_neighs = []
            for node in nodes:
                one_ord = one_ord_adj[int(node)]
                if one_ord == set():
                    to_neighs.append(set(nodes))
                else:
                    samp_neighs = []
                    for one_node in one_ord:
                        samp_neighs.append(sec_ord_adj[int(one_node)])
                    to_neighs.append(set.union(*samp_neighs))

        neigh_feats = self.aggregator(other_feature, nodes, to_neighs)
        if not self.gcn:
            if self.cuda:
                self_feats = self_feature[nodes]
            else:
                self_feats = self_feature[nodes]
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined

    def aggregator(self, features, nodes, to_neighs):
        """
                nodes --- list of nodes in a batch
                to_neighs --- list of sets, each set is the set of neighbors for node in batch
                num_sample --- number of neighbors to sample. No sampling if None.
                """
        # Local pointers to functions (speed hack)
        _set = set
        if self.num_sample:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, self.num_sample, ))
                           if len(to_neigh) >= self.num_sample
                           else to_neigh
                           for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = [i for i in set.union(*samp_neighs)]
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        if self.cuda:
            mask = mask.cuda()
        if self.cuda:
            embed_matrix = features[unique_nodes_list]
        else:
            embed_matrix = features[unique_nodes_list]

        if self.attention_matrix is None:
            mask[row_indices, column_indices] = 1
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh+1)
        else:
            for flag, node, night in zip(range(len(nodes)), nodes, to_neighs):
                at_weights = F.softmax(self.attention_matrix[node, list(night)])
                for n, w in zip(night, at_weights):
                    mask[flag, n] = w
        to_feats = mask.mm(embed_matrix)
        return to_feats


class GraphSage_net(nn.Module):
    def __init__(self, APn, UEn,
                 Band_width = 20,
                 hidden_dim=128, num_sample=10,
                 cuda=False, gcn=False):
        super(GraphSage_net, self).__init__()
        self.Bandwidth = Band_width
        self.cuda = cuda
        self.gcn = gcn
        self.APn = APn
        self.UEn = UEn
        self.hidden_dim = hidden_dim
        UE_feature_num = APn
        AP_feature_num = UEn

        # layer
        self.layer1_AP_ord1 = Mean_layer(AP_feature_num+UE_feature_num, sample_model="ap2ue",
                                     embed_dim=hidden_dim,
                                     sample_order="first",
                                     num_sample=num_sample,
                                     cuda=self.cuda, gcn=self.gcn)
        self.layer1_AP_ord2 = Mean_layer(AP_feature_num*2,sample_model="ap2ap",
                                     embed_dim=hidden_dim,
                                     sample_order="second",
                                     num_sample =num_sample,
                                     cuda=self.cuda, gcn=self.gcn)

        self.layer2_AP_ord1 = Mean_layer(hidden_dim*4, sample_model="ap2ue",
                                     embed_dim=hidden_dim,
                                     sample_order="first",
                                     num_sample =num_sample,
                                     cuda=self.cuda, gcn=self.gcn)
        self.layer2_AP_ord2 = Mean_layer(hidden_dim*4, sample_model="ap2ap",
                                     embed_dim=hidden_dim,
                                     sample_order="second",
                                     num_sample=num_sample,
                                     cuda=self.cuda, gcn=self.gcn)

        self.layer1_UE_ord1 = Mean_layer(AP_feature_num+UE_feature_num, sample_model="ue2ap",
                                     embed_dim=hidden_dim,
                                     sample_order="first",
                                     num_sample=num_sample,
                                     cuda=self.cuda, gcn=self.gcn)

        self.layer1_UE_ord2 = Mean_layer(2*UE_feature_num, sample_model="ue2ue",
                                     embed_dim=hidden_dim,
                                     sample_order="second",
                                     num_sample =num_sample,
                                     cuda=self.cuda, gcn=self.gcn)

        self.layer2_UE_ord1 = Mean_layer(hidden_dim*4, sample_model="ue2ap",
                                     embed_dim=hidden_dim,
                                     sample_order="first",
                                     num_sample =num_sample,
                                     cuda=self.cuda, gcn=self.gcn)

        self.layer2_UE_ord2 = Mean_layer(hidden_dim*4, sample_model="ue2ue",
                                     embed_dim=hidden_dim,
                                     sample_order="second",
                                     num_sample=num_sample,
                                     cuda=self.cuda, gcn=self.gcn)
        self.weight_ap1 = nn.Parameter(torch.FloatTensor(hidden_dim * 2, hidden_dim * 2))
        self.weight_ap2 = nn.Parameter(torch.FloatTensor(1, hidden_dim*2))
        self.bais_ap = nn.Parameter(torch.FloatTensor(105))

        self.weight_ue = nn.Parameter(torch.FloatTensor(APn, hidden_dim*2))
        self.bais_ue = nn.Parameter(torch.FloatTensor(UEn))
        init.xavier_uniform_(self.weight_ap1)
        init.xavier_uniform_(self.weight_ap2)
        init.xavier_uniform_(self.weight_ue)
        init.uniform_(self.bais_ue,0,1)
        init.uniform_(self.bais_ap,0,1)
        if self.cuda:
            self.weight_ap.cuda()
            self.bais_ap.cuda()
            self.weight_ue.cuda()
            self.bais_ue.cuda()
        self.batchnormal = nn.BatchNorm1d(hidden_dim * 2)

    def forward(self, pl, require, adj_ue, adj_ap):

        APnodes = list(range(self.APn))
        UEnodes = list(range(self.UEn))

        # features_ap = nn.Embedding(self.APn, self.UEn)
        # features_ap.weight = nn.Parameter(torch.FloatTensor(pl), requires_grad=False)
        #
        # # f = np.concatenate((pl.T, require[:, np.newaxis]), axis=-1)
        # features_ue = nn.Embedding(self.UEn,  self.APn)
        # features_ue.weight = nn.Parameter(torch.FloatTensor(pl.T), requires_grad=False)
        # if self.cuda:
        #     features_ap.cuda()
        #     features_ue.cuda()

        pl =  torch.from_numpy(pl.astype(np.float32))
        x1 = self.layer1_AP_ord1(pl.T, pl, adj_ue, adj_ap, APnodes)
        x2 = self.layer1_AP_ord2(pl.T, pl, adj_ue, adj_ap, APnodes)
        x3 = self.layer1_UE_ord1(pl.T, pl, adj_ue, adj_ap, UEnodes)
        x4 = self.layer1_UE_ord2(pl.T, pl, adj_ue, adj_ap, UEnodes)
        X_ap = torch.cat((x1, x2), dim=0).t()
        # features_ap = nn.Embedding(self.APn, 2*self.hidden_dim)
        # features_ap.weight = nn.Parameter(torch.FloatTensor(X_ap), requires_grad=False)
        #
        X_ue = torch.cat((x3, x4), dim=0).t()
        # features_ue = nn.Embedding(self.UEn, 2*self.hidden_dim)
        # features_ue.weight = nn.Parameter(torch.FloatTensor(X_ue), requires_grad=False)

        x5 = self.layer2_AP_ord1(X_ue, X_ap, adj_ue, adj_ap, APnodes)
        x6 = self.layer2_AP_ord2(X_ue, X_ap, adj_ue, adj_ap, APnodes)
        x7 = self.layer2_UE_ord1(X_ue, X_ap, adj_ue, adj_ap, UEnodes)
        x8 = self.layer2_UE_ord2(X_ue, X_ap, adj_ue, adj_ap, UEnodes)
        X_ap = torch.cat((x5, x6), dim=0)

        hidden_pc = self.weight_ap1.mm(X_ap) + self.bais_ap
        PCoutcome = self.weight_ap2.mm(hidden_pc)
        X_ue = torch.cat((x7, x8), dim=0)
        X_ue = self.batchnormal(X_ue.t()).t()
        UAoutcome = self.weight_ue.mm(X_ue) + self.bais_ue

        PCoutcome = F.sigmoid(PCoutcome.t())
        nor = self.Max_min(UAoutcome.t())
        UAoutcome_de = F.softmax(nor*1000, dim=1)
        UAoutcome_st = F.softmax(nor, dim=1)
        UAoutcome_ori = UAoutcome.t()

        var_ue = torch.var(self.weight_ue,1)

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # plt.figure()
        # sns.heatmap(self.bais_ue.detach().numpy())
        # plt.show()



        return UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome, torch.sum(var_ue)

    def Normalized(self,x):
        mean = torch.mean(x,dim=1,keepdim=True)
        val  = torch.var(x,dim=1,keepdim=True)
        y = (x - mean) / val
        return y
    def Max_min(self,x):
        max = torch.max(x,dim=1,keepdim=True)
        min  = torch.min(x,dim=1,keepdim=True)
        y = (x - max.values) / (max.values-min.values)
        return y
    def loss_cal(self, pl, adj_ue, adj_ap, require):

        UAoutcome, PCoutcome = self.forward(pl, require, adj_ue, adj_ap)

        UAnum = torch.sum(UAoutcome, dim=0)
        sinr = torch.zeros([self.UEn])
        rate = torch.zeros([self.UEn])
        for ue in range(self.UEn):
            interfence = 0
            signal = 0
            for ap in range(self.APn):
                signal += UAoutcome[ue, ap] * (PCoutcome[ap]*20 - pl[ue, ap])
                interfence += PCoutcome[ap]*20 - pl[ue, ap]
            sinr[ue] = signal/(interfence-signal)
            connectAP_num = torch.sum(UAoutcome[ue, :]*UAnum)
            rate[ue] = self.Bandwidth/connectAP_num*torch.log2(sinr[ue])
        loss = -torch.sum(rate)
        return loss


import numpy as np
from torch import nn
import torch
import copy
from tqdm import tqdm
from args import dev
import myfunctions as fn
import pandas as pd

class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout=0, alpha=0.2):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=dev))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=dev))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        self.linear = nn.Linear(head_n, 1)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)

        # sparse metrics
        self.a_sparse = a_sparse
        self.edges = a_sparse.indices()
        self.values = a_sparse.values()
        self.N = a_sparse.shape[0]
        a_dense = a_sparse.to_dense()
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense

    def forward(self, x):
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            h = torch.matmul(x, self.heads_dict[n, 0])
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [Ni, Nj]
            atts = self.heads_dict[n, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        mt_atts = self.linear(mt_atts)
        new_values = self.values * mt_atts.squeeze()
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        atts_mat = atts_mat.to_dense() + self.mask
        atts_mat = self.softmax(atts_mat)
        return atts_mat


class GAT_LSTM(torch.nn.Module):
    def __init__(self, args, a_sparse1, a_sparse2):
        super().__init__()
        self.seq_len = args.LOOK_BACK
        self.gat_lyr1 = MultiHeadsGATLayer(a_sparse1, self.seq_len, self.seq_len, head_n=1)
        self.gat_lyr2 = MultiHeadsGATLayer(a_sparse2, self.seq_len, self.seq_len, head_n=1)
        self.gcn = nn.Linear(in_features=self.seq_len, out_features=self.seq_len).to(dev)
        self.LSTM1 = nn.LSTM(input_size=args.layer, hidden_size=args.layer, num_layers=2, batch_first=True)
        # self.fc0 = nn.Linear(in_features=lstm_hidden_size*features, out_features=output_size)
        self.fc1 = nn.Linear(in_features=11, out_features=args.k)  # w->k, CNN filters k=3
        self.fc2 = nn.Linear(in_features=args.k, out_features=args.layer)  # k->m, k=3, m=3
        self.fc3 = nn.Linear(in_features=args.k+args.layer, out_features=1)  # m->1, m=3

        self.layer = args.layer
        self.node = args.nodes
        self.dropout = nn.Dropout(p=0.2)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x, args):
        b, s, n = x.shape
        x = x.permute(0, 2, 1)
        atts_mat = self.gat_lyr1(x)  # dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, x)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))
        atts_mat = self.gat_lyr1(occ_conv1)  # dense(nodes, nodes)
        occ_conv2 = torch.matmul(atts_mat, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))
        # atts_mat = self.gat_lyr1(occ_conv2)  # dense(nodes, nodes)
        # occ_conv3 = torch.matmul(atts_mat, occ_conv2)  # (b, n, s)
        # occ_conv3 = self.dropout(self.LeakyReLU(self.gcn(occ_conv3)))

        atts_mat = self.gat_lyr2(x)  # dense(nodes, nodes)
        occ_conv4 = torch.matmul(atts_mat, x)  # (b, n, s)
        occ_conv4 = self.dropout(self.LeakyReLU(self.gcn(occ_conv4)))
        atts_mat = self.gat_lyr2(occ_conv4)  # dense(nodes, nodes)
        occ_conv5 = torch.matmul(atts_mat, occ_conv4)  # (b, n, s)
        occ_conv5 = self.dropout(self.LeakyReLU(self.gcn(occ_conv5)))
        # atts_mat = self.gat_lyr2(occ_conv5)  # dense(nodes, nodes)
        # occ_conv6 = torch.matmul(atts_mat, occ_conv5)  # (b, n, s)
        # occ_conv6 = self.dropout(self.LeakyReLU(self.gcn(occ_conv6)))

        x = torch.stack([x, occ_conv1, occ_conv2, occ_conv4, occ_conv5], dim=3)  # best
        x = x.reshape(b*n, self.seq_len, -1)
        lstm_out, _ = self.LSTM1(x)

        #-------------TPA------------
        ht = lstm_out[:, -1, :]  # 当前时刻的输出,ht(b,3)
        hw = lstm_out[:, :-1, :]  # 取前11个时刻的隐含层状态,x(b,11,3)
        hw = torch.transpose(hw, 1, 2)  # x(b,3,11)
        Hc = self.fc1(hw)  # 完成一维卷积，得到Hc,Hc(b,3,k) k=3
        # Hck = torch.transpose(Hc, 1, 2) #得到Hck,Hck(b,k,3),k=3
        Hn = self.fc2(Hc)  # 乘以系数矩阵进行维度变换去求权重,x(b,3,3)
        ht = torch.reshape(ht, (len(ht), self.layer, 1))  # ht(b,3,1)
        a = torch.bmm(Hn, ht)  # a(b,3,1)
        a = torch.sigmoid(a)  # 得到权重系数,a(b,3,1)
        a = torch.transpose(a, 1, 2)  # a (b, 1, 3)
        vt = torch.matmul(a, Hc)  # Hc行向量加权,vt(b,3,k),k=3,  [b, 1, k]
        ht = torch.reshape(ht, (len(ht), 1, self.layer))  # ht(b,1, 3)
        hx = torch.cat((vt, ht), dim=2)
        y = self.fc3(hx)  # yt(b,1,1)
        y = torch.squeeze(y, dim=2)  # (b, 1)
        TPA_y = torch.reshape(y, (-1, self.node))
        return TPA_y, a



def training(model, x, train_y_tensor, args):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_losses = []
    for epoch in tqdm(range(args.max_epochs)):
        # --------------train-------------
        model.train()
        optimizer.zero_grad()
        output, a = model(x,args)
        loss = loss_function(output, train_y_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(loss.item())
        if epoch+1 == args.max_epochs:
            a = a.mean(dim=0)
            print(a)
    out_loss_df = pd.DataFrame(columns=['train_loss'], data=train_losses)
    out_loss_df.to_csv(
        './result_' + str(args.predict_time) + '/' + 'p2_out_loss_' + '.csv',
        encoding='gbk')

def test(model, x, test_y_tensor, args):
    y_pre = model(x,args)
    y_pre = y_pre.detach().cpu().numpy()
    test_y_numpy = test_y_tensor.detach().cpu().numpy()
    output = fn.get_metrics(y_pre, test_y_numpy)
    return output

# Modified from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import argparse


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

def moex(x, pos, batch, swap_index, norm_type='pono', epsilon=1e-5):
    B = swap_index.size(0)
    T = batch.size(0) // B
    C = x.size(-1)
    x = x.view(B, T, C)
    if norm_type == 'bn':
        norm_dims = [0, 1]
    elif norm_type == 'in':
        norm_dims = [1]
    elif norm_type == 'ln':
        norm_dims = [1, 2]
    elif norm_type == 'pono':
        norm_dims = [2]
    elif norm_type.startswith('gpono'):
        if norm_type.startswith('gpono-d'):
            # gpono-d4 means Group PONO where each group has 4 dims
            G_dim = int(norm_type[len('gpono-d'):])
            G = C // G_dim
        else:
            # gpono4 means Group PONO with 4 groups
            G = int(norm_type[len('gpono'):])
            G_dim = C // G
        assert G * G_dim == C, f'{G} * {G_dim} != {C}'
        x = x.view(B, T, G, G_dim)
        norm_dims = [3]
    elif norm_type.startswith('gn'):
        if norm_type.startswith('gn-d'):
            # gn-d4 means GN where each group has 4 dims
            G_dim = int(norm_type[len('gn-d'):])
            G = C // G_dim
        else:
            # gn4 means GN with 4 groups
            G = int(norm_type[len('gn'):])
            G_dim = C // G
        assert G * G_dim == C, f'{G} * {G_dim} != {C}'
        x = x.view(B, T, G, G_dim)
        norm_dims = [2, 3]
    else:
        raise NotImplementedError(f'norm_type={norm_type}')
        
    mean = x.mean(dim=norm_dims, keepdim=True)
    std = x.var(dim=norm_dims, keepdim=True).add(epsilon).sqrt()
    swap_mean = mean[swap_index]
    swap_std = std[swap_index]
    # output = (x - mean) / std * swap_std + swap_mean
    # equvalent but for efficient
    scale = swap_std / std
    shift = swap_mean - mean * scale
    output = x * scale + shift

    return output.view(-1, C), pos, batch


class Net(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, data, swap_index=None):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        if swap_index is not None:
            sa1_out = moex(*sa1_out, swap_index=swap_index, norm_type=args.moex_norm, epsilon=args.moex_epsilon)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


def train(epoch, args):
    model.train()

    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if torch.rand(1) < args.moex_prob:
            swap_index = torch.randperm(data.y.size(0))
            output = model(data, swap_index=swap_index)
            loss = F.nll_loss(output, data.y)
            loss_b = F.nll_loss(output, data.y[swap_index])
            loss = args.moex_lambda * loss + (1 - args.moex_lambda) * loss_b
        else:
            output = model(data)
            loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(train_loader.dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='10', choices=['10', '40'])
    parser.add_argument('--moex_prob', default=0., type=float)
    parser.add_argument('--moex_lambda', default=1., type=float)
    parser.add_argument('--moex_norm', default='pono', type=str)
    parser.add_argument('--moex_epsilon', default=1e-5, type=float)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', f'data/ModelNet{args.data}')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, args.data, True, transform, pre_transform)
    test_dataset = ModelNet(path, args.data, False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(int(args.data)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        train_acc = train(epoch, args)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Train: {:.4f} Test: {:.4f}'.format(epoch, train_acc, test_acc), flush=True)
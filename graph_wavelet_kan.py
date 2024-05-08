import numpy as np
import numpy as np
import torch
from torch import nn
import torch_sparse as thsp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, APPNP, JumpingKnowledge
from torch_geometric.datasets import Planetoid
import argparse
import time
import os
from torch_geometric.utils import to_undirected
import os.path as osp
import pandas as pd
from kan import *
from wavelet_graph_kan_layer import NaiveWaveletKANLayer

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# KAN has gpu problems, so currently using cpu. 
device = 'cpu'
print(device)


class GraphKAN(nn.Module):
    def __init__(self, num_features, num_latent, num_classes, num_bases):
        super().__init__()
       
        self.waveletkan = NaiveWaveletKANLayer(inputdim = num_latent, outdim = num_latent)

        self.lin = nn.Linear(num_features, num_latent)
        self.num_features = num_features
        self.num_latent = num_latent
        self.num_classes = num_classes
        self.num_bases = num_bases

    def forward(self, X, edge_index):
        
        X_transform = self.waveletkan(self.lin(X.double()))
        X = torch.relu(X_transform)
        return F.log_softmax(X, dim=1)




# train/test    
def train(model, optimizer, data, train_mask, args):
    model.train()
    if args.model.lower() in ['graphkan']:
        outputs = model(data.x, data.edge_index)
    

    loss = F.nll_loss(outputs[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()



def mmte(model, data, train_mask, val_mask, test_mask, args):
    model.eval()
    if args.model.lower() in ['graphkan']:
        logits, accs = model(data.x, data.edge_index), []

    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def load_data(args, root='data', rand_seed=2023):
    dataset = args.dataset
    path = osp.join(root, dataset)
    dataset = dataset.lower()

    try:
        dataset = torch.load(osp.join(path, 'dataset.pt'))
        data = dataset['data']
        num_features = dataset['num_features']
        num_classes = dataset['num_classes']

    except FileNotFoundError:
        if dataset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dataset)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]

        torch.save(dict(data=data, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset.pt'))

    data.edge_index = to_undirected(data.edge_index, num_nodes =data.x.shape[0])

    return data, num_features, num_classes

# parameters 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type=str, default='./',
                        help='Parent directory: please change to where you like')
    parser.add_argument("--directory", type=str, default='graph_storage',
                        help='Directory to store trained models')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cora', help='Currently available: cora, citeseer, cornell, texas, wisconsin')
    parser.add_argument('--model', type=str, default='GraphKAN',
                        choices=['gcn', 'glgcn', 'mlp', 'ygcn'], help='GNN model')
    parser.add_argument('--num_bases', type=int, default=3, help='value of K for the B-spline')
    # optimization parameters 
    parser.add_argument('--runs', type=int, default=10, help='Number of repeating experiments for split.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.') # 0.005 cora 83.32
    parser.add_argument('--wd', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_hid', type=int, default=96, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
    # parser.add_argument('--train_rate',
    #                     type=float,
    #                     default=0.6,
    #                     help='Training rate.')  # used for heterophily 
    # parser.add_argument('--val_rate',
    #                     type=float,
    #                     default=0.2) # used for heterophily.


    args = parser.parse_args()


    torch.manual_seed(args.seed)
    
    data, num_features, num_classes = load_data(args)
    
    



# training
    results = []
    times = []
    
    
    for run in range(args.runs):
          
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        num_bases = 3
        log = 'Iteration {} starts.'
        print(log.format(run + 1))
        
        
        model = GraphKAN(num_features, 16, num_classes, num_bases) 
        # currently the feature output dimension is set as 16, regardless of nhid in args. KAN doesnt change dimension but provide a transformation
        model = model.to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

        t1 = time.time()
        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs + 1):

            train(model, optimizer, data, train_mask, args)
            with torch.no_grad():
                train_acc, val_acc, tmp_test_acc = mmte(model, data, train_mask, val_mask, test_mask, args)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
        t2 = time.time()
        results.append(test_acc)
        times.append(t2 - t1)

    results = 100 * torch.Tensor(results)
    times = torch.Tensor(times)
    print(results)
    print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f} in {times.mean():.2f} s')

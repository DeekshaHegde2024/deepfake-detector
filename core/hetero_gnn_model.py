import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from sklearn.model_selection import train_test_split

# -------------------- MODEL DEFINITION --------------------
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn as nn

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'interacts', 'video'): SAGEConv((-1, -1), hidden_dim),
            ('video', 'rev_interacts', 'user'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')

        self.lin_user = Linear(hidden_dim, out_dim)
        self.lin_video = Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return {
            'user': self.lin_user(x_dict['user']),
            'video': self.lin_video(x_dict['video'])
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, NNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import random
from torch_geometric.data import Data

  
class GCNConv_Model(torch.nn.Module):

    def __init__(self, num_classes, num_features, embedding_size=64):
        super().__init__()

        # GCNConv n'utilise pas edge_attr
        self.conv = GCNConv(num_features, embedding_size)
        self.bn_conv = BatchNorm1d(embedding_size)

        # MLP sophistiqué (x3)
        self.fc1 = Linear(embedding_size * 2, 256)
        self.bn1 = BatchNorm1d(256)
        self.drop1 = Dropout(p=0.3)

        self.fc2 = Linear(256, 128)
        self.bn2 = BatchNorm1d(128)
        self.drop2 = Dropout(p=0.3)

        self.fc3 = Linear(128, 64)
        self.bn3 = BatchNorm1d(64)
        self.drop3 = Dropout(p=0.2)

        self.out = Linear(64, num_classes)
    
    def forward(self, x, edge_index, batch_index):
        # edge_attr est ignoré ici
        x = self.conv(x, edge_index)
        x = self.bn_conv(x)
        x = F.relu(x)

        # Global pooling (mean + max)
        x = torch.cat([gap(x, batch_index), gmp(x, batch_index)], dim=1)

        # MLP
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)

        return self.out(x)
    
class NNGConv_Model(torch.nn.Module):

    def __init__(self, num_classes, num_features, num_types, embedding_size=64):
        super().__init__()

        # MLP utilisé pour générer les poids d'arêtes (edge_attr -> transformation)
        # Ici, edge_attr est de dimension 1 → on produit une matrice (num_features × embedding_size)
        self.edge_nn = Sequential(
            Linear(num_types, 16),
            ReLU(),
            Linear(16, num_features * embedding_size)
        )

        self.conv = NNConv(num_features, embedding_size, nn=self.edge_nn, aggr='mean')
        self.bn_conv = BatchNorm1d(embedding_size)

        # MLP sophistiqué (x3)
        self.fc1 = Linear(embedding_size * 2, 256)
        self.bn1 = BatchNorm1d(256)
        self.drop1 = Dropout(p=0.3)

        self.fc2 = Linear(256, 128)
        self.bn2 = BatchNorm1d(128)
        self.drop2 = Dropout(p=0.3)

        self.fc3 = Linear(128, 64)
        self.bn3 = BatchNorm1d(64)
        self.drop3 = Dropout(p=0.2)

        self.out = Linear(64, num_classes)
    
    def forward(self, x, edge_index, edge_attr, batch_index):
        # edge_attr doit être de shape [num_edges, 1]
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn_conv(x)
        x = F.relu(x)

        # Global pooling (mean + max)
        x = torch.cat([gap(x, batch_index), gmp(x, batch_index)], dim=1)

        # MLP
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)

        return self.out(x)

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable

    print(f"Total parameters:        {total:,}")
    print(f"Trainable parameters:    {trainable:,}")
    print(f"Non-trainable parameters:{non_trainable:,}")


if __name__=="__main__":
    model_gcnn = GCNConv_Model(num_classes=97, num_features=4, embedding_size=64)
    print("GCNN_Model")
    count_parameters(model_gcnn)
    
    nn_model_1 = NNGConv_Model(num_classes=97, cat=1, num_features=4, embedding_size=64)
    print("NNGConv_Model for Original")
    print(nn_model_1)
    count_parameters(nn_model_1)

    nn_model_2 = NNGConv_Model(num_classes=97, cat=8, num_features=4, embedding_size=64)
    print("NNGConv_Model for other types with k = 8")
    count_parameters(nn_model_2)
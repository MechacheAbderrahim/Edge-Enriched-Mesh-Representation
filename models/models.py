import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, NNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import random
from torch_geometric.data import Data



class GCNN_Model_3(nn.Module):
    def __init__(self, num_classes, num_features, embedding_size, dropout=0.3):
        super().__init__()
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        self.nn1 = nn.Linear(embedding_size * 2, 512)
        self.nn2 = nn.Linear(512, 256)
        self.nn3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch_index):
        x = F.relu(self.initial_conv(x, edge_index))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = torch.cat([gap(x, batch_index), gmp(x, batch_index)], dim=1)
        x = F.relu(self.nn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.nn2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.nn3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)
    
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

class GATConv_Model(torch.nn.Module):

    def __init__(self, num_classes, num_features, embedding_size=64, heads=4):
        super().__init__()

        # GATConv supporte plusieurs heads (parallèles) → on concatène les sorties
        self.gat = GATConv(
            in_channels=num_features,
            out_channels=embedding_size // heads,
            heads=heads,
            concat=True  # concatène les sorties des heads → dim finale = out_channels * heads
        )
        self.bn_conv = BatchNorm1d(embedding_size)

        # MLP (identique)
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
        # edge_attr est ignoré
        x = self.gat(x, edge_index)
        x = self.bn_conv(x)
        x = F.relu(x)

        x = torch.cat([gap(x, batch_index), gmp(x, batch_index)], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)

        return self.out(x)

class GATConv_Model_G2V(torch.nn.Module):

    def __init__(self, num_classes, num_features, embedding_size=64, heads=4, len_n= 128):
        super().__init__()

        # GATConv supporte plusieurs heads (parallèles) → on concatène les sorties
        self.gat = GATConv(
            in_channels=num_features,
            out_channels=embedding_size // heads,
            heads=heads,
            concat=True  # concatène les sorties des heads → dim finale = out_channels * heads
        )
        self.bn_conv = BatchNorm1d(embedding_size)

        # MLP (identique)
        self.fc1 = Linear(embedding_size * 2 + len_n, 256) # gap + gmp + g2v
        self.bn1 = BatchNorm1d(256)
        self.drop1 = Dropout(p=0.3)

        self.fc2 = Linear(256, 128)
        self.bn2 = BatchNorm1d(128)
        self.drop2 = Dropout(p=0.3)

        self.fc3 = Linear(128, 64)
        self.bn3 = BatchNorm1d(64)
        self.drop3 = Dropout(p=0.2)

        self.out = Linear(64, num_classes)

    def forward(self, x, edge_index, node_2_vec, batch_index):
        # edge_attr est ignoré
        x = self.gat(x, edge_index)
        x = self.bn_conv(x)
        x = F.relu(x)

        x = torch.cat([gap(x, batch_index), gmp(x, batch_index)], dim=1)
        
        x = torch.cat([x, node_2_vec], dim=1)
        
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
    """
    model_1 = GCNN_Model_1(num_classes=97, num_features=4, embedding_size=64)
    print("GCNN_Model_1")
    count_parameters(model_1)

    model_2 = GCNN_Model_1(num_classes=97, num_features=4, embedding_size=64)
    print("GCNN_Model_1")
    count_parameters(model_2)

    model_3 = GCNN_Model_1(num_classes=97, num_features=4, embedding_size=64)
    print("GCNN_Model_1")
    count_parameters(model_3)
    """
    nn_model_1 = NNGConv_Model(num_classes=97, cat=1, num_features=4, embedding_size=64)
    print("NNGConv_Model for Original")
    print(nn_model_1)
    count_parameters(nn_model_1)

    nn_model_2 = NNGConv_Model(num_classes=97, cat=8, num_features=4, embedding_size=64)
    print("NNGConv_Model for other types with k = 8")
    count_parameters(nn_model_2)
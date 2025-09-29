import time
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import json
import random

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from models.models import GCNConv_Model, NNGConv_Model
import argparse

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = -1

    def __call__(self, metric_set, epoch):
        if self.best_score is None:
            self.best_score = metric_set
            self.best_epoch = epoch
        elif metric_set > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"EarlyStopping: Arrêt de l'entraînement à l'époque {epoch}. Commencé à l'époque {self.best_epoch}.")
        else:
            self.best_score = metric_set
            self.best_epoch = epoch
            self.counter = 0

def update_attr(value_list, bins, one_hot):
    """
    Attribue à chaque valeur un vecteur one-hot correspondant à son intervalle dans bins.
    """
    value_array = np.array(value_list)
    encoded = np.digitize(value_array, bins[1:-1], right=False)  # entiers entre 0 et k-1
    if one_hot==False:
        return torch.tensor(encoded, dtype=torch.float32)
    else:
        num_classes = len(bins) - 1

        onehot = np.zeros((len(encoded), num_classes), dtype=np.float32)
        onehot[np.arange(len(encoded)), encoded] = 1.0
        return torch.tensor(onehot)

def get_dataset(mode, arch, k):

    data_dir = {
        "train": "data/train",
        "val": "data/train",
        "test": "data/test"
    }
    df = {
        "train" : pd.read_csv("data/df_train.csv"),
        "val" : pd.read_csv("data/df_val.csv"),
        "test" : pd.read_csv("data/test_labels.csv")
    }

    labels_dict = {}
    for phase in list(df.keys()):
        for i in range(len(df[phase])):
            p_id = df[phase].loc[i, "protein_id"]
            cls = df[phase].loc[i, "class_id"]
            labels_dict[p_id] = cls

    N_CLASSES = len(np.unique(list(labels_dict.values())))
    if N_CLASSES != 97:
        raise ValueError(f"{N_CLASSES} est inférieur à 97.")

    dataset = {}
    labels = {}

    if k > 1:
        bins = np.loadtxt(f"data/bins/{arch}_{k}.txt", dtype=float)

    for phase in list(df.keys()):
        X, y = [], []
        for file_name in tqdm(list(df[phase]["protein_id"]), desc=f"Loading {phase} data"):
            file_path = os.path.join(data_dir[phase], str(file_name)+".pt")
            if os.path.exists(file_path):
                graph = torch.load(file_path, weights_only=False)

                if mode.endswith("plus") == False:
                    graph.x = graph.x[:, :3]
                
                graph.y = torch.tensor(labels_dict[file_name], dtype=torch.long)
                y.append(graph.y)

                if k == 0:
                    del graph.edge_attr
                elif k == 1:
                    graph.edge_attr[:, 0] = graph.angles
                else:
                    graph.edge_attr[:, 0] = update_attr(graph.angles, bins, one_hot=False)

                del graph.angles
                del graph.descriptors
                del graph.angles_descriptors
                del graph.node_2_vec

                X.append(graph)
                
        dataset[phase] = X
        labels[phase] = np.array(y)
        
    return dataset, labels, N_CLASSES
    ####################################################################################################

def get_loaders(train_dataset, val_dataset, test_dataset, batch_size, seed):

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, generator=seed)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, generator=seed)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, generator=seed)

    return train_loader, val_loader, test_loader

def get_model(edge_dim, nb_classes, num_features, embedding_size, lr_params):
    if edge_dim == 1:
        model = GCNConv_Model(nb_classes, num_features, embedding_size)
    else:
        model = NNGConv_Model(nb_classes, num_features, edge_dim, embedding_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_params[0], weight_decay=lr_params[1])
    scheduler = StepLR(optimizer, step_size=lr_params[2], gamma=lr_params[3])
    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, loss_fn

def train(run_id, model, edge_dim, train_loader, val_loader, test_loader, epochs, device, nb_classes, loss_fnc, optimizer, scheduler, early_stopping, show, testing):

    metric_to_save = ["train f1", "train precision", "train recall", "val f1", "val precision", "val recall"]

    history = {
        "train loss":[],
        "train f1":[],
        "train precision":[],
        "train recall":[],
        "val loss":[],
        "val f1":[],
        "val precision":[],
        "val recall":[],
        "epoch_time" : [],
        "LR" : []
    }

    best_model_path = f"results/best_model_{run_id}.pt"

    step_lr = 0
    early = False
    start = time.time()
    if show: print(f"Start training")
    model.to(device)
    for ep in range(1, epochs+1):

        total_loss = 0.0
        y_train_preds = []
        y_train_true = []
        y_val_preds = []
        y_val_true = []
        
        
        model.train()
        start_ep = time.time()
        if show: print(f"Epoch {ep}")
        for batch in train_loader:
            step_lr = step_lr +1
            optimizer.zero_grad()
            
            batch = batch.to(device)
            X_batch = batch.x.float()
            edge_index_batch = batch.edge_index.long()
            y_true_batch = batch.y.clone().detach()
            y_true_encoded = F.one_hot(y_true_batch, num_classes=nb_classes).to(torch.float32)
            if edge_dim != 1:
                edge_attr_batch = batch.edge_attr.unsqueeze(1)

            if edge_dim != 1:
                pred_batch = model(X_batch, edge_index_batch, edge_attr_batch,batch.batch)
            else:
                pred_batch = model(X_batch, edge_index_batch, batch.batch)
            
            loss = loss_fnc(pred_batch, y_true_encoded)
            total_loss += loss

            y_train_preds.extend(torch.argmax(pred_batch, dim=1).cpu().detach().numpy())
            y_train_true.extend(y_true_batch.cpu().detach().numpy())

            loss.backward()
            optimizer.step()
        
        scheduler.step()
        end_ep = time.time()
        
        train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_train_true, y_train_preds, average='weighted')
        train_precision = precision_score(y_train_true, y_train_preds, average="weighted", zero_division=0)
        train_recall = recall_score(y_train_true, y_train_preds, average="weighted", zero_division=0)
        
        if show : print("Train Set : loss : {}, F1 {:.2f} %, Precision {:.2f} %, Recall {:.2f} %.".format(train_loss, train_f1*100, train_precision*100, train_recall*100))

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for batch in val_loader:

                batch = batch.to(device)
                X_batch = batch.x.float()
                edge_index_batch = batch.edge_index.long()
                y_true_batch = batch.y.clone().detach()
                y_true_encoded = F.one_hot(y_true_batch, num_classes=nb_classes).to(torch.float32)
                if edge_dim != 1:
                    edge_attr_batch = batch.edge_attr.unsqueeze(1)

                if edge_dim != 1:
                    pred_batch = model(X_batch, edge_index_batch, edge_attr_batch,batch.batch)
                else:
                    pred_batch = model(X_batch, edge_index_batch, batch.batch)

                loss = loss_fnc(pred_batch, y_true_encoded)
                total_loss += loss

                y_val_preds.extend(torch.argmax(pred_batch, dim=1).cpu().detach().numpy())
                y_val_true.extend(y_true_batch.cpu().detach().numpy())

            val_loss = total_loss / len(val_loader)
            val_f1 = f1_score(y_val_true, y_val_preds, average='weighted')
            val_precision = precision_score(y_val_true, y_val_preds, average="weighted", zero_division=0)
            val_recall = recall_score(y_val_true, y_val_preds, average="weighted", zero_division=0)

        print_info = "VAL Set : loss : {} , F1 {:.2f} %, Precision {:.2f} %, Recall {:.2f} %.".format(val_loss, val_f1*100, val_precision*100, val_recall*100)


        history["train loss"].append(train_loss.item())
        history["train f1"].append(train_f1)
        history["train precision"].append(train_precision)
        history["train recall"].append(train_precision)
        history["val loss"].append(val_loss.item())
        history["val f1"].append(val_f1)
        history["val precision"].append(val_precision)
        history["val recall"].append(val_precision)
        history["epoch_time"].append(end_ep - start_ep)
        history["LR"].append(scheduler.get_last_lr()[0])


        if val_f1 == np.max(history["val f1"]):
            torch.save(model, best_model_path)
            print_info += " \n (Best until this epoch)"

            best_scores = {metric:history[metric][-1]*100 for metric in metric_to_save}

        if show : print(print_info, '\n', "-"*20) 
        
        early_stopping(val_loss, ep)
        if early_stopping.early_stop:
            early = True
            break

    

    end = time.time()
    print(f"Train ended in {end - start} s.")
    df = pd.DataFrame(history)
    df.to_excel(f"results/train_results_{run_id}.xlsx", index=None)

    if early:
        print("Arrêt anticipé déclenché.")

    if testing:
        best_scores["test f1"], best_scores["test precision"], best_scores["test recall"], best_scores["inference time"] = test(run_id, best_model_path, edge_dim, test_loader, device)

    return best_scores
        
def test(run_id, best_model_path, edge_dim, test_loader, device):
    best_model = torch.load(best_model_path, weights_only=False)
    best_model.to(device)
    best_model.eval()

    ids = []
    y_test_preds = []
    y_test_true = []

    test_begin = time.time()
    with torch.no_grad():

        for batch in test_loader:
            batch = batch.to(device)
            X_batch = batch.x.float()
            edge_index_batch = batch.edge_index.long()
            y_true_batch = batch.y.clone().detach()

            if edge_dim != 1:
                edge_attr_batch = batch.edge_attr.unsqueeze(1)

            if edge_dim != 1:
                pred_batch = best_model(X_batch, edge_index_batch, edge_attr_batch,batch.batch)
            else:
                pred_batch = best_model(X_batch, edge_index_batch, batch.batch)

            ids.extend(batch.name)
            y_test_preds.extend(torch.argmax(pred_batch, dim=1).cpu().detach().numpy())
            y_test_true.extend(y_true_batch.cpu().detach().numpy())
            
        test_time = time.time() - test_begin
        test_f1 = f1_score(y_test_true, y_test_preds, average='weighted')*100
        test_precision = precision_score(y_test_true, y_test_preds, average="weighted", zero_division=0)*100
        test_recall = recall_score(y_test_true, y_test_preds, average="weighted", zero_division=0)*100

        print("Test Set : F1 {:.2f} %, Precision {:.2f} %, Recall {:.2f} %.".format( test_f1, test_precision, test_recall))

    df_test = pd.DataFrame({
        "Protein_ID" : ids,
        "Cluster" :  y_test_true,
        "Predicted_Cluster" : y_test_preds,
    }, index=None)

    df_test.to_excel(f"results/Test_predictions_{run_id}.xlsx", index=None)
    return test_f1, test_precision, test_recall, test_time
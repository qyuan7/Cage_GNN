#!/usr/bin/env python

import torch
import pandas as pd
import numpy as np
from cage_gnn import *
import torch.optim as optim
import torch.nn.functional as F 
import pickle
from sklearn.metrics import precision_score, recall_score
torch.manual_seed(0)

def test(model, dataset, batch_size, val=True, device='cpu'):
    model.to(device)
    model.eval()
    y_preds, y_trues = [], []
    N = len(dataset)
    criterion = nn.CrossEntropyLoss()
    loss_total = 0
    for i in range(0, N, batch_size):
        data_batch = list(zip(*dataset[i:i+batch_size]))
        y_true = data_batch[-1]
        y_true = torch.stack(y_true)
        m1_atoms, adj1, m2_atoms, adj2,label = data_batch
        output = model(m1_atoms, adj1, m2_atoms, adj2)
        #y_true = torch.cat(data_batch[-1])
        _, y_pred = torch.max(output, 1)
        y_preds.extend(y_pred)
        y_trues.extend(y_true)
        loss = criterion(output, y_true)
        loss_total += loss
    y_trues = torch.stack(y_trues)
    y_preds = torch.stack(y_preds)
    corr = sum(y_preds == y_trues)
    precision = precision_score(y_trues, y_preds)
    recall = precision_score(y_trues, y_preds)
    if val:
        print("Val set accuracy {:.3f}; preicision:{:.3f}; recall:{:.3f}; loss:{:.3f}."
              .format(corr.item()/len(dataset), precision, recall, loss_total))
    else:
        print("Test set accuracy {:.3f}; preicision:{:.3f}; recall:{:.3f}; loss:{:.3f}."
              .format(corr.item()/len(dataset), precision, recall, loss_total))
    return loss_total


def train(model, dataset, batch_size,epoch, device='cpu'):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    np.random.seed(0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5)
    np.random.shuffle(dataset)
    N = len(dataset)
    N_train = int(N * 0.7)
    N_big_test = N - N_train
    loss_total = 0
    train_set = dataset[:N_train]
    big_test_set = dataset[N_train:]
    N_val = N_big_test // 2
    val_set = big_test_set[:N_val]
    test_set = big_test_set[N_val:]
    for e in range(epoch):
        total_loss = 0
        corr = 0
        data_size = 0
        for i in range(0, N_train, batch_size):
            optimizer.zero_grad()
            data_batch = list(zip(*train_set[i:i+batch_size]))
            y_true = data_batch[-1]
            y_true = torch.stack(y_true)
            m1_atoms, adj1, m2_atoms, adj2, _ = data_batch
            output = model(m1_atoms, adj1, m2_atoms, adj2)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss
            _, y_pred = torch.max(output, 1)
            #print(y_pred)
            b_corr = sum(y_pred == y_true)
            corr += b_corr
            data_size += len(y_true)
        acc = corr.item()/data_size
        print(f"total loss for epoch {e} is {total_loss:.2f}\t accuracy {acc:.3f}")
        val_loss = test(model, val_set, batch_size)
        test_loss = test(model, test_set, batch_size,val=False)
        scheduler.step(val_loss)
    if e == epoch-1:
        torch.save(model.state_dict(), 'test_model.ckpt')


def main():
    model = graph_cage(120,64,3)
    with open("db_neighbour/test_500_db.ckpt", "rb") as f:
        data = pickle.load(f)
    train(model, data, 64, 120)

if __name__ == '__main__':
    main()
        

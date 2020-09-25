#!/usr/bin/env python

import torch
import pandas as pd
import numpy as np
from cage_gnn_sum import *
import torch.optim as optim
import torch.nn.functional as F 
import pickle
from sklearn.metrics import precision_score, recall_score
torch.manual_seed(0)

def test(model, dataset, batch_size, device='cpu'):
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
        output = model(data_batch)
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
    print("test set accuracy {:.3f}; preicision:{:.3f}; recall:{:.3f}; loss:{:.3f}."
          .format(corr.item()/len(dataset), precision, recall, loss_total))


def train(model, dataset, batch_size,epoch, device='cpu'):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.006)
    np.random.seed(0)

    np.random.shuffle(dataset)
    N = len(dataset)
    N_train = int(N * 0.8)
    N_test = N - N_train
    loss_total = 0
    train_set = dataset[:N_train]
    test_set = dataset[N_train:]
    for e in range(epoch):
        total_loss = 0
        corr = 0
        data_size = 0
        for i in range(0, N_train, batch_size):
            optimizer.zero_grad()
            data_batch = list(zip(*train_set[i:i+batch_size]))
            y_true = data_batch[-1]
            y_true = torch.stack(y_true)
            output = model(data_batch)
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
        test(model, test_set, batch_size)
    if e == epoch-1:
        torch.save(model.state_dict(), 'test_model.ckpt')


def main():
    model = graph_cage(120,64,3)
    with open("db_neighbour/test_500_db.ckpt", "rb") as f:
        data = pickle.load(f)
    train(model, data, 64, 120)

if __name__ == '__main__':
    main()
        

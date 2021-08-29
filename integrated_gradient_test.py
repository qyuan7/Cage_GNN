#!/usr/bin/env python

from cage_gnn import *
import pickle
from gnn_integrated_gradients import *
import pandas as pd


def pred_softmax(df, data):
    pred_idxs, soft_reses, base_reses = [], [], []
    for dd in data:
        pred_idx, soft_res, base_res, _ = predict(dd, gnn)
        pred_idxs.append(pred_idx)
        soft_reses.append(soft_res)
        base_reses.append(base_res)
    df['pred_label'] = pred_idxs
    df['inpt_softmax'] = soft_reses
    df['base_softmax'] = base_reses
    print(sum(df.label==df.pred_label)/len(df))
    return df

def inte_gradient_batch(df, data):
    avg_atom1s, avg_adj1s, avg_atom2s, avg_adj2s = [], [],[],[]
    for i, dd in enumerate(data):
        avg_atom1, avg_adj1, avg_atom2, avg_adj2 = integrated_gradient(dd, gnn, target_label_idx=None, baseline=None, step=100)
        avg_atom1s.append(np.sum(avg_atom1)),
        avg_adj1s.append(np.sum(avg_adj1)), 
        avg_atom2s.append(np.sum(avg_atom2)),
        avg_adj2s.append(np.sum(avg_adj2))
        if i % 100 == 0 and i !=0:
            print(f"{i} cages processed")
    df['m1_ig'] = avg_atom1s
    df['adj1_ig'] = avg_adj1s
    df['m2_ig'] = avg_atom2s
    df['adj2_ig'] = avg_adj2s
    return df

if __name__ == '__main__':
    gnn = graph_cage(120,128,2)
    #gnn.load_state_dict(torch.load('gnn_inte_allbalance_2seed.pt',map_location=lambda storage, loc:storage))
    #gnn.load_state_dict(torch.load('gnn_inte_traintest_2seed.pt', map_location=lambda storage, loc:storage))
    gnn.load_state_dict(torch.load('gnn_allvsone1_2seed.pt', map_location=lambda storage, loc:storage))
    gnn.eval()
    #with open("db_neighbour_grad/all_cages_train.ckpt", "rb") as f:
    #    data = pickle.load(f)
    with open("db_neighbour_grad/all_cages.ckpt", "rb") as f:
        data = pickle.load(f)
        data = data[:4583]
    
    #df = pd.read_csv('filtered_train_raw.csv')
    df = pd.read_csv("filtered_all_smi.csv")
    df_ = df[df.reaction == "amine2aldehyde3"]
    df_sf = pred_softmax(df_, data)
    df_contrib = inte_gradient_batch(df_sf, data)
    #df_contrib.to_csv('filtered_train_ig_contrib_100step.csv',index=False)
    df_contrib.to_csv("ig_allvsone1_100step.csv", index=False)

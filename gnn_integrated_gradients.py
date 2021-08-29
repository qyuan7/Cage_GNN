#from utils import integrated_gradients
#from utils.integrated_gradients import predict_and_gradients
#from utils.gcn_utils import load_data, data_process
from cage_gnn import graph_cage
import torch
import logging
import sqlite3
import torch.utils.data as utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 


logger = logging.getLogger(__name__)


def dataset_prep(db, reactions, topologies):
    logger.debug(f'Reactions: {reactions}.')
    logger.debug(f'Topologies: {topologies}.')
    fps, tops, labels = load_data(db, reactions, topologies=topologies, cage_property=False)
    fps, labels = data_process(fps, labels)
    dataset = utils.TensorDataset(fps, labels)
    dataset = utils.DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    return dataset

def create_base(inpt, baseline=None):
    m1_atom, adj1, m2_atom, adj2, label = inpt

    if baseline is None:
        atom1_base = torch.zeros(m1_atom.shape, requires_grad=True)
        #adj1_base = torch.zeros(adj1.shape, requires_grad=True)
        adj1_base = adj1.clone().detach().requires_grad_()
        atom2_base = torch.zeros(m2_atom.shape, requires_grad=True)
        #adj2_base = torch.zeros(adj2.shape, requires_grad=True)
        adj2_base = adj2.clone().detach().requires_grad_()
    base_inpt = (atom1_base, adj1_base, atom2_base, adj2_base)
    return base_inpt
    

def predict(inpt, model):
    m1_atom, adj1, m2_atom, adj2, label = inpt
    base_inpt = create_base(inpt)
    atom1_base, adj1_base, atom2_base, adj2_base = base_inpt
    m1_atoms, adj1s, m2_atoms, adj2s = (m1_atom, m1_atom), (adj1, adj1,), (m2_atom, m2_atom), (adj2, adj2)
    atom1_bases, adj1_bases, atom2_bases, adj2_bases = (atom1_base, atom1_base), (adj1_base, adj1_base), (atom2_base, atom2_base), (adj2_base, adj2_base)
    output = model(m1_atoms, adj1s, m2_atoms, adj2s)
    output_base = model(atom1_bases, adj1_bases, atom2_bases, adj2_bases)
    output = F.softmax(output, dim=1)
    output_base = F.softmax(output_base, dim=1)
    #print(output_base)
    idx = torch.argmax(output[0]).item()
    #idx = torch.tensor(idx, dtype=torch.int64)
    #idx = idx.to(device)
    #out = out.gather(0, idx)
    soft_res = output[0][idx].item()
    base_res = output_base[0][idx].item()
    return idx, soft_res, base_res, output_base[0].detach().numpy().reshape((-1,2))

def predict_and_gradient(inpt,model, target_label_idx):
    # make a pseudo batch of the same input
    m1_atom, adj1, m2_atom, adj2, label = inpt
    m1_atoms, adj1s, m2_atoms, adj2s = (m1_atom, m1_atom), (adj1, adj1,), (m2_atom, m2_atom), (adj2, adj2)
    output = model(m1_atoms, adj1s, m2_atoms, adj2s)
    output = F.softmax(output, dim=1)
    #print(output)
    if target_label_idx is None:
        target_label_idx = torch.argmax(output[0]).item()
    idx = target_label_idx
    #idx = torch.tensor(idx, dtype=torch.int64)
    #idx = idx.to(device)
    #out = out.gather(0, idx)
    model.zero_grad()
    output[0][idx].backward()
    atom1_g = m1_atoms[0].grad.detach().cpu().numpy()
    adj1_g = adj1s[0].grad.detach().cpu().numpy()
    atom2_g = m2_atoms[0].grad.detach().cpu().numpy()
    adj2_g = adj2s[0].grad.detach().cpu().numpy()
    return atom1_g, adj1_g, atom2_g, adj2_g


def integrated_gradient(inpt, model, target_label_idx, baseline, step=50):
    m1_atom, adj1, m2_atom, adj2, label = inpt
    base_inpt = create_base(inpt)
    atom1_base, adj1_base, atom2_base, adj2_base = base_inpt

    scaled_atom1s = [atom1_base + (i/step)*(m1_atom-atom1_base) for i in range(0, step+1)]
    scaled_adj1s = [adj1_base + (i/step)*(adj1-adj1_base) for i in range(0, step+1)]
    scaled_atom2s = [atom2_base + (i/step)*(m2_atom-atom2_base) for i in range(0, step+1)]
    scaled_adj2s = [adj2_base + (i/step)*(adj2-adj2_base) for i in range(0, step+1)]
    scaled_labels = [label for _ in range(0, step+1)]
    atom1_grads, adj1_grads, atom2_grads, adj2_grads = [], [], [], []
    for atom1, adj1, atom2, adj2, label in zip(scaled_atom1s, scaled_adj1s, scaled_atom2s, scaled_adj2s, scaled_labels):
        atom1 = atom1.clone().detach().requires_grad_()
        adj1 = adj1.clone().detach().requires_grad_()
        atom2 = atom2.clone().detach().requires_grad_()
        adj2 = adj2.clone().detach().requires_grad_()
        inpt = (atom1, adj1, atom2, adj2, label)
        atom1_grad, adj1_grad, atom2_grad, adj2_grad = predict_and_gradient(inpt, model=model, target_label_idx=target_label_idx)
        #print(atom1_grad.shape)
        atom1_grads.append(atom1_grad)
        adj1_grads.append(adj1_grad)
        atom2_grads.append(atom2_grad)
        adj2_grads.append(adj2_grad)
    avg_atom1 = np.average(atom1_grads,axis=0)
    avg_adj1 = np.average(adj1_grads,axis=0)
    avg_atom2 = np.average(atom2_grads,axis=0)
    avg_adj2 = np.average(adj2_grads,axis=0)
    int_atom1 = (m1_atom.detach().numpy() - atom1_base.detach().numpy()) * avg_atom1
    int_adj1 = (adj1.detach().numpy() - adj1_base.detach().numpy())*avg_adj1
    int_atom2 = (m2_atom.detach().numpy() - atom2_base.detach().numpy()) *avg_atom2
    int_adj2 = (adj2.detach().numpy() - adj2_base.detach().numpy())*avg_adj2
    return int_atom1, int_adj1, int_atom2, int_adj2


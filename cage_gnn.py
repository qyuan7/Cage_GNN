#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class graph_cage(nn.Module):
    def __init__(self,atom_dim, hidden_dim, depth,  dropout=0.0,task='classification',device="cpu"):
        super().__init__()
        self.proj_atom = nn.Linear(atom_dim, hidden_dim)
        self.gen_fp = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                    for _ in range(depth)])
        self.merge_fp = nn.Linear(2*hidden_dim, hidden_dim,bias=False)
        self.mlp = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                  for _ in range(2)])
        if task == 'classification':
            self.output = nn.Linear(hidden_dim, 2)
        else:
            self.output = nn.Linear(hidden_dim, 1)
        self.depth = depth
        self.dropout = dropout
        self.device = torch.device(device)

    @staticmethod
    def pad(matrices, pad_value=0):
        shapes = [m.shape for m in matrices]
        M,  N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        m_sizes = [s[0] for s in shapes]
        zeros = torch.FloatTensor(np.zeros((M, N)))
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices, m_sizes
    
    def update(self, adj, vectors, layer):
        hidden_vectors = torch.relu(self.gen_fp[layer](vectors))
        return hidden_vectors + torch.matmul(adj, hidden_vectors)

    
    def sum(self, vectors, m_sizes):
        sum_vectors = [torch.sum(v,0) for v in torch.split(vectors, m_sizes)]
        return torch.stack(sum_vectors)
    
    def gnn(self, atoms, adjs):
        atom_fps = torch.cat(atoms)
        atom_fps = atom_fps.float()
        atom_fps = atom_fps.to(self.device)
        #print(atom_fps.type())
        adjs, m_sizes = self.pad(adjs)
        adjs = adjs.to(self.device)
        atom_vectors = self.proj_atom(atom_fps)
        for layer in range(self.depth):
            hs = self.update(adjs, atom_vectors, layer)
            atom_vectors = F.normalize(hs, 2, 1)
        molecular_vectors = self.sum(atom_vectors, m_sizes)
        return molecular_vectors
    
   
           
    def forward(self, m1_atoms, adj1, m2_atoms, adj2):
        #m1_atoms, adj1, m2_atoms, adj2,label = inputs
        m1_vectors = self.gnn(m1_atoms, adj1)
        m2_vectors = self.gnn(m2_atoms, adj2)
        m_vectors = torch.cat((m1_vectors, m2_vectors), dim=1)
        m_vectors = self.merge_fp(m_vectors)
        for i in range(2):
            m_vectors = F.relu(self.mlp[i](m_vectors))
            m_vectors = F.dropout(m_vectors, p=self.dropout, training=self.training)
        score = self.output(m_vectors)
        return score




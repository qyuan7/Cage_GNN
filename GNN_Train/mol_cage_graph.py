#!/usr/bin/env python

import rdkit
from rdkit import Chem
import numpy as np
import torch


elem_list = ['Cl', 'Si', 'C', 'B', 'O', 'Br', 'F', 'S', 'N', 'H', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)

def atom_graph(mol, idxfunc=lambda x:x.GetIdx()):
    if not mol:
        raise ValueError("Ileagal mol found!")
    n_atoms = mol.GetNumAtoms()
    fatoms = np.zeros((n_atoms, atom_fdim))
    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        if idx > n_atoms:
            raise Exception("Atom id larger than total number of atoms!")
        fatoms[idx] = atom_features(atom)
    return torch.tensor(fatoms)

def adj_graph(mol, idxfunc=lambda x:x.GetIdx()):
    if not mol:
        raise ValueError("Ileagal mol found!")
    n_atoms = mol.GetNumAtoms()
    adj = np.zeros((n_atoms, n_atoms))
    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        order = bond.GetBondTypeAsDouble()
        adj[a1][a2] = order
        adj[a2][a1] = order
    return torch.tensor(adj)
    

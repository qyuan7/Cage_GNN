#!/usr/bin/env python
import pandas as pd
from mol_cage_graph import *
import pickle

def prepare_db(fname, reactions):
    df = pd.read_csv(fname)
    sub_df = df[df['reaction'].isin(reactions)]
    smis1 = sub_df['SMILES_1'].tolist()
    smis2 = sub_df['SMILES_2'].tolist()
    labels = sub_df['label'].tolist()
    dataset = []
    for smi1, smi2, label in zip(smis1, smis2, labels):
        m1 = Chem.MolFromSmiles(smi1)
        m2 = Chem.MolFromSmiles(smi2)
        atom_graph1 = atom_graph(m1)
        adj1 = adj_graph(m1)
        atom_graph2 = atom_graph(m2)
        adj2 = adj_graph(m2)
        label = torch.tensor(label, dtype=torch.long)
        dataset.append((atom_graph1, adj1, atom_graph2, adj2, label))
    return dataset


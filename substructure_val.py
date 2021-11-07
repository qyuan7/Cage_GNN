#!/usr/bin/env python

import pandas as pd
from rdkit import Chem

def substructure1_val(smi):
    substruct = Chem.MolFromSmiles(smi)
    df = pd.read_csv("filtered_all_smi.csv")
    ids, labels = [], []
    smiles = df.SMILES_1.tolist()
    smiles2 = df.SMILES_2.tolist()
    all_labels = df.label.tolist()
    for i, smi in enumerate(smiles):
        m1 = Chem.MolFromSmiles(smi)
        m2 = Chem.MolFromSmiles(smiles2[i])
        if m1.HasSubstructMatch(substruct) or m2.HasSubstructMatch(substruct):
            ids.append(i)
            labels.append(all_labels[i])
    prob = sum(labels)/len(labels)
    print(f"{len(labels)} cages has substruct, {int(sum(labels))} collapsed")
    return prob
    
if __name__ == '__main__':
    substructure1_val("C=C1C2=C3C4=C5C6=C7C3=C(C(C7=CC=C6C(C5=CC=C14)=C)=C)C=C2")

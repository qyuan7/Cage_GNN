#!/usr/bin/env python

import pickle
import torch
import random
random.seed(0)
#with open("db_neighbour_grad/all_cages.ckpt", "rb") as f:
with open("db_neighbour_grad/all_cages_train.ckpt", "rb") as f:
    data = pickle.load(f)
collapsed_origin = []
persistent_origin = []
collapsed = []
persistent = []
cnt = 0
#idx_all = list(range(16921))
#pers_idx = random.sample(idx_all, 12167)
idx_all = list(range(10305))
add_idx = random.sample(idx_all, 4114)
with open("db_neighbour_grad/sampled_idx_notest.ckpt","wb") as f:
    pickle.dump(add_idx, f)
for dd in data:
    m1_atom, adj1, m2_atom, adj2, label = dd
    if label == 0:
        cnt += 1
        atom1_base = torch.zeros(m1_atom.shape, requires_grad=True)
        adj1_base = torch.zeros(adj1.shape, requires_grad=True)
        atom2_base = torch.zeros(m2_atom.shape, requires_grad=True)
        adj2_base = torch.zeros(adj2.shape, requires_grad=True)
        base = (atom1_base, adj1_base, atom2_base, adj2_base, label)
        persistent.append(base)
        persistent_origin.append(dd)
    else:
        atom1_base = torch.zeros(m1_atom.shape, requires_grad=True)
        adj1_base = torch.zeros(adj1.shape, requires_grad=True)
        atom2_base = torch.zeros(m2_atom.shape, requires_grad=True)
        adj2_base = torch.zeros(adj2.shape, requires_grad=True)
        base = (atom1_base, adj1_base, atom2_base, adj2_base, label)
        collapsed.append(base)
        collapsed_origin.append(dd)
#coo = [collapsed[i] for i in pers_idx]
#coo_origin = [collapsed_origin[i] for i in pers_idx]
add_per_origin = [persistent_origin[i] for i in add_idx]
add_per = [persistent[i] for i in add_idx]
final_d = data + add_per_origin + collapsed+ persistent + add_per 
print(cnt==10305)
print(len(final_d))
#with open("db_neighbour_grad/aug_cages_all_balance.ckpt", "wb") as f:
#    pickle.dump(final_d, f) 
with open("db_neighbour_grad/aug_cages_train_all_balance.ckpt", "wb") as f:
    pickle.dump(final_d ,f)

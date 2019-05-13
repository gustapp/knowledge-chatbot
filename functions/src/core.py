#%%
import functions.src.OpenKE.config as config
import functions.src.OpenKE.models as models
# import OpenKE.config as config
# import OpenKE.models as models
import tensorflow as tf
import numpy as np
import json
from flask import jsonify
__file__ = './functions/src/'
""" Load TransE FB13 Knowledge Embedding """
from os import path
root = path.dirname(path.abspath(__file__))

#%%

con = config.Config()

fb13_path = path.join(root, 'src/OpenKE/benchmarks/FB13/')
ke_path =  path.join(root, 'src/OpenKE/res_fb13/model.vec.tf')

con.set_in_path(fb13_path)
con.set_work_threads(8)
con.set_dimension(100)
con.set_import_files(ke_path)
con.init()
con.set_model(models.TransE)

#%%
""" Load Dictionaries: 
    * relation2id
    * entity2id
"""
entity2id_path = path.join(root, 'src/OpenKE/benchmarks/FB13/entity2id.txt')
relation2id_path = path.join(root, 'src/OpenKE/benchmarks/FB13/relation2id.txt')

import pandas as pd
e2i_df = pd.read_csv(entity2id_path, sep='\t', header=None, skiprows=[0])
e2i_df.columns = ['entity', 'id']
    
r2i_df = pd.read_csv(relation2id_path, sep='\t', header=None, skiprows=[0])
r2i_df.columns = ['relation', 'id']

#%%
e_h = 'jesus'
r = 'religion'

""" Predict Tail """
eh_id = e2i_df[e2i_df['entity'] == e_h]['id'].values[0]
r_id = r2i_df[r2i_df['relation'] == r]['id'].values[0]

res_ids = con.predict_tail_entity(h=eh_id, r=r_id, k=10)

[e2i_df[e2i_df['id'] == res_id]['entity'] for res_id in res_ids]

#%% [markdown]
# ## Counter Xtrike Method

#%%
# Load Embedding
ke = con.get_parameters()

ke_ent = ke['ent_embeddings']
ke_rel = ke['rel_embeddings']

#%%
eh_vec = ke_ent[eh_id]

#%%
noise = np.random.normal(0,0.05,100)

e_hat_sample = np.sum([eh_vec, noise], axis=0)

np.mean(abs(eh_vec - e_hat_sample))
# dissimilarity(jesus, jesus_hat_sample)

#%%
# Generate perturbed set of instances
e = eh_vec
n_instances = 1000
dimension = 100
noise_rate = 0.05

e_hat = []
for i in range(0, n_instances):
  noise = np.random.normal(0,noise_rate,dimension)
  # e_noise = np.sum([e, noise], axis=0)
  e_hat.append(e + noise)

#%%
# Minimize search area by choosing only the nearest neighbors
""" single prediction """
head_ent = eh_id #jesus
rel = r_id # religion

k_nn = 10
feats_tb = []
""" discover head entity features """
for rel_id in range(0,13):
  feat_candidates_per_relation = con.predict_tail_entity(h=head_ent, r=rel_id, k=k_nn)
  feats_per_rel = []
  # for candidate in feat_candidates_per_relation:
  #   is_true = con.predict_triple(h=head_ent, r=rel_id, t=candidate)
  #   if is_true:
  #     feats_per_rel.append(candidate)
  # feats_tb.append((rel_id, feats_per_rel))
  feats_tb.append((ke_rel[rel_id], [(ent_id, ke_ent[ent_id]) for ent_id in feat_candidates_per_relation]))

#%%
e_hat_feats = []
# for rel in ke_rel:
for rel, k_tails in feats_tb:
  labels = []
  for e_fake in e_hat:
    """ TransE inference """
    # e_rel = np.sum([e_fake, rel], axis=0)
    dist_per_inst = []
    id_per_inst = []
    """ Identify nearest entity to inference """
    # for tail_cand in ke_ent:
    for tail_id, tail_cand in k_tails:
      dist = np.mean(abs(e_fake + rel - tail_cand))
      # dist = np.linalg.norm(e_rel-tail_cand)
      dist_per_inst.append(dist)
      id_per_inst.append(tail_id)
    """ Classify @1 """
    tail = id_per_inst[dist_per_inst.index(min(dist_per_inst))]
    labels.append(tail)
  e_hat_feats.append(labels)
  print(str(len(e_hat_feats)))

#%%
feats_names = ['religion', 'cause_of_death', 'place_of_death', 'profession', 'location', 'gender', 'nationality', 'place_of_birth', 'institution', 'children', 'parents', 'spouse', 'ethnicity']
e_hat_feats_df = pd.DataFrame(data=list(map(list,zip(*e_hat_feats))), columns=feats_names)
e_hat_feats_df = e_hat_feats_df.applymap(lambda id: e2i_df[e2i_df['id'] == id]['entity'].values[0])
e_hat_feats_df.head(50)

#%%
e_hat_feats_df.to_csv('franz_ledermann_cod.csv')
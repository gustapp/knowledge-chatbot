#%%
import config
import models
import tensorflow as tf
import numpy as np
import json

#%%
con = config.Config()
con.set_in_path("./benchmarks/FB13/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(4)
con.set_dimension(50)
con.set_import_files("./res/model.vec.tf")
con.init()
con.set_model(models.TransE)

#%%
# Get the embeddings (numpy.array)
embeddings = con.get_parameters("numpy")
# Get the embeddings (python list)
embeddings = con.get_parameters()

#%%
""" <antoine_brutus_menier, religion, roman_catholic_church> """
con.predict_triple(h=0, t=1328, r=0)

#%%
""" <pope_boniface_iv, cause_of_death, aids> """
con.predict_triple(h=2900, r=1, t=2893)
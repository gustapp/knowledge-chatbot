import OpenKE.config as config
import OpenKE.models as models
import tensorflow as tf
import numpy as np
import json
from flask import jsonify

""" Load TransE FB13 Knowledge Embedding """
from os import path
root = path.dirname(path.abspath(__file__))

con = config.Config()

fb13_path = path.join(root, 'OpenKE/benchmarks/FB13/')
ke_path =  path.join(root, 'OpenKE/res/model.vec.tf')

con.set_in_path(fb13_path)
con.set_work_threads(4)
con.set_dimension(50)
con.set_import_files(ke_path)
con.init()
con.set_model(models.TransE)

""" Load Dictionaries: 
    * relation2id
    * entity2id
"""
entity2id_path = path.join(root, 'OpenKE/benchmarks/FB13/entity2id.txt')
relation2id_path = path.join(root, 'OpenKE/benchmarks/FB13/relation2id.txt')

import pandas as pd
e2i_df = pd.read_csv(entity2id_path, sep='\t', header=None, skiprows=[0])
e2i_df.columns = ['entity', 'id']
    
r2i_df = pd.read_csv(relation2id_path, sep='\t', header=None, skiprows=[0])
r2i_df.columns = ['relation', 'id']


def sofia_fb13_fulfillment(request, con=con, e2i_df=e2i_df, r2i_df=r2i_df):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    
    parameters = request_json['queryResult']['parameters']

    e_h = parameters['head']
    r = parameters['relation']

    """ What is the religion ? """
    eh_id = e2i_df[e2i_df['entity'] == e_h]['id'].values[0]
    r_id = r2i_df[r2i_df['relation'] == r]['id'].values[0]

    res_id = con.predict_tail_entity(h=eh_id, r=r_id, k=1)

    response = e2i_df[e2i_df['id'] == res_id[0]]['entity']

    """Wrap all return data into JSON"""

    return jsonify({ "fulfillmentText": "{}'s {} is {}".format(e_h, r, str(response.values[0])) })

# if __name__ == "main":
#     """ Runs python 3.7 Cloud Functions locally
#     Conditions:
#         * __main__ : being run directly
#         * main : being run on debugger

#         Flask app wrapper
#     """
#     from flask import Flask, request
#     app = Flask(__name__)

#     @app.route('/sofia_fb13_fullfilment', methods=['GET', 'POST'])
#     def get_sofia_fb13_fulfillment():
#         return sofia_fb13_fulfillment(request)

#     app.run('localhost', 5000, debug=True)

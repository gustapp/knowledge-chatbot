import OpenKE.config as config
import OpenKE.models as models
import tensorflow as tf
import numpy as np
import json
from flask import jsonify
__file__ = './functions/src/'
""" Load TransE FB13 Knowledge Embedding """
from os import path
root = path.dirname(path.abspath(__file__))

con = config.Config()

fb13_path = path.join(root, 'src/OpenKE/benchmarks/FB13/')
ke_path =  path.join(root, 'src/OpenKE/res_fb13/model.vec.tf')

con.set_in_path(fb13_path)
con.set_work_threads(8)
con.set_dimension(100)
con.set_import_files(ke_path)
con.init()
con.set_model(models.TransE)

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

    """ *** Explanation Engine *** """
    
    """ Load Embedding """
    ke = con.get_parameters()

    ke_ent = ke['ent_embeddings']
    ke_rel = ke['rel_embeddings']

    """ head entity vector """
    eh_vec = ke_ent[eh_id]

    """ Generate perturbed set of instances """
    import numpy as np
    
    e = eh_vec
    n_instances = 1000
    dimension = 100
    noise_rate = 0.05

    e_hat = []
    for i in range(0, n_instances):
        noise = np.random.normal(0,noise_rate,dimension)
        e_hat.append(e + noise)

    """ Minimize search area by choosing only the nearest neighbors """
    head_ent = eh_id 
    rel = r_id

    k_nn = 10
    feats_tb = []
    """ discover head entity features """
    for rel_id in range(0,13):
        feat_candidates_per_relation = con.predict_tail_entity(h=head_ent, r=rel_id, k=k_nn)
        feats_tb.append((ke_rel[rel_id], [(ent_id, ke_ent[ent_id]) for ent_id in feat_candidates_per_relation]))

    """ Discover feats noised set """
    e_hat_feats = []
    for rel, k_tails in feats_tb:
        labels = []
        for e_fake in e_hat:
            dist_per_inst = []
            id_per_inst = []
            """ Identify nearest entity to inference """
            for tail_id, tail_cand in k_tails:
                dist = np.mean(abs(e_fake + rel - tail_cand))
                dist_per_inst.append(dist)
                id_per_inst.append(tail_id)
            """ Classify @1 """
            tail = id_per_inst[dist_per_inst.index(min(dist_per_inst))]
            labels.append(tail)
        e_hat_feats.append(labels)
        print(str(len(e_hat_feats)))

    """ Build local dataset """
    feats_names = ['religion', 'cause_of_death', 'place_of_death', 'profession', 'location', 'gender', 'nationality', 'place_of_birth', 'institution', 'children', 'parents', 'spouse', 'ethnicity']
    e_hat_feats_df = pd.DataFrame(data=list(map(list,zip(*e_hat_feats))), columns=feats_names)
    e_hat_feats_df = e_hat_feats_df.applymap(lambda id: e2i_df[e2i_df['id'] == id]['entity'].values[0])
    
    """ *** Interpretable Model *** """
    
    target_rel = r
    label = response

    """ Replace target tail """
    def replace_target(item, label=label):
        if item == label:
            return 1
        else:
            return 0

    df = e_hat_feats_df

    # BROKEN HERE !!!!!
    df[target_rel] = e_hat_feats_df[target_rel].apply(replace_target)
    target = df.pop(target_rel)

    """ Encode labels to categorical features """
    from sklearn.preprocessing import LabelEncoder

    intrp_label = []
    for column in df:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        intrp_label += map(lambda x: '{}:{}'.format(column, x), list(le.classes_))

    """ Encode one hot """ 
    # import numpy as np
    from sklearn.preprocessing import OneHotEncoder

    ohc = OneHotEncoder()
    out = ohc.fit_transform(df)

    """ Full set """
    X = out
    y = target

    """ Naive Bayes """
    from sklearn.metrics import mean_squared_error, accuracy_score
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split

    """ Train Model """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    mnb = MultinomialNB()
    mnb.fit(X_train.toarray(), y_train)

    """ (Log) Accuracy """
    y_pred = mnb.predict(X_test.toarray())
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy_score(y_pred, y_test)))

    """ (Log) Confusion Matrix """
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

    """ Explanation probabilistic reasoning """
    ls_data = []
    ln_data = []
    """ log p(x_i|y'), log p(x_i|y) """
    for log_neg_lkh, log_pos_lkh in np.transpose(mnb.feature_log_prob_):
        neg_lkh = 10 ** log_neg_lkh
        pos_lkh = 10 ** log_pos_lkh

        """ Sufficiency Likelihood Ratio (LS):
            ls = p(x_i|y) / p(x_i|y') 
        """
        ls = pos_lkh / neg_lkh

        """ Necessity Likelihood Ratio (LS):
            ln = p(x_i'|y) / p(x_i'|y') 
        """
        ln = (1 - pos_lkh) / (1 - neg_lkh)

        ls_data.append(ls)
        ln_data.append(ln)

    """ Feature Importance by Likelihood Ratio """
    labels = intrp_label

    exp_df = pd.DataFrame(data={'labels': labels, 'LS': ls_data, 'LN': ln_data})
    exp_df.sort_values('LS', inplace=True, ascending=False)
    
    explanation = exp_df.head(1)

    """Wrap all return data into JSON"""

    return jsonify({ "fulfillmentText": "{}'s {} is {}, because {}".format(e_h, r, str(response.values[0]), explanation.values[0]) })

if __name__ == "main":
    """ Runs python 3.7 Cloud Functions locally
    Conditions:
        * __main__ : being run directly
        * main : being run on debugger

        Flask app wrapper
    """
    from flask import Flask, request
    app = Flask(__name__)

    @app.route('/sofia_fb13_fullfilment', methods=['GET', 'POST'])
    def get_sofia_fb13_fulfillment():
        return sofia_fb13_fulfillment(request)

    app.run('localhost', 5000, debug=True)

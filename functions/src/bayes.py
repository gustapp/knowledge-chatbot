#%%
# Parameters
# jesus
# religion       : judaism (Meh)
# religion       : christian (Great Success)
# cause_of_death : crucifixion
target_rel = 'religion'
label = 'christian'

#%%
# Load data
import pandas as pd

df = pd.read_csv('./jesus_feats.csv', index_col=0)
df.head()

#%%
# Replace target tail
def replace_target(item, label=label):
    if item == label:
        return 1
    else:
        return 0

df[target_rel] = df[target_rel].apply(replace_target)
target = df.pop(target_rel)

df.head()

#%%
# Encode labels to categorical features
from sklearn.preprocessing import LabelEncoder

intrp_label = []
for column in df:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    intrp_label += map(lambda x: '{}:{}'.format(column, x), list(le.classes_))

df.head()

# #%%
# # Split target
# target = df.pop(target_rel)

#%%
# Encode one hot 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

ohc = OneHotEncoder()
out = ohc.fit_transform(df)

#%%
# Full set
X = out
y = target

#%%
# Naive Bayes
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
mnb = MultinomialNB()
mnb.fit(X_train.toarray(), y_train)

#%%
y_pred = mnb.predict(X_test.toarray())
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy_score(y_pred, y_test)))

#%%
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#%%
# Explanation probabilistic reasoning
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

#%%
# Feature Importance
# weights = mnb.coef_
labels = intrp_label

exp_df = pd.DataFrame(data={'labels': labels, 'LS': ls_data, 'LN': ln_data})
exp_df.sort_values('LS', inplace=True, ascending=False)
exp_df.head(50)
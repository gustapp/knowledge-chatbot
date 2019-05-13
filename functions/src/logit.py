#%%
# Parameters
target_rel = 'religion'
label = 'judaism'

#%%
# Load data
import pandas as pd

df = pd.read_csv('./full_topzera.csv', index_col=0)
df.head()

#%%
# Replace target tail
def replace_target(item, label=label):
    if item == label:
        return 1
    else:
        return 0

df[target_rel] = df[target_rel].apply(replace_target)
df.head()

#%%
# Encode labels to categorical features
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df = df.apply(le.fit_transform)
df.head()

#%%
# Split target
target = df.pop(target_rel)

#%%
# Encode one hot 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

ohc = OneHotEncoder()
out = ohc.fit_transform(df)

# Recover original
np.array([ohc.active_features_[col] for col in out.sorted_indices().indices]).reshape(1000, 13 - 1) - ohc.feature_indices_[:-1]

#%%
# Full set
X = out
y = target

#%%
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#%%
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#%%
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

